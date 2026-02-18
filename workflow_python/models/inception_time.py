"""
InceptionTime Model for Time Series Classification.

Based on: Fawaz et al., "InceptionTime: Finding AlexNet for Time Series Classification" (2020).
Adapts the Inception-v4 architecture to 1D time series using multi-scale convolutional filters
with bottleneck layers and residual connections.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, Add,
    GlobalAveragePooling1D, Dense, MaxPooling1D, Concatenate, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np


class InceptionModule(tf.keras.layers.Layer):
    """Single Inception module with multi-scale 1D convolutions + bottleneck."""

    def __init__(self, nb_filters=32, kernel_sizes=[10, 20, 40],
                 bottleneck_size=32, use_bottleneck=True, weight_decay=1e-4, **kwargs):
        super(InceptionModule, self).__init__(**kwargs)
        self.nb_filters = nb_filters
        self.kernel_sizes = kernel_sizes
        self.bottleneck_size = bottleneck_size
        self.use_bottleneck = use_bottleneck
        self.weight_decay = weight_decay

    def build(self, input_shape):
        # Bottleneck layer (1x1 convolution to reduce dimensionality)
        if self.use_bottleneck and int(input_shape[-1]) > 1:
            self.bottleneck = Conv1D(
                filters=self.bottleneck_size, kernel_size=1, padding='same',
                use_bias=False, kernel_regularizer=l2(self.weight_decay)
            )
        else:
            self.bottleneck = None

        # Multi-scale convolutions
        self.conv_layers = []
        for ks in self.kernel_sizes:
            self.conv_layers.append(
                Conv1D(filters=self.nb_filters, kernel_size=ks, padding='same',
                       use_bias=False, kernel_regularizer=l2(self.weight_decay))
            )

        # MaxPool path
        self.maxpool = MaxPooling1D(pool_size=3, strides=1, padding='same')
        self.conv_maxpool = Conv1D(
            filters=self.nb_filters, kernel_size=1, padding='same',
            use_bias=False, kernel_regularizer=l2(self.weight_decay)
        )

        # BatchNorm + Activation
        self.bn = BatchNormalization()
        self.activation = Activation('relu')

        super(InceptionModule, self).build(input_shape)

    def call(self, inputs, training=None):
        # Apply bottleneck
        if self.bottleneck is not None:
            x = self.bottleneck(inputs)
        else:
            x = inputs

        # Multi-scale convolutions
        conv_outputs = []
        for conv in self.conv_layers:
            conv_outputs.append(conv(x))

        # MaxPool path (from original input, not bottleneck)
        mp = self.maxpool(inputs)
        mp = self.conv_maxpool(mp)
        conv_outputs.append(mp)

        # Concatenate all paths
        concatenated = Concatenate(axis=-1)(conv_outputs)

        # BatchNorm + ReLU
        out = self.bn(concatenated, training=training)
        out = self.activation(out)

        return out


class ShortcutLayer(tf.keras.layers.Layer):
    """Residual shortcut connection with optional 1x1 conv to match dimensions."""

    def __init__(self, weight_decay=1e-4, **kwargs):
        super(ShortcutLayer, self).__init__(**kwargs)
        self.weight_decay = weight_decay

    def build(self, input_shape):
        # input_shape is a list: [shortcut_input_shape, main_input_shape]
        shortcut_channels = input_shape[0][-1]
        main_channels = input_shape[1][-1]

        self.conv = Conv1D(
            filters=main_channels, kernel_size=1, padding='same',
            use_bias=False, kernel_regularizer=l2(self.weight_decay)
        )
        self.bn = BatchNormalization()
        self.activation = Activation('relu')

        super(ShortcutLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        shortcut_input, main_input = inputs
        shortcut = self.conv(shortcut_input)
        shortcut = self.bn(shortcut, training=training)
        added = Add()([shortcut, main_input])
        return self.activation(added)


class InceptionTime:
    """
    InceptionTime model for time series classification.

    Uses stacked Inception modules with residual connections.
    Compatible with the existing model interface (fit/evaluate/predict/save).

    For ensemble behavior (original paper uses 5-model ensemble),
    use InceptionTimeEnsemble instead.
    """

    def __init__(self, input_shape, nb_filters=32, depth=6,
                 kernel_sizes=[10, 20, 40], bottleneck_size=32,
                 learning_rate=1e-3, weight_decay=1e-4, dropout=0.2):
        self.input_shape = input_shape  # (81, 23)
        self.nb_filters = nb_filters
        self.depth = depth
        self.kernel_sizes = kernel_sizes
        self.bottleneck_size = bottleneck_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=self.input_shape)

        x = inputs
        shortcut = inputs

        # Stack Inception modules with residual connections every 3 modules
        for d in range(self.depth):
            x = InceptionModule(
                nb_filters=self.nb_filters,
                kernel_sizes=self.kernel_sizes,
                bottleneck_size=self.bottleneck_size,
                weight_decay=self.weight_decay,
                name=f'inception_{d}'
            )(x)

            # Add residual connection every 3 modules
            if (d + 1) % 3 == 0:
                x = ShortcutLayer(
                    weight_decay=self.weight_decay,
                    name=f'shortcut_{d}'
                )([shortcut, x])
                shortcut = x

        # Global Average Pooling
        x = GlobalAveragePooling1D()(x)

        # Dropout for MC Dropout compatibility
        x = Dropout(self.dropout)(x)

        # Classification head
        outputs = Dense(1, activation='sigmoid',
                        kernel_regularizer=l2(self.weight_decay))(x)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def summary(self):
        return self.model.summary()

    def fit(self, x, y, epochs=30, batch_size=128, validation_data=None):
        return self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, filepath):
        self.model.save(filepath)


class InceptionTimeEnsemble:
    """
    Ensemble of InceptionTime models (original paper uses 5 models).

    Predictions are averaged across ensemble members.
    Each member is initialized with different random weights.
    """

    def __init__(self, input_shape, n_models=5, **kwargs):
        self.input_shape = input_shape
        self.n_models = n_models
        self.kwargs = kwargs
        self.models = [InceptionTime(input_shape=input_shape, **kwargs)
                       for _ in range(n_models)]

    def fit(self, x, y, epochs=30, batch_size=128, validation_data=None):
        histories = []
        for i, m in enumerate(self.models):
            print(f"\nTraining ensemble member {i+1}/{self.n_models}")
            h = m.fit(x, y, epochs=epochs, batch_size=batch_size,
                      validation_data=validation_data)
            histories.append(h)
        return histories

    def predict(self, x):
        predictions = [m.predict(x) for m in self.models]
        return np.mean(predictions, axis=0)

    def evaluate(self, x, y):
        preds = self.predict(x)
        preds_binary = (preds > 0.5).astype(int).flatten()
        accuracy = np.mean(preds_binary == y.flatten())
        return accuracy

    def summary(self):
        return self.models[0].summary()
