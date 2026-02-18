"""
Gated Transformer Network (GTN) — Dual transformer towers with learned gating.

Source: Liu et al., 2021.
Architecture:
  - Conv1D preprocessing → positional encoding
  - Step-wise (temporal) transformer tower → feature vector S
  - Channel-wise transformer tower → feature vector C
  - Learned soft gating: fused = g1*S + g2*C
  - Sigmoid classification
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dense, TimeDistributed,
    Dropout, Permute, Lambda, Activation, GlobalAveragePooling1D,
    LayerNormalization, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class PositionalEncodingLayer(tf.keras.layers.Layer):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model, max_len, **kwargs):
        super(PositionalEncodingLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        pos = tf.cast(tf.range(seq_len)[:, tf.newaxis], tf.float32)
        i = tf.cast(tf.range(self.d_model)[tf.newaxis, :], tf.float32)
        angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(self.d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return inputs + pos_encoding

    def compute_output_shape(self, input_shape):
        return input_shape


class GatedTransformerNetwork:
    def __init__(self, input_shape, conv_filters=32, conv_kernel_size=3,
                 embedding_dim=64, transformer_layers=1, num_heads=4,
                 dropout=0.2, learning_rate=1e-4, weight_decay=1e-4):
        self.input_shape = input_shape  # (81, 23)
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.embedding_dim = embedding_dim
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = self.create_model()

    def create_model(self):
        inputs = Input(shape=self.input_shape)

        # --- Convolutional Preprocessing ---
        conv_out = Conv1D(filters=self.conv_filters,
                          kernel_size=self.conv_kernel_size,
                          padding='same', activation='relu',
                          kernel_regularizer=l2(self.weight_decay))(inputs)
        conv_out = BatchNormalization()(conv_out)
        preprocessed = TimeDistributed(
            Dense(self.embedding_dim, activation='tanh')
        )(conv_out)

        # --- Step-wise (Temporal) Transformer Tower ---
        pos_encoded = PositionalEncodingLayer(
            self.embedding_dim, self.input_shape[0]
        )(preprocessed)
        step_tower = pos_encoded
        for _ in range(self.transformer_layers):
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.embedding_dim,
                dropout=self.dropout
            )(step_tower, step_tower)
            attn_output = Dropout(self.dropout)(attn_output)
            step_tower = LayerNormalization(epsilon=1e-6)(step_tower + attn_output)
            ffn = Dense(self.embedding_dim * 4, activation='relu',
                        kernel_regularizer=l2(self.weight_decay))(step_tower)
            ffn = Dense(self.embedding_dim,
                        kernel_regularizer=l2(self.weight_decay))(ffn)
            ffn = Dropout(self.dropout)(ffn)
            step_tower = LayerNormalization(epsilon=1e-6)(step_tower + ffn)
        step_features = GlobalAveragePooling1D()(step_tower)
        S = Dense(self.embedding_dim, activation='tanh',
                  kernel_regularizer=l2(self.weight_decay))(step_features)

        # --- Channel-wise Transformer Tower ---
        channel_input = Permute((2, 1))(inputs)  # (features, timesteps)
        channel_embed = Dense(self.embedding_dim, activation='tanh',
                              kernel_regularizer=l2(self.weight_decay))(channel_input)
        channel_tower = channel_embed
        for _ in range(self.transformer_layers):
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=self.num_heads, key_dim=self.embedding_dim,
                dropout=self.dropout
            )(channel_tower, channel_tower)
            attn_output = Dropout(self.dropout)(attn_output)
            channel_tower = LayerNormalization(epsilon=1e-6)(channel_tower + attn_output)
            ffn = Dense(self.embedding_dim * 4, activation='relu',
                        kernel_regularizer=l2(self.weight_decay))(channel_tower)
            ffn = Dense(self.embedding_dim,
                        kernel_regularizer=l2(self.weight_decay))(ffn)
            ffn = Dropout(self.dropout)(ffn)
            channel_tower = LayerNormalization(epsilon=1e-6)(channel_tower + ffn)
        channel_features = GlobalAveragePooling1D()(channel_tower)
        C = Dense(self.embedding_dim, activation='tanh',
                  kernel_regularizer=l2(self.weight_decay))(channel_features)

        # --- Gating Mechanism ---
        gating_input = concatenate([S, C])
        gating = Dense(2, kernel_regularizer=l2(self.weight_decay))(gating_input)
        gating = Activation('softmax')(gating)
        g1 = Lambda(lambda x: x[:, 0:1])(gating)
        g2 = Lambda(lambda x: x[:, 1:2])(gating)
        S_weighted = Lambda(lambda x: x[0] * x[1])([S, g1])
        C_weighted = Lambda(lambda x: x[0] * x[1])([C, g2])
        fused = concatenate([S_weighted, C_weighted])

        # --- Classification ---
        outputs = Dense(1, activation='sigmoid',
                        kernel_regularizer=l2(self.weight_decay))(fused)

        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    def summary(self):
        return self.model.summary()

    def fit(self, x, y, epochs=30, batch_size=128, validation_data=None, **kwargs):
        return self.model.fit(x, y, epochs=epochs, batch_size=batch_size,
                              validation_data=validation_data, **kwargs)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, filepath):
        self.model.save(filepath)
