"""
MultiHeadAttention — MHCNN-RNN with attention mechanism.

Source: Yin et al., 2022.
Two attention variants:
  - Block 1: Feature-level attention (Dense + softmax on feature dimension)
  - Block 2: Temporal attention (Dense + softmax on time dimension)
"""

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, GRU, Dense, TimeDistributed,
    Lambda, Flatten, Reshape, concatenate, Permute, Multiply, RepeatVector
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class MultiHeadAttention:
    def __init__(self, input_shape, kernel_sizes=(8, 5, 3), filters=(16, 32, 64),
                 learning_rate=0.001, weight_decay=0.01,
                 dropout=0.1, recurrent_dropout=0.1,
                 kernel_regularizer=None, recurrent_regularizer=None,
                 attention_block_type='block1'):
        """
        Args:
            attention_block_type: 'block1' for feature-level attention,
                                  'block2' for temporal attention.
        """
        self.input_shape = input_shape
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.attention_block_type = attention_block_type
        self.model = self.create_model()

    def attention_3d_block1(self, inputs, single_attention_vector=False):
        """
        Feature-level attention: learns attention weights over the feature
        dimension at each time step via Dense + softmax.
        """
        input_dim = int(inputs.shape[2])
        a = Dense(input_dim, activation='softmax')(inputs)
        if single_attention_vector:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((1, 2), name='attention_vec')(a)
        output_attention_mul = Multiply()([inputs, a_probs])
        return output_attention_mul

    def attention_3d_block2(self, inputs, single_attention_vector=False):
        """
        Temporal attention: permutes inputs so Dense learns attention
        weights over time steps instead of features.
        """
        time_steps = int(inputs.shape[1])
        input_dim = int(inputs.shape[2])
        a = Permute((2, 1))(inputs)
        a = Dense(time_steps, activation='softmax')(a)
        if single_attention_vector:
            a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
            a = RepeatVector(input_dim)(a)
        a_probs = Permute((2, 1))(a)
        output_attention_mul = Multiply()([inputs, a_probs])
        return output_attention_mul

    def create_model(self):
        input_layer = Input(shape=self.input_shape)

        # Multi-Head CNN: per-feature conv heads
        cnn_outputs = []
        for i in range(self.input_shape[1]):
            head = Reshape((self.input_shape[0], 1))(input_layer[:, :, i:i+1])
            for j in range(len(self.kernel_sizes)):
                head = Conv1D(filters=self.filters[j],
                              kernel_size=self.kernel_sizes[j],
                              activation='relu', padding='same',
                              kernel_regularizer=l2(self.weight_decay))(head)
                head = BatchNormalization()(head)
            head = Conv1D(filters=1, kernel_size=1, padding='same',
                          activation='sigmoid',
                          kernel_regularizer=l2(self.weight_decay))(head)
            cnn_outputs.append(head)
        concatenated = concatenate(cnn_outputs)

        # GRU
        gru_layer = GRU(20, dropout=self.dropout,
                        recurrent_dropout=self.recurrent_dropout,
                        return_sequences=True,
                        kernel_regularizer=self.kernel_regularizer,
                        recurrent_regularizer=self.recurrent_regularizer)(concatenated)

        # Attention
        if self.attention_block_type == 'block1':
            attention_output = self.attention_3d_block1(gru_layer)
        elif self.attention_block_type == 'block2':
            attention_output = self.attention_3d_block2(gru_layer)
        else:
            raise ValueError(f"Unknown attention_block_type: {self.attention_block_type}")

        # Classification head
        td_dense = TimeDistributed(Dense(100, activation='tanh',
                                         kernel_regularizer=self.kernel_regularizer))(attention_output)
        td_output = TimeDistributed(Dense(1, activation='sigmoid',
                                          kernel_regularizer=self.kernel_regularizer))(td_dense)
        flattened = Flatten()(td_output)

        def mil_aggregation(x):
            return tf.reduce_max(x, axis=1, keepdims=True)

        mil_output = Lambda(mil_aggregation, output_shape=(1,))(flattened)

        model = Model(inputs=input_layer, outputs=mil_output)
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])
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
