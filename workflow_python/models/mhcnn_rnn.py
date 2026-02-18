"""
MultiHeadCnnRnn (MHCNN-RNN) — Per-feature CNN heads + GRU with MIL aggregation.

Source: Bleu Laine et al., 2022.
Architecture: Per-feature Conv1D heads → concatenate → GRU → MIL max-aggregation.
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, GRU, Dense, TimeDistributed,
    Lambda, Flatten, Reshape, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class MultiHeadCnnRnn:
    def __init__(self, input_shape, kernel_sizes=(8, 5, 3), filters=(16, 32, 64),
                 learning_rate=0.001, weight_decay=0.01,
                 dropout=0.1, recurrent_dropout=0.1,
                 kernel_regularizer=None, recurrent_regularizer=None):
        self.input_shape = input_shape
        self.kernel_sizes = kernel_sizes
        self.filters = filters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.model = self.create_model()

    def create_model(self):
        input_layer = Input(shape=self.input_shape)

        # Multi-Head CNN: process each feature channel separately
        cnn_outputs = []
        for i in range(self.input_shape[1]):
            head = Reshape((self.input_shape[0], 1))(input_layer[:, :, i:i+1])
            for j in range(len(self.kernel_sizes)):
                head = Conv1D(filters=self.filters[j],
                              kernel_size=self.kernel_sizes[j],
                              activation='relu',
                              kernel_regularizer=l2(self.weight_decay))(head)
                head = BatchNormalization()(head)
            head = Conv1D(filters=1, kernel_size=1, padding='same',
                          activation='sigmoid',
                          kernel_regularizer=l2(self.weight_decay))(head)
            cnn_outputs.append(head)
        concatenated = concatenate(cnn_outputs)

        # GRU + MIL aggregation
        gru_layer = GRU(20, dropout=self.dropout,
                        recurrent_dropout=self.recurrent_dropout,
                        return_sequences=True,
                        kernel_regularizer=self.kernel_regularizer,
                        recurrent_regularizer=self.recurrent_regularizer)(concatenated)
        td_dense = TimeDistributed(Dense(100, activation='tanh',
                                         kernel_regularizer=self.kernel_regularizer))(gru_layer)
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

    def fit(self, x, y, epochs=30, batch_size=128, validation_data=None):
        return self.model.fit(x, y, epochs=epochs, batch_size=batch_size,
                              validation_data=validation_data)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, filepath):
        self.model.save(filepath)
