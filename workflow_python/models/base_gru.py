"""
BaseGRU (DT-MIL) — GRU with Multiple Instance Learning aggregation.

Source: Janakiraman, 2018 — Deep Temporal Multiple Instance Learning.
Architecture: GRU → TimeDistributed Dense → MIL max-aggregation → sigmoid.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU, Dense, TimeDistributed, Lambda, Flatten, Dropout, Masking
)
from tensorflow.keras.optimizers import Adam


class BaseGRU:
    def __init__(self, input_shape, learning_rate=0.002,
                 dropout=0.1, recurrent_dropout=0.1,
                 kernel_regularizer=None, recurrent_regularizer=None):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.kernel_regularizer = kernel_regularizer
        self.recurrent_regularizer = recurrent_regularizer
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=self.input_shape))
        model.add(GRU(20,
                       dropout=self.dropout,
                       recurrent_dropout=self.recurrent_dropout,
                       return_sequences=True,
                       kernel_regularizer=self.kernel_regularizer,
                       recurrent_regularizer=self.recurrent_regularizer))
        model.add(TimeDistributed(Dense(500, activation='tanh',
                                        kernel_regularizer=self.kernel_regularizer)))
        model.add(TimeDistributed(Dense(1, activation='sigmoid',
                                        kernel_regularizer=self.kernel_regularizer)))
        model.add(Flatten())

        def mil_aggregation(x):
            return tf.reduce_max(x, axis=1, keepdims=True)

        model.add(Lambda(mil_aggregation, output_shape=(1,)))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def summary(self):
        return self.model.summary()

    def fit(self, x, y, epochs=10, batch_size=32, validation_data=None):
        return self.model.fit(x, y, epochs=epochs, batch_size=batch_size,
                              validation_data=validation_data)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, filepath):
        self.model.save(filepath)
