"""
EARLIEST: Adaptive-Halting Policy Network for Early Classification of Time Series.

Based on: Hartvigsen et al., "Adaptive-Halting Policy Network for Early Classification" (2019).
Uses an LSTM encoder paired with a reinforcement learning-based halting policy network
that learns when to stop observing and make a classification decision.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Lambda, Concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class HaltingPolicy(tf.keras.layers.Layer):
    """
    Policy network that decides whether to halt (stop observing) at each timestep.

    Outputs a halting probability at each timestep based on the LSTM hidden state.
    Uses the REINFORCE algorithm for training the discrete halt/continue decision.
    """

    def __init__(self, hidden_dim=64, **kwargs):
        super(HaltingPolicy, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim

    def build(self, input_shape):
        self.dense1 = Dense(self.hidden_dim, activation='relu')
        self.dense2 = Dense(1, activation='sigmoid')  # Halt probability
        super(HaltingPolicy, self).build(input_shape)

    def call(self, hidden_state, training=None):
        x = self.dense1(hidden_state)
        halt_prob = self.dense2(x)  # (batch, 1) — probability of halting
        return halt_prob


class EARLIEST:
    """
    EARLIEST model for early time series classification.

    Architecture:
    - LSTM encoder processes the time series step by step
    - At each timestep, a policy network decides whether to halt
    - A classifier head makes predictions from the LSTM hidden state
    - Training uses a composite loss: classification + earliness penalty

    For the checkpoint-based evaluation in this study, EARLIEST is evaluated
    both with its learned halting policy AND at forced checkpoint percentages,
    allowing direct comparison with other models.

    Compatible with the existing model interface (fit/evaluate/predict/save).
    """

    def __init__(self, input_shape, lstm_units=128, policy_hidden=64,
                 learning_rate=1e-3, dropout=0.2, weight_decay=1e-4,
                 earliness_weight=0.5, n_classes=1):
        self.input_shape = input_shape  # (81, 23)
        self.lstm_units = lstm_units
        self.policy_hidden = policy_hidden
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.earliness_weight = earliness_weight
        self.n_classes = n_classes

        # Build components
        self.encoder = self._build_encoder()
        self.classifier = self._build_classifier()
        self.policy = HaltingPolicy(hidden_dim=policy_hidden)
        self.model = self._build_full_model()

    def _build_encoder(self):
        """LSTM encoder that processes time series step by step."""
        return LSTM(
            self.lstm_units,
            return_sequences=True,
            return_state=True,
            dropout=self.dropout,
            recurrent_dropout=self.dropout,
            kernel_regularizer=l2(self.weight_decay),
            recurrent_regularizer=l2(self.weight_decay)
        )

    def _build_classifier(self):
        """Classification head operating on LSTM hidden state."""
        return tf.keras.Sequential([
            Dense(64, activation='relu', kernel_regularizer=l2(self.weight_decay)),
            Dropout(self.dropout),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(self.weight_decay))
        ])

    def _build_full_model(self):
        """
        Build the end-to-end model for standard Keras training.

        For simplicity and compatibility with the existing pipeline,
        this builds a model that:
        1. Runs the LSTM encoder over the full sequence
        2. At each timestep, computes a halt probability and classification
        3. Uses the halt probabilities to compute a weighted prediction
        """
        inputs = Input(shape=self.input_shape)  # (batch, timesteps, features)

        # Run LSTM over full sequence
        lstm_outputs, final_h, final_c = self.encoder(inputs)
        # lstm_outputs shape: (batch, timesteps, lstm_units)

        # Compute halt probabilities and predictions at each timestep
        # We'll use a custom training step, but the model structure is needed
        # for the standard interface

        # For the standard predict/evaluate interface, use final hidden state
        x = Dropout(self.dropout)(final_h)
        predictions = self.classifier(x)

        model = Model(inputs=inputs, outputs=predictions)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        return model

    def get_halting_points(self, X, threshold=0.5):
        """
        Compute halting points for each sample using the learned policy.

        Returns:
            halt_times: array of shape (n_samples,) with the timestep index
                        at which the policy chose to halt for each sample.
            halt_predictions: array of predicted probabilities at the halting point.
        """
        batch_size = X.shape[0]
        timesteps = X.shape[1]

        halt_times = np.full(batch_size, timesteps - 1)  # Default: use full sequence
        halt_predictions = np.zeros(batch_size)

        # Process step by step through the sequence
        # Initialize LSTM states
        h = tf.zeros((batch_size, self.lstm_units))
        c = tf.zeros((batch_size, self.lstm_units))
        halted = np.zeros(batch_size, dtype=bool)

        for t in range(timesteps):
            # Get input at timestep t
            x_t = X[:, t:t+1, :]  # (batch, 1, features)

            # Run one step of LSTM
            lstm_out, h, c = self.encoder(x_t, initial_state=[h, c])

            # Get halt probability
            halt_prob = self.policy(h).numpy().flatten()

            # Get classification at this timestep
            pred = self.classifier(h).numpy().flatten()

            # Update halting decisions
            newly_halted = (~halted) & (halt_prob >= threshold)
            halt_times[newly_halted] = t
            halt_predictions[newly_halted] = pred[newly_halted]
            halted = halted | newly_halted

            if halted.all():
                break

        # For samples that never halted, use the final prediction
        halt_predictions[~halted] = pred[~halted]

        return halt_times, halt_predictions

    def train_with_policy(self, train_X, train_y, epochs=30, batch_size=128,
                          validation_data=None):
        """
        Train with the composite loss: classification + earliness penalty.

        This is the EARLIEST-specific training that trains both the classifier
        and the halting policy jointly.
        """
        optimizer = Adam(learning_rate=self.learning_rate)
        timesteps = train_X.shape[1]

        train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        train_dataset = train_dataset.shuffle(10000).batch(batch_size)

        history = {'loss': [], 'accuracy': [], 'avg_halt_time': []}
        if validation_data is not None:
            history['val_loss'] = []
            history['val_accuracy'] = []

        for epoch in range(epochs):
            epoch_loss = []
            epoch_acc = []
            epoch_halt_times = []

            for batch_x, batch_y in train_dataset:
                loss, acc, avg_halt = self._train_step(
                    batch_x, batch_y, optimizer, timesteps
                )
                epoch_loss.append(loss.numpy())
                epoch_acc.append(acc.numpy())
                epoch_halt_times.append(avg_halt.numpy())

            avg_loss = np.mean(epoch_loss)
            avg_acc = np.mean(epoch_acc)
            avg_halt = np.mean(epoch_halt_times)

            history['loss'].append(avg_loss)
            history['accuracy'].append(avg_acc)
            history['avg_halt_time'].append(avg_halt)

            # Validation
            if validation_data is not None:
                val_x, val_y = validation_data
                val_preds = self.model.predict(val_x, verbose=0)
                val_loss = tf.keras.losses.binary_crossentropy(
                    val_y.flatten(), val_preds.flatten()
                ).numpy().mean()
                val_acc = np.mean((val_preds.flatten() > 0.5).astype(int) == val_y.flatten())
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - "
                      f"acc: {avg_acc:.4f} - avg_halt: {avg_halt:.1f}/{timesteps} - "
                      f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - "
                      f"acc: {avg_acc:.4f} - avg_halt: {avg_halt:.1f}/{timesteps}")

        return history

    @tf.function
    def _train_step(self, batch_x, batch_y, optimizer, timesteps):
        """Single training step with composite loss."""
        batch_size_actual = tf.shape(batch_x)[0]

        with tf.GradientTape() as tape:
            # Run LSTM over full sequence
            lstm_outputs, final_h, final_c = self.encoder(batch_x, training=True)

            # Compute halt probabilities at each timestep
            halt_probs = []
            predictions = []
            for t in range(timesteps):
                h_t = lstm_outputs[:, t, :]  # (batch, lstm_units)
                halt_prob = self.policy(h_t, training=True)
                pred = self.classifier(h_t, training=True)
                halt_probs.append(halt_prob)
                predictions.append(pred)

            # Stack: (timesteps, batch, 1) -> (batch, timesteps, 1)
            halt_probs = tf.stack(halt_probs, axis=1)  # (batch, timesteps, 1)
            predictions = tf.stack(predictions, axis=1)  # (batch, timesteps, 1)

            # Compute halting distribution (geometric-like)
            # P(halt at t) = halt_prob[t] * prod(1 - halt_prob[s] for s < t)
            halt_probs_squeezed = tf.squeeze(halt_probs, axis=-1)  # (batch, timesteps)
            # Clamp for numerical stability
            halt_probs_clamped = tf.clip_by_value(halt_probs_squeezed, 1e-7, 1.0 - 1e-7)

            log_survive = tf.math.cumsum(
                tf.math.log(1.0 - halt_probs_clamped), axis=1, exclusive=True
            )
            halting_distribution = halt_probs_clamped * tf.exp(log_survive)
            # (batch, timesteps)

            # Weighted prediction: sum over timesteps of halting_dist * prediction
            predictions_squeezed = tf.squeeze(predictions, axis=-1)  # (batch, timesteps)
            weighted_pred = tf.reduce_sum(
                halting_distribution * predictions_squeezed, axis=1
            )  # (batch,)

            # Classification loss
            batch_y_flat = tf.cast(tf.reshape(batch_y, [-1]), tf.float32)
            cls_loss = tf.keras.losses.binary_crossentropy(batch_y_flat, weighted_pred)

            # Earliness penalty: expected halting time / total timesteps
            time_indices = tf.cast(tf.range(timesteps), tf.float32)
            expected_halt_time = tf.reduce_sum(
                halting_distribution * time_indices[tf.newaxis, :], axis=1
            )
            earliness_loss = tf.reduce_mean(expected_halt_time) / tf.cast(timesteps, tf.float32)

            # Composite loss
            total_loss = tf.reduce_mean(cls_loss) + self.earliness_weight * earliness_loss

        # Update all trainable variables
        all_vars = (self.encoder.trainable_variables +
                    self.classifier.trainable_variables +
                    self.policy.trainable_variables)
        gradients = tape.gradient(total_loss, all_vars)
        optimizer.apply_gradients(zip(gradients, all_vars))

        # Accuracy
        pred_binary = tf.cast(weighted_pred > 0.5, tf.float32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_binary, batch_y_flat), tf.float32))

        avg_halt = tf.reduce_mean(expected_halt_time)

        return total_loss, accuracy, avg_halt

    # Standard interface methods (compatible with existing pipeline)
    def summary(self):
        return self.model.summary()

    def fit(self, x, y, epochs=30, batch_size=128, validation_data=None):
        """
        Standard fit interface.

        Uses the composite training (classifier + policy) internally,
        then returns a history-like object compatible with the existing pipeline.
        """
        history = self.train_with_policy(x, y, epochs=epochs, batch_size=batch_size,
                                         validation_data=validation_data)
        # Wrap in a Keras-compatible history object
        class HistoryWrapper:
            def __init__(self, hist_dict):
                self.history = hist_dict
        return HistoryWrapper(history)

    def evaluate(self, x, y):
        return self.model.evaluate(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, filepath):
        self.model.save(filepath)
