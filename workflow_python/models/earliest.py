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
        # Small weight init on dense1 so pre-activation to dense2 stays
        # close to 0, preserving the -3.0 bias effect at initialization.
        self.dense1 = Dense(self.hidden_dim, activation='relu',
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
        # Bias initialized to -3 so initial halt prob ≈ sigmoid(-3) ≈ 0.05.
        # Without this, halt prob starts at ~0.5, causing the geometric
        # halting distribution to collapse to timestep 0 immediately.
        self.dense2 = Dense(1, activation='sigmoid',
                            bias_initializer=tf.keras.initializers.Constant(-3.0),
                            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
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
                 earliness_weight=0.001, n_classes=1):
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

    def _predict_with_policy(self, X):
        """
        Compute predictions using the same halting distribution math as _train_step.

        This is the correct method to use for validation — it mirrors the training
        computation exactly (weighted sum of per-timestep predictions by the halting
        distribution), so val_loss is on the same scale as train_loss.

        Unlike get_halting_points(), this does NOT use a hard threshold, so it
        works correctly even when halt probs are small (e.g., during warmup).
        """
        timesteps = X.shape[1]
        X_tensor = tf.cast(X, tf.float32)

        lstm_outputs, _, _ = self.encoder(X_tensor, training=False)

        halt_probs_list = []
        preds_list = []
        for t in range(timesteps):
            h_t = lstm_outputs[:, t, :]
            halt_probs_list.append(self.policy(h_t, training=False))
            preds_list.append(self.classifier(h_t, training=False))

        halt_probs = tf.squeeze(tf.stack(halt_probs_list, axis=1), axis=-1)  # (batch, T)
        predictions = tf.squeeze(tf.stack(preds_list, axis=1), axis=-1)      # (batch, T)

        halt_probs_clamped = tf.clip_by_value(halt_probs, 1e-7, 1.0 - 1e-7)
        log_survive = tf.math.cumsum(
            tf.math.log(1.0 - halt_probs_clamped), axis=1, exclusive=True
        )
        halting_dist = halt_probs_clamped * tf.exp(log_survive)  # (batch, T)

        weighted_pred = tf.reduce_sum(halting_dist * predictions, axis=1)  # (batch,)

        # Also compute expected halt time for reporting
        time_indices = tf.cast(tf.range(timesteps), tf.float32)
        expected_halt = tf.reduce_mean(
            tf.reduce_sum(halting_dist * time_indices[tf.newaxis, :], axis=1)
        )

        return weighted_pred.numpy(), expected_halt.numpy()

    def get_halting_points(self, X, threshold=0.1):
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
                          validation_data=None, warmup_epochs=5):
        """
        Train with the composite loss: classification + earliness penalty.

        This is the EARLIEST-specific training that trains both the classifier
        and the halting policy jointly.

        Uses a warmup strategy to avoid policy collapse:
        - During warmup_epochs, earliness_weight is 0 (pure classification).
          This lets the classifier learn from the full sequence first.
        - After warmup, earliness_weight activates and the policy learns
          when to halt based on a now-competent classifier.
        """
        # clipnorm=1.0: prevents RL gradient spikes from destabilizing the LSTM.
        # Without clipping, a single bad batch during policy learning can produce
        # gradients orders of magnitude larger than normal, causing the LSTM weights
        # to jump and breaking all previously learned representations.
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)
        timesteps = train_X.shape[1]

        # Best-epoch checkpointing. EARLIEST is RL-based, so the last epoch is
        # rarely the best (training curves are jagged). We save encoder+classifier
        # +policy as a tf.train.Checkpoint (not model.save_weights) because they
        # are separate TF objects not registered inside self.model.
        import tempfile, os
        _ckpt_dir = tempfile.mkdtemp(prefix="earliest_best_")
        _ckpt = tf.train.Checkpoint(
            encoder=self.encoder,
            classifier=self.classifier,
            policy=self.policy,
        )
        _ckpt_manager = tf.train.CheckpointManager(_ckpt, _ckpt_dir, max_to_keep=1)
        best_val_acc = -1.0

        train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
        train_dataset = train_dataset.shuffle(10000).batch(batch_size)

        history = {'loss': [], 'accuracy': [], 'avg_halt_time': []}
        if validation_data is not None:
            history['val_loss'] = []
            history['val_accuracy'] = []

        for epoch in range(epochs):
            # Warmup strategy: disable earliness penalty for first N epochs
            # so the classifier learns before the policy starts optimizing
            if epoch < warmup_epochs:
                active_earliness_weight = 0.0
                is_warmup = True
            else:
                active_earliness_weight = self.earliness_weight
                is_warmup = False

            epoch_loss = []
            epoch_acc = []
            epoch_halt_times = []

            for batch_x, batch_y in train_dataset:
                loss, acc, avg_halt = self._train_step(
                    batch_x, batch_y, optimizer, timesteps,
                    active_earliness_weight, is_warmup
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

            # Phase indicator for logging
            phase = "[WARMUP]" if is_warmup else "[POLICY]"

            # Validation
            if validation_data is not None:
                val_x, val_y = validation_data
                # Use _predict_with_policy — mirrors training math exactly.
                # get_halting_points() uses a hard threshold which is never
                # reached when halt probs are small, defaulting everything to
                # t=80 and causing exploding val_loss.
                val_preds, _ = self._predict_with_policy(val_x)
                val_loss = tf.keras.losses.binary_crossentropy(
                    val_y.flatten().astype(np.float32), val_preds.flatten()
                ).numpy().mean()
                val_acc = np.mean((val_preds.flatten() > 0.5).astype(int) == val_y.flatten())
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)

                # Save best epoch weights
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    _ckpt_manager.save()
                    print(f"Epoch {epoch+1}/{epochs} {phase} - loss: {avg_loss:.4f} - "
                          f"acc: {avg_acc:.4f} - avg_halt: {avg_halt:.1f}/{timesteps} - "
                          f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} [BEST ✓]")
                else:
                    print(f"Epoch {epoch+1}/{epochs} {phase} - loss: {avg_loss:.4f} - "
                          f"acc: {avg_acc:.4f} - avg_halt: {avg_halt:.1f}/{timesteps} - "
                          f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} {phase} - loss: {avg_loss:.4f} - "
                      f"acc: {avg_acc:.4f} - avg_halt: {avg_halt:.1f}/{timesteps}")

        # Restore best-epoch weights before returning so evaluate() / predict()
        # use the best checkpoint rather than epoch N's weights.
        if validation_data is not None and _ckpt_manager.latest_checkpoint:
            _ckpt.restore(_ckpt_manager.latest_checkpoint).expect_partial()
            print(f"Restored best weights (val_acc={best_val_acc:.4f}) for testing.")

        return history

    @tf.function
    def _train_step(self, batch_x, batch_y, optimizer, timesteps,
                    active_earliness_weight, is_warmup):
        """
        Single training step with composite loss.

        During warmup (is_warmup=True):
        - earliness_weight is 0 (pure classification)
        - 50% of batches use forced exploration: halt probs are zeroed out,
          forcing the model to observe the full sequence. This ensures the
          classifier sees late-timestep data and learns meaningful features.

        After warmup:
        - Full composite loss: classification + earliness penalty
        - No forced exploration (policy learns freely)
        """
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

            # Forced exploration during warmup: with 50% probability,
            # zero out halt probs to force the model to read the full sequence.
            # This ensures the classifier trains on late-timestep hidden states.
            if is_warmup:
                explore = tf.random.uniform(()) < 0.5
                halt_probs_squeezed = tf.cond(
                    explore,
                    lambda: tf.zeros_like(halt_probs_squeezed) + 1e-7,  # force read all
                    lambda: halt_probs_squeezed
                )

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

            # Composite loss with dynamic weight (0 during warmup)
            total_loss = tf.reduce_mean(cls_loss) + active_earliness_weight * earliness_loss

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

    def fit(self, x, y, epochs=30, batch_size=128, validation_data=None, **kwargs):
        """
        Standard fit interface.

        Uses the composite training (classifier + policy) internally,
        then returns a history-like object compatible with the existing pipeline.

        Note: Keras callbacks (EarlyStopping, ReduceLROnPlateau) are accepted
        via **kwargs but not used — EARLIEST has its own custom training loop.
        """
        history = self.train_with_policy(x, y, epochs=epochs, batch_size=batch_size,
                                         validation_data=validation_data)
        # Wrap in a Keras-compatible history object
        class HistoryWrapper:
            def __init__(self, hist_dict):
                self.history = hist_dict
        return HistoryWrapper(history)

    def evaluate(self, x, y):
        # Must use _predict_with_policy, NOT self.model.evaluate().
        # self.model evaluates at the final hidden state (h_81), but the
        # classifier was trained on the policy's halting state (~h_20).
        preds, _ = self._predict_with_policy(x)
        loss = tf.keras.losses.binary_crossentropy(
            y.flatten().astype(np.float32), preds.flatten()
        ).numpy().mean()
        acc = float(np.mean((preds.flatten() > 0.5).astype(int) == y.flatten()))
        return loss, acc

    def predict(self, x):
        return self.model.predict(x)

    def save(self, filepath):
        self.model.save(filepath)
