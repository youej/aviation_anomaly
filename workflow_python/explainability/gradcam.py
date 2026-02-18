"""
Grad-CAM for 1D Temporal Sequences.

Adapts Gradient-weighted Class Activation Mapping (Grad-CAM) from 2D images
to 1D time series. Highlights which timesteps contributed most to a prediction.

For models without convolutional layers (e.g., BaseGRU), a gradient saliency
fallback is provided.

References:
    - Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks" (2017)
"""

import numpy as np
import tensorflow as tf


def find_last_conv_layer(model):
    """
    Find the last Conv1D layer in a Keras model.

    Args:
        model: Keras model (or wrapper with .model attribute).

    Returns:
        Layer name of the last Conv1D layer, or None if no Conv1D layers exist.
    """
    keras_model = getattr(model, 'model', model)
    last_conv = None
    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            last_conv = layer.name
    return last_conv


def grad_cam_1d(model, input_data, target_layer_name=None):
    """
    Generate Grad-CAM heatmap for 1D temporal input.

    Computes gradient-weighted activation maps from the last Conv1D layer
    to produce a temporal heatmap showing which timesteps are most important
    for the prediction.

    Args:
        model: Keras model (or wrapper with .model attribute).
        input_data: Single input sample, shape (1, timesteps, features)
                    or batch shape (batch, timesteps, features).
        target_layer_name: Name of the Conv1D layer to use. If None,
                          automatically finds the last Conv1D layer.

    Returns:
        heatmaps: Array of shape (batch, timesteps) with importance scores
                  for each timestep. Higher values = more important.
                  Returns None if no Conv1D layer found (use gradient_saliency instead).
    """
    keras_model = getattr(model, 'model', model)

    # Ensure batch dimension
    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, axis=0)

    input_data = tf.cast(input_data, tf.float32)

    # Find target layer
    if target_layer_name is None:
        target_layer_name = find_last_conv_layer(model)
    if target_layer_name is None:
        return None  # No Conv1D layers — use gradient saliency instead

    # Build gradient model
    target_layer = keras_model.get_layer(target_layer_name)
    grad_model = tf.keras.Model(
        inputs=keras_model.inputs,
        outputs=[target_layer.output, keras_model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(input_data)
        # For binary classification, use the positive class score
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_output)
    # grads shape: (batch, conv_timesteps, filters)

    # Global average pooling over the temporal dimension to get filter weights
    # (importance weight per filter)
    weights = tf.reduce_mean(grads, axis=1)  # (batch, filters)

    # Weighted combination of feature maps
    # conv_output: (batch, conv_timesteps, filters)
    # weights: (batch, filters)
    heatmap = tf.reduce_sum(
        conv_output * weights[:, tf.newaxis, :], axis=-1
    )  # (batch, conv_timesteps)

    # Apply ReLU (keep only positive contributions)
    heatmap = tf.nn.relu(heatmap)

    # Normalize to [0, 1]
    heatmap_max = tf.reduce_max(heatmap, axis=1, keepdims=True)
    heatmap_max = tf.maximum(heatmap_max, 1e-10)  # Avoid division by zero
    heatmap = heatmap / heatmap_max

    heatmap = heatmap.numpy()

    # Resize heatmap to match original input timesteps if needed
    original_timesteps = input_data.shape[1]
    conv_timesteps = heatmap.shape[1]
    if conv_timesteps != original_timesteps:
        # Interpolate to match original timestep count
        from scipy.interpolate import interp1d
        batch_size = heatmap.shape[0]
        resized = np.zeros((batch_size, original_timesteps))
        for b in range(batch_size):
            x_old = np.linspace(0, 1, conv_timesteps)
            x_new = np.linspace(0, 1, original_timesteps)
            f = interp1d(x_old, heatmap[b], kind='linear')
            resized[b] = f(x_new)
        heatmap = resized

    return heatmap


def gradient_saliency(model, input_data):
    """
    Gradient saliency map for models without Conv1D layers (e.g., BaseGRU).

    Computes the absolute gradient of the output with respect to the input
    to identify which timesteps and features are most influential.

    Args:
        model: Keras model (or wrapper with .model attribute).
        input_data: Input data, shape (batch, timesteps, features).

    Returns:
        temporal_saliency: Array of shape (batch, timesteps) — saliency per timestep
                          (aggregated across features).
        full_saliency: Array of shape (batch, timesteps, features) — per-feature saliency.
    """
    keras_model = getattr(model, 'model', model)

    if len(input_data.shape) == 2:
        input_data = np.expand_dims(input_data, axis=0)

    input_tensor = tf.cast(input_data, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_tensor)
        predictions = keras_model(input_tensor)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, input_tensor)
    # grads shape: (batch, timesteps, features)

    # Take absolute value (magnitude of gradient)
    full_saliency = tf.abs(grads).numpy()

    # Aggregate across features to get temporal saliency
    temporal_saliency = np.mean(full_saliency, axis=-1)  # (batch, timesteps)

    # Normalize per sample
    max_vals = temporal_saliency.max(axis=1, keepdims=True)
    max_vals = np.maximum(max_vals, 1e-10)
    temporal_saliency = temporal_saliency / max_vals

    return temporal_saliency, full_saliency


def explain_model(model, input_data):
    """
    Model-agnostic gradient-based explanation.

    Automatically selects Grad-CAM for models with Conv1D layers,
    or gradient saliency for models without.

    Args:
        model: Any model from the benchmark suite.
        input_data: Input data, shape (batch, timesteps, features).

    Returns:
        dict with keys:
            'method': 'grad_cam' or 'gradient_saliency'
            'temporal_heatmap': shape (batch, timesteps)
            'full_saliency': shape (batch, timesteps, features) — only for saliency.
    """
    conv_layer = find_last_conv_layer(model)

    if conv_layer is not None:
        heatmap = grad_cam_1d(model, input_data, target_layer_name=conv_layer)
        return {
            'method': 'grad_cam',
            'temporal_heatmap': heatmap,
            'target_layer': conv_layer,
        }
    else:
        temporal, full = gradient_saliency(model, input_data)
        return {
            'method': 'gradient_saliency',
            'temporal_heatmap': temporal,
            'full_saliency': full,
        }


def aggregate_heatmaps(heatmaps, y_true=None, class_label=None):
    """
    Aggregate Grad-CAM heatmaps across multiple samples.

    Useful for producing a single summary heatmap showing which
    timesteps are generally most important.

    Args:
        heatmaps: Array of shape (n_samples, timesteps).
        y_true: Optional true labels for class-specific aggregation.
        class_label: If provided, only aggregate samples with this label.

    Returns:
        mean_heatmap: Average heatmap across samples.
        std_heatmap: Standard deviation heatmap.
    """
    if y_true is not None and class_label is not None:
        mask = y_true.flatten() == class_label
        heatmaps = heatmaps[mask]

    mean_heatmap = np.mean(heatmaps, axis=0)
    std_heatmap = np.std(heatmaps, axis=0)

    return mean_heatmap, std_heatmap
