"""
Perturbation-Based Explainability: LIME and SHAP for Time Series.

Model-agnostic module using segment-level perturbation for computational efficiency.
Groups adjacent timesteps into segments and treats each segment as a feature
for LIME's local surrogate or SHAP's Shapley value computation.

References:
    - Ribeiro et al., "Why Should I Trust You? LIME" (2016)
    - Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions (SHAP)" (2017)
"""

import numpy as np
import warnings


def create_segments(timesteps, n_segments=10):
    """
    Divide timesteps into approximately equal segments.

    Args:
        timesteps: Total number of timesteps (e.g., 81).
        n_segments: Number of segments to create.

    Returns:
        segments: List of (start, end) tuples for each segment.
    """
    segment_size = timesteps // n_segments
    remainder = timesteps % n_segments
    segments = []
    start = 0
    for i in range(n_segments):
        end = start + segment_size + (1 if i < remainder else 0)
        segments.append((start, end))
        start = end
    return segments


def perturb_segments(X_sample, segments, perturbation_mask, baseline=None):
    """
    Apply segment-level perturbation to a time series sample.

    Args:
        X_sample: Single sample, shape (timesteps, features).
        segments: List of (start, end) tuples from create_segments().
        perturbation_mask: Binary array of length n_segments (1=keep, 0=perturb).
        baseline: Baseline value to replace perturbed segments (default: zeros).

    Returns:
        perturbed: Perturbed sample, shape (timesteps, features).
    """
    perturbed = X_sample.copy()
    if baseline is None:
        baseline = np.zeros_like(X_sample)

    for seg_idx, (start, end) in enumerate(segments):
        if perturbation_mask[seg_idx] == 0:
            perturbed[start:end, :] = baseline[start:end, :]

    return perturbed


def _create_segment_predictor(model, segments, sample_shape):
    """
    Create a prediction function that operates on segment-level binary masks.

    Args:
        model: Keras model (or wrapper with .model/.predict attribute).
        segments: List of (start, end) tuples.
        sample_shape: Shape of a single sample (timesteps, features).

    Returns:
        predict_fn: Function mapping (n_perturbations, n_segments) -> (n_perturbations, 2).
    """
    def predict_fn(segment_masks, reference_sample=None):
        """
        Predict for each perturbation mask.

        Args:
            segment_masks: Binary array, shape (n_perturbations, n_segments).
            reference_sample: The sample to perturb, shape (timesteps, features).
        """
        if reference_sample is None:
            raise ValueError("reference_sample must be provided")

        n_perturbations = segment_masks.shape[0]
        perturbed_batch = np.zeros((n_perturbations,) + tuple(sample_shape))

        for i in range(n_perturbations):
            perturbed_batch[i] = perturb_segments(
                reference_sample, segments, segment_masks[i]
            )

        # Get predictions
        keras_model = getattr(model, 'model', model)
        preds = keras_model.predict(perturbed_batch, verbose=0).flatten()

        # Return probabilities for both classes (required by LIME)
        return np.column_stack([1 - preds, preds])

    return predict_fn


def lime_explain(model, X_sample, n_segments=10, n_perturbations=500, seed=42):
    """
    Compute LIME explanation for a single time series sample.

    Uses segment-level perturbation: groups of adjacent timesteps are either
    kept or replaced with zeros, and a local linear model is fit to the
    resulting predictions.

    Args:
        model: Keras model (or wrapper).
        X_sample: Single sample, shape (timesteps, features).
        n_segments: Number of temporal segments.
        n_perturbations: Number of perturbed samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        dict with keys:
            'segment_importance': Array of shape (n_segments,) with importance scores.
            'segments': List of (start, end) tuples.
            'intercept': Intercept of the local linear model.
            'r_squared': R² score of the local linear model fit.
            'prediction': Original model prediction for this sample.
    """
    np.random.seed(seed)

    if len(X_sample.shape) == 3:
        X_sample = X_sample[0]  # Remove batch dim

    timesteps, features = X_sample.shape
    segments = create_segments(timesteps, n_segments)

    # Generate perturbation masks (binary vectors)
    masks = np.random.binomial(1, 0.5, size=(n_perturbations, n_segments))

    # Always include the original (all ones) and fully perturbed (all zeros)
    masks[0] = np.ones(n_segments)
    masks[1] = np.zeros(n_segments)

    # Create perturbed samples and get predictions
    perturbed_batch = np.zeros((n_perturbations, timesteps, features))
    for i in range(n_perturbations):
        perturbed_batch[i] = perturb_segments(X_sample, segments, masks[i])

    keras_model = getattr(model, 'model', model)
    preds = keras_model.predict(perturbed_batch, verbose=0).flatten()

    # Compute distances (cosine similarity between masks)
    original_mask = np.ones(n_segments)
    distances = np.sqrt(np.sum((masks - original_mask) ** 2, axis=1))
    # Kernel weights (exponential kernel)
    kernel_width = 0.25 * n_segments
    weights = np.exp(-(distances ** 2) / (2 * kernel_width ** 2))

    # Fit weighted linear regression
    from sklearn.linear_model import Ridge
    lr = Ridge(alpha=1.0)
    lr.fit(masks, preds, sample_weight=weights)

    # Original prediction
    original_pred = keras_model.predict(
        X_sample[np.newaxis, :, :], verbose=0
    ).flatten()[0]

    # R² score
    r_squared = lr.score(masks, preds, sample_weight=weights)

    return {
        'segment_importance': lr.coef_,
        'segments': segments,
        'intercept': lr.intercept_,
        'r_squared': r_squared,
        'prediction': original_pred,
    }


def shap_explain(model, X_sample, X_background, n_segments=10):
    """
    Compute SHAP values for a single time series sample using KernelSHAP.

    Uses segment-level representation for computational efficiency.

    Args:
        model: Keras model (or wrapper).
        X_sample: Single sample, shape (timesteps, features).
        X_background: Background dataset for SHAP, shape (n_bg, timesteps, features).
                     Use a small representative subset (e.g., 50-100 samples).
        n_segments: Number of temporal segments.

    Returns:
        dict with keys:
            'shap_values': SHAP values per segment, shape (n_segments,).
            'segments': List of (start, end) tuples.
            'base_value': Expected model output on background data.
            'prediction': Original model prediction for this sample.
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "The 'shap' package is required. Install with: pip install shap"
        )

    if len(X_sample.shape) == 3:
        X_sample = X_sample[0]

    timesteps, features = X_sample.shape
    segments = create_segments(timesteps, n_segments)

    # Convert background data to segment representation
    def sample_to_segments(X, segs):
        """Average each segment to create segment-level features."""
        n = X.shape[0] if len(X.shape) == 3 else 1
        if len(X.shape) == 2:
            X = X[np.newaxis, :, :]
        seg_features = np.zeros((n, len(segs)))
        for j, (s, e) in enumerate(segs):
            seg_features[:, j] = np.mean(np.abs(X[:, s:e, :]), axis=(1, 2))
        return seg_features

    bg_segments = sample_to_segments(X_background, segments)
    sample_segments = sample_to_segments(X_sample, segments)

    # Create prediction function for segment-level inputs
    def segment_predict(segment_masks):
        """Maps segment importance masks to model predictions."""
        n = segment_masks.shape[0]
        full_inputs = np.zeros((n, timesteps, features))

        for i in range(n):
            sample = X_sample.copy()
            for j, (s, e) in enumerate(segments):
                # Scale segment by the mask value relative to background
                if segment_masks[i, j] < np.mean(bg_segments[:, j]):
                    sample[s:e, :] = 0.0  # Zero out this segment
            full_inputs[i] = sample

        keras_model = getattr(model, 'model', model)
        return keras_model.predict(full_inputs, verbose=0).flatten()

    # Run KernelSHAP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.KernelExplainer(segment_predict, bg_segments)
        shap_values = explainer.shap_values(sample_segments, nsamples=200)

    # Original prediction
    keras_model = getattr(model, 'model', model)
    original_pred = keras_model.predict(
        X_sample[np.newaxis, :, :], verbose=0
    ).flatten()[0]

    return {
        'shap_values': shap_values.flatten() if isinstance(shap_values, np.ndarray) else np.array(shap_values).flatten(),
        'segments': segments,
        'base_value': float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0]),
        'prediction': original_pred,
    }


def batch_lime_explain(model, X_samples, n_segments=10, n_perturbations=500,
                       max_samples=100):
    """
    Run LIME on a batch of samples.

    Args:
        model: Keras model.
        X_samples: Multiple samples, shape (n_samples, timesteps, features).
        n_segments: Number of temporal segments.
        n_perturbations: Perturbations per sample.
        max_samples: Maximum number of samples to explain (for efficiency).

    Returns:
        dict with keys:
            'all_importances': shape (n_explained, n_segments).
            'mean_importance': shape (n_segments,).
            'std_importance': shape (n_segments,).
            'segments': List of (start, end) tuples.
    """
    n_samples = min(len(X_samples), max_samples)
    indices = np.random.choice(len(X_samples), n_samples, replace=False)

    all_importances = []
    for i, idx in enumerate(indices):
        if (i + 1) % 20 == 0:
            print(f"  LIME: Explaining sample {i+1}/{n_samples}")
        result = lime_explain(model, X_samples[idx], n_segments=n_segments,
                              n_perturbations=n_perturbations, seed=idx)
        all_importances.append(result['segment_importance'])

    all_importances = np.array(all_importances)

    return {
        'all_importances': all_importances,
        'mean_importance': np.mean(all_importances, axis=0),
        'std_importance': np.std(all_importances, axis=0),
        'segments': result['segments'],
    }


def batch_shap_explain(model, X_samples, X_background, n_segments=10,
                       max_samples=50):
    """
    Run SHAP on a batch of samples.

    Args:
        model: Keras model.
        X_samples: Multiple samples, shape (n_samples, timesteps, features).
        X_background: Background data, shape (n_bg, timesteps, features).
        n_segments: Number of temporal segments.
        max_samples: Maximum number of samples to explain.

    Returns:
        dict with same structure as batch_lime_explain.
    """
    n_samples = min(len(X_samples), max_samples)
    indices = np.random.choice(len(X_samples), n_samples, replace=False)

    all_shap_values = []
    for i, idx in enumerate(indices):
        if (i + 1) % 10 == 0:
            print(f"  SHAP: Explaining sample {i+1}/{n_samples}")
        result = shap_explain(model, X_samples[idx], X_background,
                              n_segments=n_segments)
        all_shap_values.append(result['shap_values'])

    all_shap_values = np.array(all_shap_values)

    return {
        'all_importances': all_shap_values,
        'mean_importance': np.mean(all_shap_values, axis=0),
        'std_importance': np.std(all_shap_values, axis=0),
        'segments': result['segments'],
    }
