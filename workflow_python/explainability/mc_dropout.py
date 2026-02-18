"""
Monte Carlo Dropout for Uncertainty Quantification.

Model-agnostic module that works with any Keras/TF neural network containing Dropout layers.
Performs N stochastic forward passes with dropout active at inference time to estimate
epistemic uncertainty (model uncertainty).

References:
    - Gal & Ghahramani, "Dropout as a Bayesian Approximation" (2016)
"""

import numpy as np
import tensorflow as tf
from sklearn.calibration import calibration_curve


def mc_dropout_predict(model, X, n_forward=50, batch_size=256):
    """
    Perform Monte Carlo Dropout inference.

    Runs N stochastic forward passes with dropout enabled (training=True)
    to approximate the posterior predictive distribution.

    Args:
        model: Keras model (or wrapper with .model attribute) containing Dropout layers.
        X: Input data, shape (n_samples, timesteps, features).
        n_forward: Number of stochastic forward passes (default: 50).
        batch_size: Batch size for each forward pass.

    Returns:
        dict with keys:
            'mean_prediction': Mean predicted probability, shape (n_samples,).
            'epistemic_uncertainty': Variance across forward passes, shape (n_samples,).
            'predictive_entropy': Entropy of mean prediction, shape (n_samples,).
            'confidence_interval_low': 5th percentile prediction, shape (n_samples,).
            'confidence_interval_high': 95th percentile prediction, shape (n_samples,).
            'all_predictions': Raw predictions, shape (n_forward, n_samples).
    """
    # Get the underlying Keras model if wrapped
    keras_model = getattr(model, 'model', model)

    predictions = []
    for i in range(n_forward):
        # Run forward pass with training=True to keep dropout active
        preds = []
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch_pred = keras_model(X[start:end], training=True)
            preds.append(batch_pred.numpy())
        preds = np.concatenate(preds, axis=0).flatten()
        predictions.append(preds)

    predictions = np.array(predictions)  # (n_forward, n_samples)

    # Mean prediction (Bayesian model averaging)
    mean_pred = predictions.mean(axis=0)

    # Epistemic uncertainty (variance across forward passes)
    epistemic_uncertainty = predictions.var(axis=0)

    # Predictive entropy for binary classification
    # H(p) = -[p*log(p) + (1-p)*log(1-p)]
    p = np.clip(mean_pred, 1e-10, 1.0 - 1e-10)
    predictive_entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))

    # Confidence intervals (5th and 95th percentile)
    ci_low = np.percentile(predictions, 5, axis=0)
    ci_high = np.percentile(predictions, 95, axis=0)

    return {
        'mean_prediction': mean_pred,
        'epistemic_uncertainty': epistemic_uncertainty,
        'predictive_entropy': predictive_entropy,
        'confidence_interval_low': ci_low,
        'confidence_interval_high': ci_high,
        'all_predictions': predictions,
    }


def compute_calibration(y_true, mean_predictions, n_bins=10):
    """
    Compute calibration curve for reliability diagrams.

    Args:
        y_true: True binary labels, shape (n_samples,).
        mean_predictions: Mean predicted probabilities from MC Dropout.
        n_bins: Number of bins for calibration curve.

    Returns:
        dict with keys:
            'fraction_of_positives': Fraction of positives per bin.
            'mean_predicted_value': Mean predicted probability per bin.
            'expected_calibration_error': Scalar ECE value.
    """
    y_true = y_true.flatten()
    mean_predictions = mean_predictions.flatten()

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, mean_predictions, n_bins=n_bins, strategy='uniform'
    )

    # Expected Calibration Error (ECE)
    # Weighted average of |accuracy - confidence| per bin
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(len(fraction_of_positives)):
        bin_mask = (mean_predictions >= bin_boundaries[i]) & \
                   (mean_predictions < bin_boundaries[i + 1])
        bin_count = bin_mask.sum()
        if bin_count > 0:
            ece += (bin_count / len(mean_predictions)) * \
                   abs(fraction_of_positives[i] - mean_predicted_value[i])

    return {
        'fraction_of_positives': fraction_of_positives,
        'mean_predicted_value': mean_predicted_value,
        'expected_calibration_error': ece,
    }


def uncertainty_rejection_curve(y_true, mean_predictions, epistemic_uncertainty):
    """
    Compute accuracy vs. coverage (rejection) curve.

    Allows evaluating the quality of uncertainty estimates:
    if uncertainty is well-calibrated, rejecting the most uncertain predictions
    should improve accuracy on the remaining samples.

    Args:
        y_true: True labels, shape (n_samples,).
        mean_predictions: Mean MC Dropout predictions.
        epistemic_uncertainty: Variance from MC Dropout.

    Returns:
        dict with keys:
            'coverages': Fraction of data retained (1.0 = all data).
            'accuracies': Accuracy at each coverage level.
    """
    y_true = y_true.flatten()
    mean_predictions = mean_predictions.flatten()
    epistemic_uncertainty = epistemic_uncertainty.flatten()

    # Sort by uncertainty (ascending — keep most certain first)
    sorted_idx = np.argsort(epistemic_uncertainty)
    y_sorted = y_true[sorted_idx]
    pred_sorted = (mean_predictions[sorted_idx] > 0.5).astype(int)

    n = len(y_true)
    coverages = []
    accuracies = []

    for k in range(1, n + 1):
        coverage = k / n
        acc = np.mean(pred_sorted[:k] == y_sorted[:k])
        coverages.append(coverage)
        accuracies.append(acc)

    return {
        'coverages': np.array(coverages),
        'accuracies': np.array(accuracies),
    }


def uncertainty_analysis(y_true, mc_results):
    """
    Comprehensive uncertainty analysis.

    Args:
        y_true: True labels.
        mc_results: Output dict from mc_dropout_predict().

    Returns:
        dict with analysis results for visualization.
    """
    y_true = y_true.flatten()
    mean_pred = mc_results['mean_prediction']
    uncertainty = mc_results['epistemic_uncertainty']
    entropy = mc_results['predictive_entropy']

    pred_binary = (mean_pred > 0.5).astype(int)
    correct = pred_binary == y_true
    incorrect = ~correct

    return {
        'correct_uncertainty_mean': uncertainty[correct].mean() if correct.any() else 0,
        'incorrect_uncertainty_mean': uncertainty[incorrect].mean() if incorrect.any() else 0,
        'correct_uncertainty_std': uncertainty[correct].std() if correct.any() else 0,
        'incorrect_uncertainty_std': uncertainty[incorrect].std() if incorrect.any() else 0,
        'correct_entropy_mean': entropy[correct].mean() if correct.any() else 0,
        'incorrect_entropy_mean': entropy[incorrect].mean() if incorrect.any() else 0,
        'uncertainty_correct': uncertainty[correct],
        'uncertainty_incorrect': uncertainty[incorrect],
        'calibration': compute_calibration(y_true, mean_pred),
        'rejection_curve': uncertainty_rejection_curve(y_true, mean_pred, uncertainty),
    }
