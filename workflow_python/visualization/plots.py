"""
Visualization utilities for aviation anomaly prediction experiments.

Generates publication-quality plots for:
  - Cross-validation accuracy curves
  - Multi-metric comparison tables/charts
  - Confusion matrices
  - ROC curves
  - MC Dropout uncertainty distributions and calibration
  - Grad-CAM temporal heatmaps
  - LIME/SHAP segment importance plots
  - Training time comparisons
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
})


# ── Cross-Validation Accuracy ──────────────────────────────────────────────

def plot_fold_accuracies(train_accuracies, test_accuracies, model_name='Model',
                         epochs=None, save_path=None):
    """
    Plot per-fold training and validation accuracy curves.

    Args:
        train_accuracies: List of lists, shape (n_folds, n_epochs).
        test_accuracies: List of lists, shape (n_folds, n_epochs).
        model_name: Name for the plot title.
        epochs: Number of epochs (auto-detected if None).
        save_path: Optional path to save the figure.
    """
    if epochs is None:
        epochs = len(train_accuracies[0])
    epoch_range = range(1, epochs + 1)

    colors = ['#2196F3', '#4CAF50', '#FF5722', '#9C27B0', '#FF9800']
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
        color = colors[i % len(colors)]
        ax.plot(epoch_range, train_acc, label=f'Fold {i+1} Train',
                color=color, linestyle='-', alpha=0.8)
        ax.plot(epoch_range, test_acc, label=f'Fold {i+1} Val',
                color=color, linestyle='--', alpha=0.8)

    ax.set_title(f'{model_name} — Training & Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.4, 1.05])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ── Multi-Metric Comparison ───────────────────────────────────────────────

def plot_metric_comparison(results_dict, metric='accuracy', split='test',
                           save_path=None):
    """
    Bar chart comparing a metric across models and checkpoints.

    Args:
        results_dict: Dict mapping experiment_name → result dict with 'averages'.
        metric: Metric key (accuracy, f1, auc_roc, precision, recall).
        split: 'train' or 'test'.
        save_path: Optional path to save the figure.
    """
    names = []
    means = []
    stds = []

    for exp_name, result in results_dict.items():
        avg = result['averages']
        names.append(exp_name)
        means.append(avg[split][f'{metric}_mean'])
        stds.append(avg[split][f'{metric}_std'])

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 6))
    x = np.arange(len(names))

    bars = ax.bar(x, means, yerr=stds, capsize=4, color='#2196F3', alpha=0.8,
                  edgecolor='white', linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(f'{split.title()} {metric.replace("_", " ").title()}')
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ── Confusion Matrices ────────────────────────────────────────────────────

def plot_confusion_matrices(results_dict, save_path=None):
    """
    Plot confusion matrices for all experiments in a grid.

    Args:
        results_dict: Dict mapping experiment_name → result dict with per_fold data.
        save_path: Optional path to save the figure.
    """
    n = len(results_dict)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (exp_name, result) in enumerate(results_dict.items()):
        ax = axes[idx]
        # Aggregate confusion matrix across folds
        cm_total = np.zeros((2, 2))
        for fold in result['per_fold']:
            cm_total += np.array(fold['test']['confusion_matrix'])
        cm_total = cm_total.astype(int)

        im = ax.imshow(cm_total, cmap='Blues', aspect='auto')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm_total[i, j]),
                        ha='center', va='center', fontsize=14, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Anomaly'])
        ax.set_yticklabels(['Normal', 'Anomaly'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(exp_name, fontsize=10)

    # Hide empty subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Confusion Matrices (Aggregated)', fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ── ROC Curves ─────────────────────────────────────────────────────────────

def plot_roc_curves(results_dict, save_path=None):
    """
    Plot ROC curves for each experiment.

    Requires 'predictions' in per_fold data.
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))

    for idx, (exp_name, result) in enumerate(results_dict.items()):
        # Use last fold's predictions
        last_fold = result['per_fold'][-1]
        if 'predictions' not in last_fold:
            continue
        y_true = np.array(last_fold['predictions']['test_true'])
        y_prob = np.array(last_fold['predictions']['test_preds'])

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, color=colors[idx], linewidth=2,
                label=f'{exp_name} (AUC={roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ── MC Dropout Uncertainty ─────────────────────────────────────────────────

def plot_uncertainty_distribution(uncertainty_correct, uncertainty_incorrect,
                                  model_name='Model', save_path=None):
    """
    Histogram of epistemic uncertainty for correct vs. incorrect predictions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(uncertainty_correct, bins=50, alpha=0.6, label='Correct',
            color='#4CAF50', density=True)
    ax.hist(uncertainty_incorrect, bins=50, alpha=0.6, label='Incorrect',
            color='#F44336', density=True)

    ax.set_xlabel('Epistemic Uncertainty (Variance)')
    ax.set_ylabel('Density')
    ax.set_title(f'{model_name} — Uncertainty Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_calibration(fraction_of_positives, mean_predicted_value,
                     ece=None, model_name='Model', save_path=None):
    """Reliability (calibration) diagram."""
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(mean_predicted_value, fraction_of_positives, 's-',
            color='#2196F3', linewidth=2, markersize=8, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    title = f'{model_name} — Calibration'
    if ece is not None:
        title += f' (ECE={ece:.4f})'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


def plot_rejection_curve(coverages, accuracies, model_name='Model',
                         save_path=None):
    """Accuracy vs. coverage (rejection) curve."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(coverages, accuracies, color='#2196F3', linewidth=2)
    ax.set_xlabel('Coverage (Fraction of Data Retained)')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_name} — Uncertainty Rejection Curve')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ── Grad-CAM Heatmaps ─────────────────────────────────────────────────────

def plot_gradcam_heatmap(heatmap, raw_signal=None, timesteps=None,
                          model_name='Model', save_path=None):
    """
    Plot a Grad-CAM temporal heatmap, optionally overlaid on a raw signal.

    Args:
        heatmap: 1D array of shape (timesteps,) with importance scores.
        raw_signal: Optional 1D array of a representative feature.
        timesteps: X-axis labels (default: 0..N).
        model_name: Name for the title.
    """
    if timesteps is None:
        timesteps = np.arange(len(heatmap))

    fig, ax1 = plt.subplots(figsize=(14, 4))

    # Heatmap as colored background
    ax1.imshow(heatmap.reshape(1, -1), aspect='auto', cmap='hot',
               extent=[0, len(heatmap), ax1.get_ylim()[0], ax1.get_ylim()[1]],
               alpha=0.6)
    ax1.set_xlabel('Timestep')
    ax1.set_title(f'{model_name} — Grad-CAM Temporal Heatmap')

    if raw_signal is not None:
        ax2 = ax1.twinx()
        ax2.plot(timesteps, raw_signal, color='white', alpha=0.8, linewidth=1.5)
        ax2.set_ylabel('Signal Value')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ── Segment Importance (LIME/SHAP) ────────────────────────────────────────

def plot_segment_importance(importance, segments, method='LIME',
                            model_name='Model', save_path=None):
    """
    Bar chart of segment-level importance scores from LIME or SHAP.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(importance))
    labels = [f'{s[0]}-{s[1]}' for s in segments]

    colors = ['#4CAF50' if v >= 0 else '#F44336' for v in importance]
    ax.bar(x, importance, color=colors, alpha=0.8, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Temporal Segment (timestep range)')
    ax.set_ylabel('Importance Score')
    ax.set_title(f'{model_name} — {method} Segment Importance')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


# ── Training Time Comparison ──────────────────────────────────────────────

def plot_training_times(results_dict, save_path=None):
    """Bar chart of average training times across experiments."""
    names = list(results_dict.keys())
    times = [r['averages']['training_time_s_mean'] for r in results_dict.values()]

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 1.2), 6))
    x = np.arange(len(names))
    ax.bar(x, times, color='#FF9800', alpha=0.8, edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Average Training Time per Model')
    ax.grid(True, alpha=0.3, axis='y')

    for i, t in enumerate(times):
        ax.text(i, t + 0.5, f'{t:.1f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
