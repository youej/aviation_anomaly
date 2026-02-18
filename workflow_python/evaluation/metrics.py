"""
Cross-validation with comprehensive metrics for aviation anomaly prediction.

Replaces both the old model_eval.ipynb (only tracked accuracy/loss, didn't
reinitialize models) and the colab_realtime.ipynb cross() function (used
fragile isinstance checks for model reinitialization).

Now uses a model_factory callable for clean per-fold reinitialization.
"""

import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def cross_validate(model_factory, train_x_list, train_y_list,
                   test_x_list, test_y_list,
                   num_folds=5, epochs=30, batch_size=128,
                   verbose=True):
    """
    K-fold cross-validation with comprehensive metrics.

    Uses a model_factory callable to create a fresh model for each fold,
    avoiding weight carryover between folds.

    Args:
        model_factory: Callable that returns a new model instance.
            Example: lambda: BaseGRU(input_shape=(81, 23), ...)
        train_x_list: List of training data per fold.
        train_y_list: List of training labels per fold.
        test_x_list: List of test data per fold.
        test_y_list: List of test labels per fold.
        num_folds: Number of folds to use.
        epochs: Training epochs per fold.
        batch_size: Batch size for training.
        verbose: Whether to print progress.

    Returns:
        dict with keys:
            'per_fold': List of per-fold metric dicts.
            'averages': Averaged metrics across folds.
            'epoch_histories': Per-epoch train/test accuracy per fold.
    """
    per_fold = []
    epoch_train_accs = []
    epoch_test_accs = []

    for fold in range(num_folds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold+1}/{num_folds}")
            print(f"{'='*60}")

        train_X = train_x_list[fold]
        train_y = train_y_list[fold]
        test_X = test_x_list[fold]
        test_y = test_y_list[fold]

        # Fresh model per fold — critical to avoid weight leakage
        tf.keras.backend.clear_session()
        model = model_factory()

        # Count parameters
        keras_model = getattr(model, 'model', model)
        n_params = keras_model.count_params()

        # --- Train ---
        t_start = time.time()
        history = model.fit(
            train_X, train_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test_X, test_y)
        )
        training_time = time.time() - t_start

        # Store epoch-level histories
        epoch_train_accs.append(history.history['accuracy'])
        epoch_test_accs.append(history.history.get('val_accuracy',
                                                    history.history['accuracy']))

        # --- Predict with timing ---
        t_start = time.time()
        train_preds_prob = model.predict(train_X).flatten()
        test_preds_prob = model.predict(test_X).flatten()
        inference_time = time.time() - t_start

        train_preds = (train_preds_prob > 0.5).astype(int)
        test_preds = (test_preds_prob > 0.5).astype(int)
        train_y_flat = train_y.flatten()
        test_y_flat = test_y.flatten()

        # --- Compute metrics ---
        fold_result = {
            'fold': fold + 1,
            'n_params': int(n_params),
            'training_time_s': round(training_time, 2),
            'inference_time_s': round(inference_time, 4),
            'train': _compute_metrics(train_y_flat, train_preds, train_preds_prob,
                                      history.history['loss'][-1]),
            'test': _compute_metrics(test_y_flat, test_preds, test_preds_prob,
                                     history.history.get('val_loss',
                                                          history.history['loss'])[-1]),
        }
        fold_result['test']['confusion_matrix'] = confusion_matrix(
            test_y_flat, test_preds
        ).tolist()

        per_fold.append(fold_result)

        if verbose:
            t = fold_result['train']
            v = fold_result['test']
            print(f"\n  Train — Acc: {t['accuracy']:.4f}  F1: {t['f1']:.4f}  "
                  f"AUC: {t['auc_roc']:.4f}")
            print(f"  Test  — Acc: {v['accuracy']:.4f}  F1: {v['f1']:.4f}  "
                  f"AUC: {v['auc_roc']:.4f}")
            print(f"  Time: {fold_result['training_time_s']:.1f}s  "
                  f"Params: {n_params:,}")

    # --- Averages ---
    averages = _compute_averages(per_fold)

    if verbose:
        a = averages
        print(f"\n{'='*60}")
        print(f"AVERAGE ({num_folds} folds)")
        print(f"{'='*60}")
        print(f"  Test Accuracy:  {a['test']['accuracy_mean']:.4f} ± "
              f"{a['test']['accuracy_std']:.4f}")
        print(f"  Test F1:        {a['test']['f1_mean']:.4f} ± "
              f"{a['test']['f1_std']:.4f}")
        print(f"  Test AUC-ROC:   {a['test']['auc_roc_mean']:.4f} ± "
              f"{a['test']['auc_roc_std']:.4f}")
        print(f"  Test Precision: {a['test']['precision_mean']:.4f} ± "
              f"{a['test']['precision_std']:.4f}")
        print(f"  Test Recall:    {a['test']['recall_mean']:.4f} ± "
              f"{a['test']['recall_std']:.4f}")
        print(f"  Avg Train Time: {a['training_time_s_mean']:.1f}s")
        print(f"  Parameters:     {per_fold[0]['n_params']:,}")

    return {
        'per_fold': per_fold,
        'averages': averages,
        'epoch_histories': {
            'train_accuracy': epoch_train_accs,
            'test_accuracy': epoch_test_accs,
        },
    }


def _compute_metrics(y_true, y_pred, y_prob, loss):
    """Compute all classification metrics for one split."""
    return {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'recall': round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        'f1': round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'auc_roc': round(float(roc_auc_score(y_true, y_prob)), 4),
        'loss': round(float(loss), 4),
    }


def _compute_averages(per_fold):
    """Compute mean ± std across folds."""
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'loss']
    averages = {'train': {}, 'test': {}}

    for split in ['train', 'test']:
        for key in metric_keys:
            values = [f[split][key] for f in per_fold]
            averages[split][f'{key}_mean'] = round(float(np.mean(values)), 4)
            averages[split][f'{key}_std'] = round(float(np.std(values)), 4)

    # Timing averages
    averages['training_time_s_mean'] = round(
        float(np.mean([f['training_time_s'] for f in per_fold])), 2
    )
    averages['inference_time_s_mean'] = round(
        float(np.mean([f['inference_time_s'] for f in per_fold])), 4
    )
    averages['n_params'] = per_fold[0]['n_params']

    return averages
