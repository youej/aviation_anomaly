"""
Simple interactive entry point for the aviation anomaly prediction pipeline.

Replaces the notebook-style workflow. Can be used interactively in
a Python REPL or as a standalone script.

Usage:
    python main.py                         # Run default (Base GRU, 100% checkpoint)
    python main.py --model gtn --cp 75     # Run GTN at 75% checkpoint
    python main.py --model all --cp all    # Full benchmark
"""

import argparse
import json
import os
import sys
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Aviation Anomaly Prediction — Experiment Runner'
    )
    parser.add_argument('--model', type=str, default='base_gru',
                        help='Model key or "all" (run --list to see options)')
    parser.add_argument('--cp', '--checkpoint', type=str, default='100',
                        dest='checkpoint',
                        help='Checkpoint %% or "all" (25, 50, 75, 100)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override default epochs')
    parser.add_argument('--folds', type=int, default=None,
                        help='Override number of folds')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Override results directory')

    # Explainability flags
    parser.add_argument('--mc-dropout', action='store_true')
    parser.add_argument('--gradcam', action='store_true')
    parser.add_argument('--lime', action='store_true')
    parser.add_argument('--shap', action='store_true')
    parser.add_argument('--all-explain', action='store_true',
                        help='Enable all explainability methods')

    # Utility
    parser.add_argument('--list', action='store_true',
                        help='List available models')

    args = parser.parse_args()

    # Import config (deferred to avoid TF import on --list)
    from config import (
        DATA_DIR, RESULTS_DIR, EPOCHS, NUM_FOLDS, RANDOM_SEED,
        CHECKPOINT_MAPS, get_model_factories,
        EARLY_STOPPING, REDUCE_LR,
        MC_DROPOUT_CONFIG, LIME_CONFIG, SHAP_CONFIG, GRADCAM_CONFIG,
    )

    if args.list:
        registry = get_model_factories()
        print(f"\n{'Key':<20} {'Name':<35} {'Tier':<25}")
        print("-" * 80)
        for key, info in registry.items():
            print(f"{key:<20} {info['name']:<35} {info['tier']:<25}")
        return

    # Resolve settings
    data_dir = args.data_dir or DATA_DIR
    output_dir = args.output_dir or RESULTS_DIR
    epochs = args.epochs or EPOCHS
    num_folds = args.folds or NUM_FOLDS

    if args.all_explain:
        args.mc_dropout = args.gradcam = args.lime = args.shap = True

    # Resolve model and checkpoint lists
    registry = get_model_factories()
    models = list(registry.keys()) if args.model == 'all' else [args.model]
    checkpoints = [25, 50, 75, 100] if args.checkpoint == 'all' else [int(args.checkpoint)]

    for m in models:
        if m not in registry:
            print(f"Error: Unknown model '{m}'. Use --list to see options.")
            return

    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──
    from data import load_data, preprocess_data, truncate_pad_data, shuffle_data
    from evaluation import cross_validate

    print(f"\n{'#'*60}")
    print(f"Aviation Anomaly Prediction Benchmark")
    print(f"{'#'*60}")
    print(f"Models: {', '.join(models)}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Epochs: {epochs}, Folds: {num_folds}")

    train_y, test_y, train_X, test_X = load_data(data_dir)
    train_X_filtered, test_X_filtered = preprocess_data(train_X, test_X)
    input_shape = (train_X_filtered[0].shape[1], train_X_filtered[0].shape[2])

    print(f"Input shape (after preprocessing): {input_shape}")

    # ── Run experiments ──
    all_results = {}

    for model_key in models:
        model_info = registry[model_key]
        for cp in checkpoints:
            exp_name = f"{model_key}_cp{cp}"
            print(f"\n{'='*60}")
            print(f"{model_info['name']} @ {cp}%")
            print(f"{'='*60}")

            # Truncate data
            cp_map = CHECKPOINT_MAPS[cp]
            max_ts = train_X_filtered[0].shape[1]

            train_x_cp = []
            train_y_cp = []
            for i in range(num_folds):
                tx = truncate_pad_data(train_X_filtered[i], cp_map, max_ts)
                tx_s, ty_s = shuffle_data(tx, train_y[i])
                train_x_cp.append(tx_s)
                train_y_cp.append(ty_s)

            test_x_cp = []
            test_y_cp = []
            for i in range(num_folds):
                tex = truncate_pad_data(test_X_filtered[i], cp_map, max_ts)
                tex_s, tey_s = shuffle_data(tex, test_y[i])
                test_x_cp.append(tex_s)
                test_y_cp.append(tey_s)

            # Cross-validate with factory
            # Refresh the factory for this input shape
            fresh_registry = get_model_factories(input_shape)
            factory = fresh_registry[model_key]['factory']

            cv_results = cross_validate(
                model_factory=factory,
                train_x_list=train_x_cp,
                train_y_list=train_y_cp,
                test_x_list=test_x_cp,
                test_y_list=test_y_cp,
                num_folds=num_folds,
                epochs=epochs,
                early_stopping_cfg=EARLY_STOPPING,
                reduce_lr_cfg=REDUCE_LR,
                seed=RANDOM_SEED,
            )

            result = {
                'model': model_key,
                'model_name': model_info['name'],
                'checkpoint': cp,
                'cross_validation': cv_results,
            }

            # ── Explainability ──
            if args.mc_dropout or args.gradcam or args.lime or args.shap:
                # Train one more model for explainability on the last fold
                import tensorflow as tf
                tf.keras.backend.clear_session()
                explain_model_instance = factory()
                last = num_folds - 1
                explain_model_instance.fit(
                    train_x_cp[last], train_y_cp[last],
                    epochs=epochs,
                    validation_data=(test_x_cp[last], test_y_cp[last])
                )

                explain_results = {}

                if args.mc_dropout:
                    from explainability import mc_dropout_predict, uncertainty_analysis
                    print("\n--- MC Dropout ---")
                    mc = mc_dropout_predict(
                        explain_model_instance, test_x_cp[last],
                        **MC_DROPOUT_CONFIG
                    )
                    analysis = uncertainty_analysis(test_y_cp[last], mc)
                    explain_results['mc_dropout'] = {
                        'ece': float(analysis['calibration']['expected_calibration_error']),
                        'correct_unc': float(analysis['correct_uncertainty_mean']),
                        'incorrect_unc': float(analysis['incorrect_uncertainty_mean']),
                        # Save full arrays for visualization
                        'uncertainty_correct': analysis['uncertainty_correct'].tolist(),
                        'uncertainty_incorrect': analysis['uncertainty_incorrect'].tolist(),
                        'calibration': {
                            'fraction_of_positives': analysis['calibration']['fraction_of_positives'].tolist(),
                            'mean_predicted_value': analysis['calibration']['mean_predicted_value'].tolist(),
                        },
                    }
                    print(f"  ECE: {explain_results['mc_dropout']['ece']:.4f}")

                if args.gradcam:
                    from explainability import explain_model as explain_fn
                    print("\n--- Grad-CAM ---")
                    n = min(GRADCAM_CONFIG['max_samples'], len(test_x_cp[last]))
                    gc = explain_fn(explain_model_instance, test_x_cp[last][:n])
                    
                    # 'temporal_heatmap' is the key returned by both grad_cam and gradient_saliency
                    heatmap_data = gc['temporal_heatmap']
                    
                    explain_results['gradient'] = {
                        'method': gc['method'],
                        'heatmap_mean': heatmap_data.mean(axis=0).tolist(),
                        'heatmap': heatmap_data.tolist()[:10],  # Save first 10 for detailed check
                    }
                    print(f"  Method: {gc['method']}")

                if args.lime:
                    from explainability import batch_lime_explain
                    print("\n--- LIME ---")
                    lime_r = batch_lime_explain(
                        explain_model_instance, test_x_cp[last],
                        n_segments=LIME_CONFIG['n_segments'],
                        max_samples=LIME_CONFIG['max_samples'],
                    )
                    explain_results['lime'] = {
                        'mean_importance': lime_r['mean_importance'].tolist(),
                    }

                if args.shap:
                    from explainability import batch_shap_explain
                    print("\n--- SHAP ---")
                    bg_idx = np.random.choice(
                        len(train_x_cp[last]),
                        min(SHAP_CONFIG['n_background'], len(train_x_cp[last])),
                        replace=False
                    )
                    shap_r = batch_shap_explain(
                        explain_model_instance, test_x_cp[last],
                        train_x_cp[last][bg_idx],
                        n_segments=SHAP_CONFIG['n_segments'],
                        max_samples=SHAP_CONFIG['max_samples'],
                    )
                    explain_results['shap'] = {
                        'mean_importance': shap_r['mean_importance'].tolist(),
                    }

                result['explainability'] = explain_results

            all_results[exp_name] = result

            # Save per-experiment
            path = os.path.join(output_dir, f'{exp_name}.json')
            with open(path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nSaved: {path}")

    # ── Summary ──
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'CP':<5} {'Acc':<10} {'F1':<10} {'AUC':<10} {'Time(s)':<10}")
    print("-" * 80)
    for exp_name, r in all_results.items():
        a = r['cross_validation']['averages']
        print(f"{r['model_name']:<30} {r['checkpoint']:<5} "
              f"{a['test']['accuracy_mean']:<10.4f} "
              f"{a['test']['f1_mean']:<10.4f} "
              f"{a['test']['auc_roc_mean']:<10.4f} "
              f"{a['training_time_s_mean']:<10.1f}")


if __name__ == '__main__':
    main()
