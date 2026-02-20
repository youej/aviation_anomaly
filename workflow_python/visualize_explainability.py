import json
import numpy as np
import os
import sys

# Ensure we can import from local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt

from visualization.plots import (
    plot_uncertainty_distribution,
    plot_calibration,
    plot_segment_importance,
    plot_gradcam_heatmap
)

# Configuration
RESULTS_FILE = '../results/baseline_full/inception_time_cp100.json'

print(f"Loading results from {RESULTS_FILE}...")
if not os.path.exists(RESULTS_FILE):
    print(f"Error: {RESULTS_FILE} not found.")
    print("Please ensure you are running this from the 'workflow_python' directory")
    print("and that you have run 'main.py' to generate the results.")
    exit(1)

with open(RESULTS_FILE, 'r') as f:
    data = json.load(f)

# Extract explainability data
# The JSON structure is the result dictionary itself, not wrapped in a key
explain_data = data.get('explainability', {})

if not explain_data:
    print("No explainability data found in JSON! Did you run main.py with --all-explain?")
    print("Keys found in JSON:", list(data.keys()))
    exit(1)

# 1. MC Dropout Plots
mc = explain_data.get('mc_dropout', {})
if mc:
    print("Generating MC Dropout plots...")
    
    unc_correct = mc.get('uncertainty_correct')
    unc_incorrect = mc.get('uncertainty_incorrect')
    
    if unc_correct and unc_incorrect:
        plot_uncertainty_distribution(
            np.array(unc_correct), 
            np.array(unc_incorrect),
            model_name='InceptionTime',
            save_path='inception_uncertainty_dist.png'
        )
        print(" -> Saved inception_uncertainty_dist.png")
    else:
        print("Warning: Full uncertainty arrays not found. Re-run main.py.")
        
    # Calibration
    cal = mc.get('calibration', {})
    if cal:
        plot_calibration(
            np.array(cal['fraction_of_positives']),
            np.array(cal['mean_predicted_value']),
            ece=mc.get('ece'),
            model_name='InceptionTime',
            save_path='inception_calibration.png'
        )
        print(" -> Saved inception_calibration.png")

# 2. LIME Plots
lime = explain_data.get('lime', {})
if lime:
    print("Generating LIME plots...")
    importance = np.array(lime['mean_importance'])
    # 10 segments for 81 timesteps
    segments = [(i*8, (i+1)*8) for i in range(10)]
    segments[-1] = (72, 81)
    
    plot_segment_importance(
        importance, 
        segments, 
        method='LIME',
        model_name='InceptionTime',
        save_path='inception_lime.png'
    )
    print(" -> Saved inception_lime.png")

# 3. SHAP Plots
shap_data = explain_data.get('shap', {})
if shap_data:
    print("Generating SHAP plots...")
    importance = np.array(shap_data['mean_importance'])
    segments = [(i*8, (i+1)*8) for i in range(10)]
    segments[-1] = (72, 81)
    
    plot_segment_importance(
        importance, 
        segments, 
        method='SHAP',
        model_name='InceptionTime',
        save_path='inception_shap.png'
    )
    print(" -> Saved inception_shap.png")

# 4. Grad-CAM Plots
grad = explain_data.get('gradient', {})
if grad:
    print("Generating Grad-CAM heatmap...")
    # New key name in main.py is 'heatmap_mean'
    heatmap_mean = grad.get('heatmap_mean')
    
    if heatmap_mean:
        plot_gradcam_heatmap(
            np.array(heatmap_mean),
            timesteps=np.arange(len(heatmap_mean)),
            model_name='InceptionTime',
            save_path='inception_gradcam.png'
        )
        print(" -> Saved inception_gradcam.png")
    else:
        print("Warning: 'heatmap_mean' not found. Re-run main.py.")

print("\nAll Done!")
