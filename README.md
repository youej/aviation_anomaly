# Aviation Anomaly Detection

Deep learning benchmark for proactive flight safety anomaly prediction, developed at the Vanderbilt Data Science Institute under Professor Yingxiao Kong.

## Research Contributions

### 1. Proactive Safety Checkpoint System

Rather than analyzing flights post-hoc, this system evaluates model predictions at **25%, 50%, 75%, and 100%** of flight completion. Flight recorder data is truncated to the checkpoint percentage and zero-padded, simulating a real-time deployment where classification decisions must be made _before_ the flight ends. This enables airlines and regulators to intervene mid-flight — e.g., flagging a landing approach as anomalous at 75% completion while corrective action is still possible.

### 2. Explainability Modules

Three complementary explainability methods are integrated into the benchmark, providing both temporal and feature-level transparency:

- **MC Dropout** — Epistemic uncertainty quantification via stochastic forward passes. Answers: _"How confident is the model in this prediction?"_
- **Grad-CAM (1D)** — Gradient-weighted class activation maps adapted for time series. Answers: _"Which timesteps drove this prediction?"_
- **LIME / SHAP** — Perturbation-based segment importance using model-agnostic methods. Answers: _"Which flight phases matter most?"_

This allows stakeholders to not only know _what_ the model predicts, but _why_ — critical for trust in safety-critical aviation applications.

## Models

Seven architectures across three tiers:

| Model | Tier | Key Architecture |
|-------|------|-----------------|
| Base GRU (DT-MIL) | Aviation Baseline | GRU → Dense → MIL max-aggregation |
| MHCNN-RNN | Aviation Baseline | Per-feature Conv1D heads → GRU → MIL |
| MultiHead Attention (Block 1) | Cross-Domain SOTA | MHCNN-RNN + feature-level attention |
| MultiHead Attention (Block 2) | Cross-Domain SOTA | MHCNN-RNN + temporal attention |
| Gated Transformer Network | Cross-Domain SOTA | Dual transformers + soft gating |
| InceptionTime | Cross-Domain SOTA | Multi-scale 1D Inception + residual connections |
| EARLIEST | Early Classification | LSTM encoder + RL halting policy |

## Project Structure

```
aviation_anomaly/
├── data/                          # Pre-split 5-fold .npy files
│   ├── train_x_list.npy
│   ├── train_y_list.npy
│   ├── test_x_list.npy
│   └── test_y_list.npy
├── workflow_python/               # Main codebase
│   ├── models/                    # 7 model architectures
│   │   ├── base_gru.py
│   │   ├── mhcnn_rnn.py
│   │   ├── multi_head_attention.py
│   │   ├── gated_transformer.py
│   │   ├── inception_time.py
│   │   └── earliest.py
│   ├── data/
│   │   └── loader.py              # Loading, preprocessing, truncation
│   ├── evaluation/
│   │   └── metrics.py             # Cross-validation, all metrics
│   ├── explainability/
│   │   ├── mc_dropout.py          # Uncertainty quantification
│   │   ├── gradcam.py             # 1D Grad-CAM + saliency fallback
│   │   └── perturbation.py        # LIME + SHAP
│   ├── visualization/
│   │   └── plots.py               # Publication-quality charts
│   ├── config.py                  # All hyperparameters + model registry
│   └── main.py                    # CLI entry point
├── workflow_colab/                # Legacy Jupyter notebooks (reference only)
├── launcher.ipynb                 # Colab launcher notebook
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x (GPU recommended)
- NumPy, Scikit-learn, Matplotlib

### Option A: Google Colab (Recommended)

Best for GPU access without local setup.

1. Upload the full `aviation_anomaly/` folder to Google Drive
2. Open `launcher.ipynb` in [Google Colab](https://colab.research.google.com)
3. Set the runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
4. Update `PROJECT_ROOT` in cell 2 to your Drive path (default: `/content/drive/MyDrive/Flight-Safety`)
5. Run cells in order — the smoke test (cell 6) verifies everything works before the full benchmark

### Option B: Local

```bash
# Install dependencies
pip install tensorflow numpy scikit-learn matplotlib

# Navigate to codebase
cd workflow_python

# List available models
python main.py --list

# Smoke test (2 epochs, quick run)
python main.py --model base_gru --cp 100 --epochs 2 --folds 2

# Single model, all checkpoints
python main.py --model gtn --cp all --epochs 50

# Full benchmark (all models, all checkpoints)
python main.py --model all --cp all --epochs 50

# Full benchmark with explainability
python main.py --model all --cp all --epochs 50 --all-explain
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Model key or `all` | `base_gru` |
| `--cp` | Checkpoint % or `all` (25, 50, 75, 100) | `100` |
| `--epochs` | Max training epochs | `50` |
| `--folds` | Number of CV folds | `5` |
| `--mc-dropout` | Enable MC Dropout uncertainty | off |
| `--gradcam` | Enable Grad-CAM explanations | off |
| `--lime` | Enable LIME explanations | off |
| `--shap` | Enable SHAP explanations | off |
| `--all-explain` | Enable all explainability methods | off |
| `--data-dir` | Override data directory | `../data/` |
| `--output-dir` | Override results directory | `../results/` |
| `--list` | List available models | — |

### Training Stability

All models use **EarlyStopping** (patience=5, restore best weights) and **ReduceLROnPlateau** (halve LR after 3 epochs of no improvement). This means:
- Simple models (BaseGRU) stop early at ~8–12 epochs
- Complex models (GTN, InceptionTime) use 25–40 epochs
- Each model gets exactly the training it needs

Results are saved as JSON files in the `results/` directory, one per experiment (e.g., `base_gru_cp100.json`).

## Data

The dataset consists of Flight Operations Quality Assurance (FOQA) flight recorder data with 25 sensor parameters recorded over 81 timesteps. Each flight is labeled as **normal** or **anomalous**. The data is pre-split into 5 stratified cross-validation folds.

After preprocessing, 2 redundant features (TAS, GS) are excluded, leaving 23 features per timestep. All features are standardized per-fold (fit on training data only).

## References

- Janakiraman, V. (2018). _Explaining Aviation Safety Incidents Using Deep Temporal Multiple Instance Learning._
- Bleu Laine et al. (2022). _Multi-Head CNN-RNN for Multi-Time Series Anomaly Detection._
- Yin et al. (2022). _Multi-Head Attention for Time Series Classification._
- Liu et al. (2021). _Gated Transformer Networks for Multivariate Time Series Classification._
- Ismail Fawaz et al. (2020). _InceptionTime: Finding AlexNet for Time Series Classification._
- Hartvigsen et al. (2019). _Adaptive-Halting Policy Network for Early Classification of Time Series._
