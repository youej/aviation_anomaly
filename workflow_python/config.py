"""
Central configuration for aviation anomaly prediction experiments.

All hyperparameters and constants in one place. No more hardcoded values
scattered across notebooks.
"""

from tensorflow.keras.regularizers import l2


# ── Data Configuration ─────────────────────────────────────────────────────

DATA_DIR = '../data/'                      # Relative to workflow_python/
RESULTS_DIR = '../results/'                # Where experiment results are saved
MAX_TIMESTEPS = 81                         # Maximum number of timesteps per flight
INPUT_SHAPE = (81, 23)                     # (timesteps, features) after filtering
NUM_FOLDS = 5                              # Cross-validation folds


# ── Checkpoint Configuration ───────────────────────────────────────────────

CHECKPOINTS = [25, 50, 75, 100]            # Percentage of flight completion

# Percentage maps for different scenarios
CHECKPOINT_MAPS = {
    100: {(100, 100): 1},                  # Full flight
    75:  {(75, 75): 1},                    # 75% checkpoint
    50:  {(50, 50): 1},                    # 50% checkpoint
    25:  {(25, 25): 1},                    # 25% checkpoint
    'mixed': {                              # Mixed training set
        (25, 50): 0.25,
        (50, 75): 0.25,
        (75, 100): 0.25,
        (100, 100): 0.25,
    },
}


# ── Training Configuration ────────────────────────────────────────────────

EPOCHS = 50                                # Max epochs (early stopping will cut short)
BATCH_SIZE = 128
RANDOM_SEED = 42                           # For reproducibility

# Early stopping: stop training if val_loss doesn't improve for `patience` epochs
EARLY_STOPPING = {
    'monitor': 'val_loss',
    'patience': 5,
    'restore_best_weights': True,           # Roll back to best epoch
    'verbose': 1,
}

# Reduce learning rate when val_loss plateaus
REDUCE_LR = {
    'monitor': 'val_loss',
    'factor': 0.5,                          # Halve the LR
    'patience': 3,                          # Wait 3 epochs before reducing
    'min_lr': 1e-6,
    'verbose': 1,
}


# ── GRU Hyperparameters ───────────────────────────────────────────────────

GRU_CONFIG = {
    'learning_rate': 0.002,
    'dropout': 0.1,
    'recurrent_dropout': 0.1,
    'kernel_regularizer': l2(0.01),
    'recurrent_regularizer': l2(0.01),
}


# ── CNN Hyperparameters (shared by MHCNN-RNN and MultiHeadAttention) ──────

CNN_CONFIG = {
    'kernel_sizes': [8, 5, 3],
    'filters': [16, 32, 64],
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'dropout': 0.1,
    'recurrent_dropout': 0.1,
    'kernel_regularizer': l2(0.01),
    'recurrent_regularizer': l2(0.01),
}


# ── Transformer Hyperparameters (GTN) ─────────────────────────────────────

GTN_CONFIG = {
    'conv_filters': 32,
    'conv_kernel_size': 3,
    'embedding_dim': 64,
    'transformer_layers': 1,
    'num_heads': 4,
    'dropout': 0.2,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
}


# ── InceptionTime Hyperparameters ─────────────────────────────────────────

INCEPTION_CONFIG = {
    'nb_filters': 32,
    'depth': 6,
    'kernel_sizes': [10, 20, 40],
    'bottleneck_size': 32,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'dropout': 0.2,
}


# ── EARLIEST Hyperparameters ──────────────────────────────────────────────

EARLIEST_CONFIG = {
    'lstm_units': 128,
    'policy_hidden': 64,
    'learning_rate': 1e-3,
    'dropout': 0.2,
    'weight_decay': 1e-4,
    'earliness_weight': 0.5,
}


# ── Explainability Configuration ──────────────────────────────────────────

MC_DROPOUT_CONFIG = {
    'n_forward_passes': 50,
    'batch_size': 256,
}

GRADCAM_CONFIG = {
    'max_samples': 200,        # Max samples to generate heatmaps for
}

LIME_CONFIG = {
    'n_segments': 10,
    'n_perturbations': 500,
    'max_samples': 100,
}

SHAP_CONFIG = {
    'n_segments': 10,
    'n_background': 50,        # Background samples for KernelSHAP
    'max_samples': 50,
}


# ── Model Registry ────────────────────────────────────────────────────────

def get_model_factories(input_shape=INPUT_SHAPE):
    """
    Return a dict mapping model_key → (model_factory, metadata).

    model_factory is a callable that returns a fresh model instance.
    This is the recommended way to create models — ensures clean
    initialization for each fold.
    """
    from models import (BaseGRU, MultiHeadCnnRnn, MultiHeadAttention,
                        GatedTransformerNetwork, InceptionTime, EARLIEST)

    return {
        'base_gru': {
            'name': 'Base GRU (DT-MIL)',
            'tier': 'Aviation Baseline',
            'has_conv': False,
            'factory': lambda: BaseGRU(
                input_shape=input_shape, **GRU_CONFIG
            ),
        },
        'mhcnn_rnn': {
            'name': 'MHCNN-RNN',
            'tier': 'Aviation Baseline',
            'has_conv': True,
            'factory': lambda: MultiHeadCnnRnn(
                input_shape=input_shape, **CNN_CONFIG
            ),
        },
        'attn_block1': {
            'name': 'MultiHead Attention (Block 1)',
            'tier': 'Cross-Domain SOTA',
            'has_conv': True,
            'factory': lambda: MultiHeadAttention(
                input_shape=input_shape, **CNN_CONFIG,
                attention_block_type='block1'
            ),
        },
        'attn_block2': {
            'name': 'MultiHead Attention (Block 2)',
            'tier': 'Cross-Domain SOTA',
            'has_conv': True,
            'factory': lambda: MultiHeadAttention(
                input_shape=input_shape, **CNN_CONFIG,
                attention_block_type='block2'
            ),
        },
        'gtn': {
            'name': 'Gated Transformer Network',
            'tier': 'Cross-Domain SOTA',
            'has_conv': True,
            'factory': lambda: GatedTransformerNetwork(
                input_shape=input_shape, **GTN_CONFIG
            ),
        },
        'inception_time': {
            'name': 'InceptionTime',
            'tier': 'Cross-Domain SOTA',
            'has_conv': True,
            'factory': lambda: InceptionTime(
                input_shape=input_shape, **INCEPTION_CONFIG
            ),
        },
        'earliest': {
            'name': 'EARLIEST',
            'tier': 'Early Classification',
            'has_conv': False,
            'factory': lambda: EARLIEST(
                input_shape=input_shape, **EARLIEST_CONFIG
            ),
        },
    }
