"""
Data loading, preprocessing, truncation, and shuffling for aviation anomaly prediction.

Handles:
  - Loading pre-split 5-fold cross-validation data from .npy files
  - Feature exclusion (TAS, GS — redundant with other speed/position features)
  - StandardScaler normalization (fit on train, transform both)
  - Checkpoint-based truncation and zero-padding
  - Synchronized data shuffling
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


# Full parameter list matching the columns in the raw data (25 features)
PARAMETER_LIST = [
    'LATP', 'LONP', 'RALT', 'GS', 'TAS', 'IVV', 'BLAC', 'CTAC', 'FPAC',
    'LATG', 'N1_1', 'FLAP', 'PTCH', 'ROLL', 'AOA1', 'AIL_1', 'ELEV_1',
    'RUDD', 'LOC', 'GLS', 'ALT', 'LATG', 'PTRM', 'LONG', 'OIT_1'
]

# TAS (True Airspeed) and GS (Ground Speed) are excluded because they are
# highly correlated with other features and introduce redundancy.
EXCLUDED_PARAMETERS = ['TAS', 'GS']


def load_data(data_dir):
    """
    Load pre-split 5-fold cross-validation data from .npy files.

    Args:
        data_dir: Path to directory containing train_x_list.npy,
                  train_y_list.npy, test_x_list.npy, test_y_list.npy.

    Returns:
        Tuple of (train_y, test_y, train_X, test_X), each shape (5, n_samples, ...).
    """
    if not data_dir.endswith('/'):
        data_dir += '/'

    print('Loading data ...')
    train_y = np.load(data_dir + 'train_y_list.npy', allow_pickle=True)
    test_y = np.load(data_dir + 'test_y_list.npy', allow_pickle=True)
    train_X = np.load(data_dir + 'train_x_list.npy', allow_pickle=True)
    test_X = np.load(data_dir + 'test_x_list.npy', allow_pickle=True)

    print(f"Data shapes:")
    print(f"  train_X: {train_X.shape}  (per fold: {train_X[0].shape})")
    print(f"  train_y: {train_y.shape}  (per fold: {train_y[0].shape})")
    print(f"  test_X:  {test_X.shape}   (per fold: {test_X[0].shape})")
    print(f"  test_y:  {test_y.shape}   (per fold: {test_y[0].shape})")

    return train_y, test_y, train_X, test_X


def _reshape_for_scaling(data):
    """Flatten 3D (samples, timesteps, features) → 2D for StandardScaler."""
    n_samples, n_timesteps, n_features = data.shape
    return data.reshape(n_samples * n_timesteps, n_features), data.shape


def _unflatten(flat_data, original_shape):
    """Restore 2D → 3D."""
    return flat_data.reshape(original_shape)


def preprocess_data(train_X_list, test_X_list,
                    excluded_params=None, param_list=None):
    """
    Preprocess data: exclude redundant features + standardize.

    Steps:
      1. Remove columns for excluded parameters (default: TAS, GS)
      2. Fit StandardScaler on training data, transform both train and test

    Args:
        train_X_list: Array of training data per fold, shape (n_folds, n_samples, timesteps, features).
        test_X_list: Array of test data per fold.
        excluded_params: List of parameter names to exclude (default: ['TAS', 'GS']).
        param_list: Full ordered parameter list (default: PARAMETER_LIST).

    Returns:
        Tuple of (train_filtered, test_filtered), both lists of arrays.
    """
    if excluded_params is None:
        excluded_params = EXCLUDED_PARAMETERS
    if param_list is None:
        param_list = PARAMETER_LIST

    print('Preprocessing data ...')

    # Compute indices to exclude and keep
    excluded_idx = [param_list.index(tag) for tag in excluded_params
                    if tag in param_list]
    keep_idx = np.sort(list(
        set(range(train_X_list[0].shape[-1])) - set(excluded_idx)
    ))

    # Filter features
    train_filtered = [data[:, :, keep_idx] for data in train_X_list]
    test_filtered = [data[:, :, keep_idx] for data in test_X_list]

    # Standardize per fold (fit on train only)
    for i in range(len(train_filtered)):
        scaler = StandardScaler()

        train_flat, train_shape = _reshape_for_scaling(train_filtered[i])
        test_flat, test_shape = _reshape_for_scaling(test_filtered[i])

        # Fit on train, transform both
        train_filtered[i] = _unflatten(
            scaler.fit_transform(train_flat), train_shape
        )
        test_filtered[i] = _unflatten(
            scaler.transform(test_flat), test_shape
        )

    n_features_after = train_filtered[0].shape[-1]
    print(f"  Features: {train_X_list[0].shape[-1]} → {n_features_after} "
          f"(excluded: {excluded_params})")

    return train_filtered, test_filtered


def truncate_pad_data(data, percentage_map, original_length, dtype=np.float32):
    """
    Truncate sequences to a percentage of their original length, then zero-pad
    back to the original length. This simulates checkpoint-based evaluation.

    Args:
        data: Array of shape (n_samples, timesteps, features).
        percentage_map: Dict mapping (min_pct, max_pct) → proportion of data.
            Example: {(75, 100): 1.0} means all data truncated to 75–100%.
            Example: {(25, 25): 0.25, (50, 50): 0.25, (75, 75): 0.25, (100, 100): 0.25}
        original_length: Number of timesteps in original sequences.
        dtype: Output data type.

    Returns:
        Truncated and padded array of shape (n_samples, original_length, features).
    """
    total_samples = len(data)
    truncated_padded = []
    start_idx = 0

    for (min_pct, max_pct), proportion in percentage_map.items():
        n_samples = int(total_samples * proportion)
        segment = data[start_idx:start_idx + n_samples]

        segment_results = []
        for sequence in segment:
            pct = np.random.randint(min_pct, max_pct + 1)
            trunc_len = max(1, int(original_length * (pct / 100)))

            truncated = sequence[:trunc_len]
            padded = np.pad(
                truncated,
                ((0, original_length - trunc_len), (0, 0)),
                mode='constant', constant_values=0.0
            ).astype(dtype)
            segment_results.append(padded)

        truncated_padded.append(np.array(segment_results, dtype=dtype))
        start_idx += n_samples

    return np.concatenate(truncated_padded, axis=0).astype(dtype)


def shuffle_data(X, y, seed=None):
    """
    Shuffle X and y arrays in synchronized order.

    Args:
        X: Input features, shape (n_samples, ...).
        y: Labels, shape (n_samples, ...).
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple of (shuffled_X, shuffled_y).
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random
    indices = np.arange(len(X))
    rng.shuffle(indices)
    return X[indices], y[indices]
