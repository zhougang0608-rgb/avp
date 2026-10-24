"""Load development audio and create a fixed training/validation split."""
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from check_data import check_recording
from features import extract_features
from settings import NAMES, MAX_FRAMES


def load_development_data(data_dir, n_filters=26, use_delta=False, use_energy=False):
    """Read the 20 expected training recordings per name; never read test/."""
    if len(NAMES) != 20 or len(set(NAMES)) != 20:
        raise ValueError("Names.txt must contain 20 distinct names.")
    x, y, paths = [], [], []
    for label, name in enumerate(NAMES):
        for index in range(1, 21):
            relative = Path("train") / name / f"{name}_train_{index:02d}.wav"
            path = Path(data_dir) / relative
            if not path.is_file():
                raise FileNotFoundError(f"Missing development recording: {path}")
            problems = check_recording(path)
            if problems:
                raise ValueError(f"{relative}: {'; '.join(problems)}")
            features = extract_features(path, n_filters, use_delta, use_energy)
            width = (13 + int(use_energy)) * (2 if use_delta else 1)
            if features.shape != (MAX_FRAMES, width) or not np.isfinite(features).all():
                raise ValueError(f"Invalid features: {relative}")
            x.append(features)
            y.append(label)
            paths.append(relative.as_posix())
    # Conv2D input axes: recordings, time steps, coefficients, channels.
    return np.asarray(x, dtype=np.float32)[..., np.newaxis], np.asarray(y), paths


def prepare_split(x, y):
    """Fit normalisation on training data only and apply it to both subsets."""
    train_ids, val_ids = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
    x_train, x_val = x[train_ids], x[val_ids]
    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True) + 1e-6
    return ((x_train - mean) / std, (x_val - mean) / std,
            train_ids, val_ids, mean, std)
