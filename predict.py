"""Recognise 1-10 WAV files with the selected model and saved training statistics."""
import argparse
import json
from pathlib import Path

import numpy as np

from features import extract_features
from settings import ROOT, NAMES, FEATURE_VERSION, MAX_FRAMES


def load_recogniser():
    folder = ROOT / "models"
    try:
        config = json.loads((folder / "config.json").read_text())
        if config["names"] != NAMES or config["feature_version"] != FEATURE_VERSION:
            raise ValueError("The model does not match the current names or feature version.")
        model_path = folder / config["model_file"]
        with np.load(folder / config["normalisation_file"]) as stats:
            mean, std = stats["mean"], stats["std"]
        width = (13 + int(config["energy"])) * (2 if config["delta"] else 1)
        expected = (MAX_FRAMES, width, 1)
        if mean.shape != (1, 1, width, 1) or std.shape != mean.shape:
            raise ValueError("Saved normalisation dimensions do not match the model features.")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError("Saved normalisation statistics are invalid.")
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
    except (OSError, KeyError) as error:
        raise ValueError("Selected model assets are missing or incomplete. Restore them or run compare.py.") from error
    from tensorflow import keras
    model = keras.models.load_model(model_path)
    if tuple(model.input_shape[1:]) != expected or model.output_shape[-1] != len(NAMES):
        raise ValueError("Saved model input/output dimensions do not match the configuration.")
    return model, config, mean, std


def predict_scores(paths, recogniser=None):
    """Preserve file order; share this exact preprocessing with evaluation and live use."""
    if not 1 <= len(paths) <= 10:
        raise ValueError("Please select 1 to 10 WAV files at a time.")
    files = [Path(path).expanduser() for path in paths]
    for path in files:
        if path.suffix.lower() != ".wav":
            raise ValueError(f"Expected a WAV file: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Recording not found: {path}")
    model, config, mean, std = recogniser if recogniser is not None else load_recogniser()
    features = [extract_features(path, config["mel_filters"], config["delta"], config["energy"])
                for path in files]
    x = np.asarray(features, dtype=np.float32)[..., np.newaxis]
    if tuple(x.shape[1:]) != tuple(model.input_shape[1:]) or not np.isfinite(x).all():
        raise ValueError("Recording features do not match the selected model.")
    scores = model.predict((x - mean) / std, verbose=0)
    if scores.shape != (len(files), len(NAMES)) or not np.isfinite(scores).all():
        raise ValueError("The model returned invalid prediction scores.")
    return scores


def recognise(paths):
    scores = predict_scores(paths)
    labels = np.argmax(scores, axis=1)
    names = []
    for path, label, row in zip(paths, labels, scores):
        name = NAMES[label]
        print(f"{path}: {name}, model score={row[label]:.3f}")
        names.append(name)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="1-10 WAV paths, in the desired output order")
    args = parser.parse_args()
    try:
        recognise(args.paths)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Recognition failed: {error}\n")
    print("Model scores are not guaranteed probabilities of correctness.")


if __name__ == "__main__":
    main()
