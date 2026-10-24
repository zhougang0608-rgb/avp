"""Compare seven configurations on one fixed split; never read test recordings."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import load_development_data, prepare_split
from model import build_model
from settings import ROOT, NAMES, FEATURE_VERSION


# Each tuple changes just one factor from baseline.
# name, Mel bands, delta, energy, convolutional filters, blocks, dropout
EXPERIMENTS = [
    ("baseline", 26, False, False, 8, 1, 0.0),
    ("more_mel", 40, False, False, 8, 1, 0.0),
    ("with_delta", 26, True, False, 8, 1, 0.0),
    ("with_energy", 26, False, True, 8, 1, 0.0),
    ("more_filters", 26, False, False, 16, 1, 0.0),
    ("two_blocks", 26, False, False, 8, 2, 0.0),
    ("with_dropout", 26, False, False, 8, 1, 0.3),
]


def save_curves(history, target):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    epochs = range(1, len(history["loss"]) + 1)
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(epochs, history[metric], label="train")
        ax.plot(epochs, history["val_" + metric], label="validation")
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(target)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--epochs", type=int, default=45)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Epochs must be a positive integer.")
    result_dir = ROOT / "results" / "comparison"
    model_dir = ROOT / "models"
    selected_dir = model_dir / "selected"
    result_dir.mkdir(parents=True, exist_ok=True)
    selected_dir.mkdir(parents=True, exist_ok=True)
    reference_split = None
    best_correct = -1
    best_config = None
    rows = []
    for name, mel, use_delta, use_energy, filters, blocks, dropout in EXPERIMENTS:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        x, y, paths = load_development_data(args.data_dir, mel, use_delta, use_energy)
        x_train, x_val, train_ids, val_ids, mean, std = prepare_split(x, y)
        split = {"train": [paths[i] for i in train_ids],
                 "validation": [paths[i] for i in val_ids]}
        if reference_split is None:
            reference_split = split
        elif split != reference_split:
            raise RuntimeError("Training/validation files changed between experiments.")
        model = build_model(x_train.shape[1:], filters, blocks, dropout)
        stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
        history = model.fit(x_train, y[train_ids],
                            validation_data=(x_val, y[val_ids]), epochs=args.epochs,
                            batch_size=16, callbacks=[stop], verbose=2)
        loss, accuracy = model.evaluate(x_val, y[val_ids], verbose=0)
        predictions = np.argmax(model.predict(x_val, verbose=0), axis=1)
        correct = int(np.sum(predictions == y[val_ids]))
        if not np.isclose(accuracy, correct / len(val_ids)):
            raise RuntimeError("Validation accuracy differs from prediction count.")
        row = {"experiment": name, "mel_filters": mel, "delta": use_delta,
               "energy": use_energy, "filters": filters, "blocks": blocks,
               "dropout": dropout, "validation_loss": float(loss),
               "validation_correct": correct, "validation_count": len(val_ids),
               "validation_accuracy": correct / len(val_ids),
               "epochs_run": len(history.history["loss"]),
               "restored_epoch": int(np.argmin(history.history["val_loss"])) + 1}
        rows.append(row)
        (result_dir / f"{name}_history.json").write_text(json.dumps(history.history, indent=2) + "\n")
        save_curves(history.history, result_dir / f"{name}_curves.png")
        print(f"{name}: {correct}/{len(val_ids)} ({correct / len(val_ids):.2%})")
        # A strict comparison keeps the first experiment when accuracy ties.
        if correct > best_correct:
            best_correct = correct
            model.save(selected_dir / "best.keras")
            np.savez(selected_dir / "normalisation.npz", mean=mean, std=std)
            best_config = {**row, "names": NAMES, "feature_version": FEATURE_VERSION,
                           "input_shape": list(x_train.shape[1:]), "dense_units": 32,
                           "learning_rate": 0.001, "model_file": "selected/best.keras",
                           "normalisation_file": "selected/normalisation.npz"}
            selected_raw = x[val_ids].copy()
            selected_predictions = predictions.copy()
    restored = tf.keras.models.load_model(selected_dir / "best.keras")
    with np.load(selected_dir / "normalisation.npz") as stats:
        restored_x = (selected_raw - stats["mean"]) / stats["std"]
    restored_predictions = np.argmax(restored.predict(restored_x, verbose=0), axis=1)
    if not np.array_equal(selected_predictions, restored_predictions):
        raise RuntimeError("Selected model predictions changed after saving/reloading.")
    with (result_dir / "experiments.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (result_dir / "split.json").write_text(json.dumps(reference_split, indent=2) + "\n")
    (model_dir / "config.json").write_text(json.dumps(best_config, indent=2) + "\n")
    summary = {"version": 4, "experiments": len(rows),
               "training_recordings": 320, "validation_recordings": 80,
               "selected_experiment": best_config["experiment"],
               "validation_correct": best_correct,
               "validation_accuracy": best_correct / 80,
               "tie_rule": "first in experiment order",
               "seed": 42, "batch_size": 16, "epochs_requested": args.epochs,
               "early_stopping": {"monitor": "val_loss", "patience": 5},
               "tensorflow_version": tf.__version__,
               "saved_model_reload_matches": True, "test_set_used": False}
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Selected: {best_config['experiment']}; validation {best_correct}/80.")
    print("Saved-model reload check passed. Test evaluation is a later stage.")


if __name__ == "__main__":
    main()
