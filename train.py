"""Train one baseline on development recordings; leave test recordings unread."""
import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--epochs", type=int, default=45)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Epochs must be a positive integer.")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)
    x, y, paths = load_development_data(args.data_dir)
    x_train, x_val, train_ids, val_ids, mean, std = prepare_split(x, y)
    print(f"Development: {len(y)}; training: {len(train_ids)}; validation: {len(val_ids)}")
    print("This baseline uses 26 Mel filters. Test recordings are not read.")
    model = build_model(x_train.shape[1:])
    stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5, restore_best_weights=True)
    history = model.fit(x_train, y[train_ids],
                        validation_data=(x_val, y[val_ids]),
                        epochs=args.epochs, batch_size=16, callbacks=[stop], verbose=2)
    loss, accuracy = model.evaluate(x_val, y[val_ids], verbose=0)
    predictions = np.argmax(model.predict(x_val, verbose=0), axis=1)
    correct = int(np.sum(predictions == y[val_ids]))
    if not np.isclose(accuracy, correct / len(val_ids)):
        raise RuntimeError("Validation metric differs from the prediction count.")
    result_dir, model_dir = ROOT / "results", ROOT / "models"
    result_dir.mkdir(exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "baseline.keras"
    model.save(model_path)
    np.savez(model_dir / "normalisation.npz", mean=mean, std=std)
    # Check the saved model and statistics, rather than trusting the save alone.
    restored = tf.keras.models.load_model(model_path)
    with np.load(model_dir / "normalisation.npz") as stats:
        restored_x = (x[val_ids] - stats["mean"]) / stats["std"]
    restored_predictions = np.argmax(restored.predict(restored_x, verbose=0), axis=1)
    if not np.array_equal(predictions, restored_predictions):
        raise RuntimeError("Reloaded-model predictions differ from the trained model.")
    summary = {
        "version": 3, "experiment": "baseline", "mel_filters": 26,
        "feature_version": FEATURE_VERSION, "input_shape": list(x_train.shape[1:]),
        "development_recordings": len(y), "training_recordings": len(train_ids),
        "validation_recordings": len(val_ids), "validation_correct": correct,
        "validation_accuracy": correct / len(val_ids), "validation_loss": float(loss),
        "epochs_requested": args.epochs, "epochs_run": len(history.history["loss"]),
        "restored_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        "early_stopping": {"monitor": "val_loss", "patience": 5},
        "seed": 42, "batch_size": 16, "tensorflow_version": tf.__version__,
        "python_note": "Results may vary across library versions and hardware.",
        "saved_model_reload_matches": True, "test_set_used": False,
    }
    (result_dir / "baseline_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    config = {"names": NAMES, "feature_version": FEATURE_VERSION,
              "mel_filters": 26, "filters": 8, "blocks": 1, "dense_units": 32,
              "learning_rate": 0.001, "model_file": "baseline.keras"}
    (model_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    split = {"train": [paths[i] for i in train_ids],
             "validation": [paths[i] for i in val_ids]}
    (result_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    (result_dir / "baseline_history.json").write_text(json.dumps(history.history, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(range(1, len(history.history[metric]) + 1), history.history[metric], label="train")
        ax.plot(range(1, len(history.history[metric]) + 1), history.history["val_" + metric], label="validation")
        ax.set(xlabel="Epoch", ylabel=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "baseline_curves.png")
    plt.close(fig)
    print(f"Validation accuracy: {correct}/{len(val_ids)} ({correct / len(val_ids):.2%})")
    print("Saved model reload check passed. This is not test or live accuracy.")


if __name__ == "__main__":
    main()
