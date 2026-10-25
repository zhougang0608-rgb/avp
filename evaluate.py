"""Evaluate the selected model on 100 existing test files; never retrain it."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from check_data import check_recording
from predict import load_recogniser, predict_scores
from settings import ROOT, NAMES


def test_files(data_dir):
    """Use exactly the five named test examples per class and reject incomplete data."""
    paths, labels = [], []
    for label, name in enumerate(NAMES):
        for index in range(1, 6):
            path = Path(data_dir) / "test" / name / f"{name}_test_{index:02d}.wav"
            if not path.is_file():
                raise FileNotFoundError(f"Missing test recording: {path}")
            problems = check_recording(path)
            if problems:
                raise ValueError(f"{path.name}: {'; '.join(problems)}")
            paths.append(path)
            labels.append(label)
    return paths, np.asarray(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    try:
        paths, truth = test_files(args.data_dir)
        recogniser = load_recogniser()
        scores = np.concatenate([predict_scores(paths[start:start + 10], recogniser)
                                 for start in range(0, len(paths), 10)])
    except (ValueError, OSError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    predictions = np.argmax(scores, axis=1)
    correct = int(np.sum(predictions == truth))
    matrix = confusion_matrix(truth, predictions, labels=np.arange(len(NAMES)))
    result_dir = ROOT / "results"
    result_dir.mkdir(exist_ok=True)
    with (result_dir / "test_predictions.csv").open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["file", "true_name", "predicted_name", "model_score"])
        for path, actual, predicted, row in zip(paths, truth, predictions, scores):
            writer.writerow([path.relative_to(args.data_dir).as_posix(), NAMES[actual],
                             NAMES[predicted], float(row[predicted])])
    np.savetxt(result_dir / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    fig, ax = plt.subplots(figsize=(12, 11))
    ConfusionMatrixDisplay(matrix, display_labels=NAMES).plot(
        ax=ax, xticks_rotation=90, colorbar=False)
    ax.set_title(f"100 test recordings: {correct}/100 correct ({correct / len(truth):.0%})")
    fig.tight_layout()
    fig.savefig(result_dir / "confusion_matrix.png")
    plt.close(fig)
    summary = {"version": 5, "test_files": len(paths), "correct": correct,
               "accuracy": correct / len(paths), "experiment": recogniser[1]["experiment"],
               "model_retrained": False,
               "data_note": "Existing test recordings were previously inspected during debugging; not an untouched final test set."}
    (result_dir / "test_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Test accuracy: {correct}/{len(paths)} ({correct / len(paths):.2%})")
    print("Rows: true names. Columns: predicted names. The diagonal contains correct predictions.")
    print(summary["data_note"])


if __name__ == "__main__":
    main()
