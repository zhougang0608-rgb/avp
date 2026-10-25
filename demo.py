"""A short menu for file recognition, saved results and optional microphone use on macOS."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from settings import ROOT, NAMES


def resolve_files(text, data_dir):
    entries = shlex.split(text)
    if not 1 <= len(entries) <= 10:
        raise ValueError("Enter 1 to 10 WAV paths or name shortcuts.")
    paths = []
    for entry in entries:
        if entry in NAMES:
            path = data_dir / "test" / entry / f"{entry}_test_01.wav"
        else:
            path = Path(entry).expanduser()
            if not path.is_absolute():
                path = ROOT / path
        if path.suffix.lower() != ".wav" or not path.is_file():
            raise ValueError(f"WAV recording not found: {path}")
        paths.append(path)
    return paths


def open_result(path):
    if not path.is_file():
        raise FileNotFoundError(f"Saved result not found: {path}")
    subprocess.run(["open", str(path)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    while True:
        print("\n=== Speech Recogniser Demo ===")
        print("1  Recognise 1-10 existing WAV files")
        print("2  Evaluate 100 test files (updates test result files)")
        print("3  Open the saved test confusion matrix")
        print("4  Open the saved seven-configuration comparison")
        print("5  Open the selected model's training curves")
        print("6  Microphone recognition and replay")
        print("q  Quit")
        try:
            choice = input("Choose an option: ").strip().lower()
            if choice == "q":
                return
            if choice == "1":
                print("Enter WAV paths separated by spaces; quote paths containing spaces.")
                print("Or use name shortcuts, for example: Charlie Samuel Toby")
                print("Each shortcut selects that name's first test recording (test_01.wav).")
                paths = resolve_files(input("Files or names: "), args.data_dir)
                for index, path in enumerate(paths, 1):
                    print(f"Input {index}: {path}")
                from predict import recognise
                recognise(paths)
                print("Model scores are not guaranteed probabilities of correctness.")
            elif choice == "2":
                subprocess.run([sys.executable, "-B", str(ROOT / "evaluate.py"),
                                "--data-dir", str(args.data_dir)], cwd=ROOT, check=True)
            elif choice == "3":
                open_result(ROOT / "results/confusion_matrix.png")
            elif choice == "4":
                open_result(ROOT / "results/comparison/experiments.csv")
            elif choice == "5":
                config = json.loads((ROOT / "models/config.json").read_text())
                open_result(ROOT / "results/comparison" / f"{config['experiment']}_curves.png")
            elif choice == "6":
                subprocess.run([sys.executable, "-B", str(ROOT / "live.py")], cwd=ROOT, check=True)
            else:
                print("Please enter 1-6 or q.")
        except (EOFError, KeyboardInterrupt):
            print("\nDemo closed.")
            return
        except (ValueError, OSError, subprocess.CalledProcessError) as error:
            print(f"Unable to complete this action: {error}")


if __name__ == "__main__":
    main()
