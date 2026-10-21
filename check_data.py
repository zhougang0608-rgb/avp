"""Check recording counts and audio quality without training a model."""
import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from settings import ROOT, NAMES, SAMPLE_RATE, SECONDS


def check_recording(path):
    """Return problems found in one recording; do not change the audio."""
    try:
        rate, audio = wavfile.read(path)
    except (ValueError, OSError) as error:
        return [f"cannot read WAV: {error}"]
    problems = []
    if rate != SAMPLE_RATE:
        problems.append(f"sample rate {rate} Hz; expected {SAMPLE_RATE} Hz")
    if audio.ndim != 1:
        problems.append("expected mono audio")
    if audio.size == 0:
        return problems + ["empty recording"]
    if not np.isfinite(audio).all():
        return problems + ["non-finite sample values"]
    if rate <= 0 or abs(len(audio) / rate - SECONDS) > 1 / SAMPLE_RATE:
        problems.append(f"expected a duration of {SECONDS} seconds")
    amplitude = audio.astype(float)
    if np.issubdtype(audio.dtype, np.unsignedinteger):
        midpoint = float(np.iinfo(audio.dtype).max + 1) / 2
        amplitude = (amplitude - midpoint) / midpoint
    elif np.issubdtype(audio.dtype, np.signedinteger):
        scale = max(abs(int(np.iinfo(audio.dtype).min)), int(np.iinfo(audio.dtype).max))
        amplitude /= scale
    if np.max(np.abs(amplitude)) < 0.001:
        problems.append("recording is silent or too quiet")
    return problems


def check_split(data_dir, split):
    """Check expected files under folders whose names supply the labels."""
    count = 20 if split == "train" else 5
    missing = invalid = valid = 0
    for name in NAMES:
        present = 0
        for index in range(1, count + 1):
            path = data_dir / split / name / f"{name}_{split}_{index:02d}.wav"
            if not path.is_file():
                missing += 1
                continue
            present += 1
            problems = check_recording(path)
            if problems:
                invalid += 1
                print(f"  INVALID {path}: {'; '.join(problems)}")
            else:
                valid += 1
        print(f"{split:5s} {name:10s}: {present}/{count} expected files present")
    print(f"Summary: {valid} valid, {invalid} invalid, {missing} missing.")
    print("These checks do not verify the spoken name or recognition accuracy.")
    return missing == 0 and invalid == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="recording root containing train/ and test/ folders")
    args = parser.parse_args()
    if len(NAMES) != 20 or len(set(NAMES)) != 20 or any(not name.strip() for name in NAMES):
        parser.error("Names.txt must contain 20 distinct, non-empty names.")
    return 0 if check_split(args.data_dir, args.split) else 1


if __name__ == "__main__":
    raise SystemExit(main())
