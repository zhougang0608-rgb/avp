"""Check MFCC shape and finite values for development recordings; do not train a model."""
import argparse
import json
from pathlib import Path

import numpy as np

from features import extract_features
from settings import ROOT, NAMES, MAX_FRAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="recording root containing the train/ folder")
    parser.add_argument("--filters", type=int, default=26)
    parser.add_argument("--report", type=Path, help="optional path for a JSON check report")
    args = parser.parse_args()
    if not 13 <= args.filters <= 256:
        parser.error("Use between 13 and 256 Mel filters.")
    if len(NAMES) != 20 or len(set(NAMES)) != 20 or any(not name.strip() for name in NAMES):
        parser.error("Names.txt must contain 20 distinct, non-empty names.")
    valid = missing = invalid = 0
    errors = []
    for name in NAMES:
        name_valid = 0
        for index in range(1, 21):
            relative = Path("train") / name / f"{name}_train_{index:02d}.wav"
            path = args.data_dir / relative
            if not path.is_file():
                missing += 1
                errors.append({"file": str(relative), "error": "missing recording"})
                continue
            try:
                features = extract_features(path, args.filters)
                if features.shape != (MAX_FRAMES, 13):
                    raise ValueError(f"unexpected feature shape: {features.shape}")
                if not np.isfinite(features).all():
                    raise ValueError("non-finite feature values")
            except (ValueError, OSError) as error:
                invalid += 1
                errors.append({"file": str(relative), "error": str(error)})
                continue
            valid += 1
            name_valid += 1
        print(f"{name:10s}: {name_valid}/20 recordings passed feature checks")
    summary = {
        "version": 2,
        "scope": "MFCC software checks on existing development recordings; no model or test accuracy",
        "mel_filters": args.filters,
        "expected_shape": [MAX_FRAMES, 13],
        "expected_recordings": len(NAMES) * 20,
        "valid": valid,
        "invalid": invalid,
        "missing": missing,
        "errors": errors,
    }
    print(f"Summary: {valid} valid, {invalid} invalid, {missing} missing.")
    print("Passing these checks does not establish recognition accuracy or correct spoken labels.")
    for error in errors[:5]:
        print(f"  {error['file']}: {error['error']}")
    if len(errors) > 5:
        print(f"  ... {len(errors) - 5} further issues")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
        print("Report saved:", args.report)
    return 0 if missing == 0 and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
