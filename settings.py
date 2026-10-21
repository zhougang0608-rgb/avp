"""Basic recording settings shared by the Version 1 scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = (ROOT / "Names.txt").read_text().splitlines()
SAMPLE_RATE = 16000
SECONDS = 2.5
