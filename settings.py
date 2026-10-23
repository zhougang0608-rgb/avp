"""Shared recording and MFCC settings for Version 3."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = (ROOT / "Names.txt").read_text().splitlines()
SAMPLE_RATE = 16000
SECONDS = 2.5

FRAME_LENGTH = 400  # 25 milliseconds at 16 kHz.
HOP_LENGTH = 160    # Start a frame every 10 milliseconds.
N_FFT = 512        # Zero-pad each frame before the FFT.
MAX_FRAMES = 100   # Interpolate the whole feature sequence to this length.
FEATURE_VERSION = "mfcc-baseline-v2"
