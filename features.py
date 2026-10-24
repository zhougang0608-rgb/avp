"""Compute MFCCs using basic array operations, FFT, and DCT, without a ready-made MFCC function."""
import numpy as np
from scipy.io import wavfile
from scipy.fft import dct
from settings import SAMPLE_RATE, FRAME_LENGTH, HOP_LENGTH, N_FFT, MAX_FRAMES


def read_audio(path):
    rate, signal = wavfile.read(path)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path}: expected 16000 Hz, received {rate} Hz")
    # Convert integer WAV samples to floating-point amplitudes from -1 to 1, preserving volume information for recording checks.
    if signal.dtype == np.uint8:
        signal = (signal.astype(float) - 128) / 128
    elif np.issubdtype(signal.dtype, np.integer):
        scale = max(abs(np.iinfo(signal.dtype).min), np.iinfo(signal.dtype).max)
        signal = signal.astype(float) / scale
    else:
        signal = signal.astype(float)
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    return signal


def prepare_audio(signal):
    """Find speech boundaries using short-time volume; this is simple endpoint detection, not a general speech detector."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1 or len(signal) < FRAME_LENGTH:
        raise ValueError("The recording is too short or is not mono audio.")
    if not np.isfinite(signal).all():
        raise ValueError("The recording contains invalid values.")
    if len(signal) > SAMPLE_RATE * 5:
        raise ValueError("Use a recording of no more than 5 seconds containing only one spoken name.")
    peak = np.max(np.abs(signal))
    if peak < 0.003:
        raise ValueError("The recording is too quiet. Check the microphone and move closer.")
    signal = signal - signal.mean()   # Remove the DC offset.
    frames = make_frames(signal)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    noise_level = np.percentile(rms, 10)
    threshold = max(0.001, 0.12 * rms.max(), 3 * noise_level)
    active = np.flatnonzero(rms > threshold)
    if len(active) < 8:
        raise ValueError("The detected sound is too short. Please say one complete name.")
    # Keep 80 ms on each side to reduce the risk of cutting off quiet consonants.
    start = max(0, active[0] * HOP_LENGTH - 1280)
    end = min(len(signal), active[-1] * HOP_LENGTH + FRAME_LENGTH + 1280)
    info = {"peak": float(peak), "start": start / SAMPLE_RATE,
            "end": end / SAMPLE_RATE, "seconds": (end - start) / SAMPLE_RATE}
    signal = signal[start:end]
    signal = signal / (np.max(np.abs(signal)) + 1e-10)
    return signal, info


def make_frames(signal):
    # Round up and zero-pad to preserve the final samples.
    count = 1 + int(np.ceil(max(0, len(signal) - FRAME_LENGTH) / HOP_LENGTH))
    total = (count - 1) * HOP_LENGTH + FRAME_LENGTH
    signal = np.pad(signal, (0, total - len(signal)))
    frames = np.zeros((count, FRAME_LENGTH))
    for i in range(count):
        start = i * HOP_LENGTH
        frames[i] = signal[start:start + FRAME_LENGTH]
    return frames


def make_filterbank(n_filters):
    # Mel spacing is denser at low frequencies; create equally spaced Mel points, then convert to Hz.
    max_mel = 2595 * np.log10(1 + (SAMPLE_RATE / 2) / 700)
    mel_points = np.linspace(0, max_mel, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    frequencies = np.fft.rfftfreq(N_FFT, 1 / SAMPLE_RATE)
    bank = np.zeros((n_filters, len(frequencies)))
    for i in range(n_filters):
        left, centre, right = hz_points[i:i + 3]
        rising = (frequencies - left) / (centre - left)
        falling = (right - frequencies) / (right - centre)
        bank[i] = np.maximum(0, np.minimum(rising, falling))
    return bank


def delta(features):
    """Central difference with repeated edge frames."""
    padded = np.pad(features, ((1, 1), (0, 0)), mode="edge")
    return (padded[2:] - padded[:-2]) / 2


def extract_features(path, n_filters=26, use_delta=False, use_energy=False):
    if not 13 <= n_filters <= N_FFT // 2:
        raise ValueError("Use between 13 and 256 Mel filters.")
    signal, _ = prepare_audio(read_audio(path))
    # Pre-emphasis boosts higher frequencies; keep the first sample unchanged because it has no previous sample.
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    frames = make_frames(signal)
    energy = np.log(np.maximum(np.sum(frames ** 2, axis=1), 1e-10))
    frames = frames * np.hamming(FRAME_LENGTH)   # Reduce abrupt changes at frame boundaries.
    spectrum = np.fft.rfft(frames, n=N_FFT, axis=1)
    power = np.abs(spectrum) ** 2 / N_FFT
    bank = make_filterbank(n_filters)
    filter_energy = power @ bank.T               # Weighted energy in each frequency band.
    log_energy = np.log(np.maximum(filter_energy, 1e-10))
    features = dct(log_energy, type=2, axis=1, norm="ortho")[:, :13]
    if use_energy:
        features = np.column_stack((features, energy))
    if use_delta:
        features = np.column_stack((features, delta(features)))
    # Interpolate to 100 frames using relative time, preserving the whole spoken name instead of taking only the first 100 frames.
    # Interpolation aligns faster and slower pronunciations, but reduces absolute duration information.
    old_time = np.linspace(0, 1, len(features))
    new_time = np.linspace(0, 1, MAX_FRAMES)
    result = np.zeros((MAX_FRAMES, features.shape[1]), dtype=np.float32)
    for column in range(features.shape[1]):
        result[:, column] = np.interp(new_time, old_time, features[:, column])
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract MFCCs from one existing WAV file.")
    parser.add_argument("path", help="path to a 16 kHz WAV recording")
    parser.add_argument("--filters", type=int, default=26, help="number of Mel bands (default: 26)")
    args = parser.parse_args()
    try:
        features = extract_features(args.path, args.filters)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Feature extraction failed: {error}\n")
    print("Recording:", args.path)
    print("Mel filters:", args.filters)
    print("MFCC shape:", features.shape)
    print("All values finite:", bool(np.isfinite(features).all()))
    print("First time step:", np.array2string(features[0], precision=3))
    print("These are audio features, not a predicted name or recognition accuracy.")


if __name__ == "__main__":
    main()
