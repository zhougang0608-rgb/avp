"""Record missing audio files; automatically skip existing files to avoid overwriting them."""
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from settings import ROOT, NAMES, SAMPLE_RATE, SECONDS


def main():
    split = input("Enter train or test: ").strip()
    if split not in ["train", "test"]:
        raise ValueError("Please enter only train or test.")
    count = 20 if split == "train" else 5
    for name in NAMES:
        folder = ROOT / "data" / split / name
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(1, count + 1):
            path = folder / f"{name}_{split}_{i:02d}.wav"
            if path.exists():
                continue
            answer = input(f"Press Enter to record {name} recording {i}; q to quit: ")
            if answer.lower() == "q":
                return
            print("Recording... Please say one name.")
            audio = sd.rec(int(SAMPLE_RATE * SECONDS), samplerate=SAMPLE_RATE,
                           channels=1, dtype="float32")
            sd.wait()
            audio = audio[:, 0]
            if np.max(np.abs(audio)) < 0.001:
                raise ValueError("The volume is too low. Check the microphone and run the program again.")
            wavfile.write(path, SAMPLE_RATE, audio)
            print("Saved:", path.name)


if __name__ == "__main__":
    main()
