"""Optional microphone recognition: Enter records, p replays, q exits."""
import csv
from datetime import datetime
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from settings import ROOT, SAMPLE_RATE, SECONDS
from features import prepare_audio
from predict import load_recogniser, predict_scores

# Adjust these settings to your own preferences and explain your design choices.
TITLE = "My Name Recogniser"
TOP_COUNT = 3
LOW_SCORE = 0.60   # This is only a warning threshold; it does not mean 60% accuracy.
SMALL_GAP = 0.15   # Suggest recording again when the top two scores are close.


def main():
    recogniser = load_recogniser()
    _, config, _, _ = recogniser
    folder = ROOT / "live_recordings"
    folder.mkdir(exist_ok=True)
    log_path = folder / "practice.csv"
    last_audio = None
    print(f"\n=== {TITLE} ===")
    print("Press Enter, wait for the prompt, then say one name. Press p to replay the last recording or q to quit.")
    print("Candidate scores are model outputs, not guaranteed probabilities of correctness.")
    while True:
        command = input("\nEnter to record / p to replay / q to quit: ").strip().lower()
        if command == "q":
            break
        if command == "p":
            if last_audio is not None:
                sd.play(last_audio, SAMPLE_RATE)
                sd.wait()
            else:
                print("There is no recording to replay yet.")
            continue
        if command:
            print("Please press Enter, p, or q.")
            continue
        # Start the microphone before prompting the user to speak to reduce missing initial sounds.
        audio = sd.rec(int(SAMPLE_RATE * SECONDS), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32")
        print("Recording: say one name now...")
        sd.wait()
        last_audio = audio[:, 0]
        try:
            _, info = prepare_audio(last_audio)
        except ValueError as error:
            print("Please record again:", error)
            continue
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = folder / f"live_{stamp}.wav"
        wavfile.write(path, SAMPLE_RATE, last_audio)
        # Use the same function as training and file prediction instead of a separate MFCC implementation.
        scores = predict_scores([path], recogniser)[0]
        order = np.argsort(scores)[::-1]
        best, second = order[:2]
        print(f"\nPredicted name: {config['names'][best]}")
        print(f"Speech segment: {info['start']:.2f}–{info['end']:.2f} seconds")
        for rank, index in enumerate(order[:TOP_COUNT], start=1):
            print(f"{rank}. {config['names'][index]:10s} model score {scores[index]:.1%}")
        if scores[best] < LOW_SCORE or scores[best] - scores[second] < SMALL_GAP:
            print("The result is uncertain. Replay the recording to check it, then try again.")
        if info['start'] == 0 or info['end'] >= SECONDS:
            print("Speech is close to the recording boundary. Try again if the beginning or end was cut off.")
        # User-entered true labels are only for practice logs; they are not fed back to the model or treated as an independent test set.
        actual = input("Which name did you say? Press Enter to skip entering the true label: ").strip()
        if actual and actual not in config["names"]:
            print("The name is not in Names.txt. The true label will be left blank; please use the spelling in the list next time.")
            actual = ""
        new_log = not log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_log:
                writer.writerow(["file", "prediction", "score", "actual", "correct"])
            correct = "" if not actual else actual == config["names"][best]
            writer.writerow([path.name, config["names"][best], float(scores[best]), actual, correct])
        if actual:
            print("The prediction was correct." if correct else "The incorrect prediction has been logged for later analysis.")


if __name__ == "__main__":
    main()
