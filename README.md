# AVP Speech Recogniser - Version 1

## Scope

Version 1 prepares recording collection and dataset checks for a speaker-dependent recogniser of the 20 names in `Names.txt`. It does not yet extract MFCCs, train a neural network or predict names.

This new repository starts on 21 October 2026. Its initial baseline reuses the name list and recording script from the owner's existing speech-recognition project, with simplified shared settings and a new data-checking script. The commits record this new staged project, not the original dates of earlier development.

## Files

| File | Purpose |
| --- | --- |
| `Names.txt` | The 20 expected name labels |
| `settings.py` | Project path, sample rate and recording duration |
| `record.py` | Record missing examples without overwriting existing files |
| `check_data.py` | Check expected filenames, counts, sample rate, mono format, duration, finite samples and minimum volume |
| `requirements.txt` | Dependencies needed by this version |
| `AI_USAGE.md` | Record of AI assistance for this version |

## Recording scheme

Record only the owner's voice, saying one complete name per file. The settings are 16 kHz, mono and 2.5 seconds. The target is 20 development recordings and five additional test recordings per name:

```text
data/train/Aaron/Aaron_train_01.wav
...
data/train/Aaron/Aaron_train_20.wav
data/test/Aaron/Aaron_test_01.wav
...
data/test/Aaron/Aaron_test_05.wav
```

The training folder holds 400 development recordings in total. A training/validation split will be introduced in a later version. Test data will be kept separate. Audio remains local and is excluded from Git; no recordings are supplied by cloning this repository.

## Run

Use the existing `speech` Python environment on the owner's Mac, or install `requirements.txt` into a separate Python environment.

```bash
conda activate speech
python -B record.py
```

Enter `train` or `test` when prompted. Press Enter to record the displayed name, or `q` to quit. Existing expected files are skipped. Keep the microphone distance and input volume consistent.

Check recordings:

```bash
python -B check_data.py --split train
python -B check_data.py --split test
```

To inspect existing recordings in another folder without copying or modifying them:

```bash
python -B check_data.py --split train --data-dir ~/Desktop/speech-recogniser/data
```

Missing or invalid recordings give a non-zero exit status. File counts and audio checks are not recognition accuracy and cannot establish that the spoken names are correct. Listen to recordings to verify the content and labels.

## Verification and limitations

- Syntax checks completed for this version's three Python files.
- The checker was run read-only against the existing project's 400 development recordings and 100 test recordings. This is reuse of existing audio, not new data collection.
- Focused checks covered silence, incorrect sample rates, stereo audio, incorrect duration, non-finite samples and unreadable WAV files.
- No real microphone recording was performed while preparing this version.
- This repository has no local `data/` initially; checks against its default data location will report missing recordings until audio is recorded or restored.

## Planned stages

1. Version 1: recording collection and dataset checks (this version).
2. Later: implement and explain MFCC feature extraction.
3. Later: introduce a fixed training/validation split and a baseline convolutional model.
4. Later: compare configurations and evaluate file recognition.

These later stages have not been added to this repository yet.
