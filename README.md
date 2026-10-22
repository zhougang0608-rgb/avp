# AVP Speech Recogniser - Version 2

## Scope

Version 2 adds manually implemented MFCC feature extraction and feature checks to the recording baseline for the 20 names in `Names.txt`. There is no neural-network training or name prediction in this version.

This new repository starts on 21 October 2026. Its initial baseline reuses the name list and recording script from the owner's existing speech-recognition project, with simplified shared settings and a new data-checking script. The commits record this new staged project, not the original dates of earlier development.

## Files

| File | Purpose |
| --- | --- |
| `Names.txt` | The 20 expected name labels |
| `settings.py` | Project path, sample rate and recording duration |
| `record.py` | Record missing examples without overwriting existing files |
| `check_data.py` | Check expected filenames, counts, sample rate, mono format, duration, finite samples and minimum volume |
| `features.py` | Convert a WAV recording to a 100 by 13 MFCC array |
| `check_features.py` | Check development-recording feature shapes and finite values |
| `results/feature_check_v2.json` | Stored report from checking 400 existing development recordings |
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

## MFCC feature extraction (Version 2)

Version 2 reuses and adapts the baseline MFCC implementation from the existing complete project. It combines NumPy array and FFT operations with SciPy WAV reading and DCT; no ready-made MFCC function is used.

The pipeline is:

1. Read 16 kHz audio, convert sample amplitudes and locate the speech segment using frame RMS energy.
2. Keep an 80 ms margin around the detected segment and apply peak normalisation.
3. Apply pre-emphasis (0.97), split into 25 ms frames with a 10 ms hop, and apply a Hamming window.
4. Compute a 512-point FFT and power spectrum.
5. Sum energy through 26 Mel-spaced triangular filters, take logarithms, and apply a DCT.
6. Keep 13 coefficients and interpolate the complete sequence to 100 time steps.

Inspect one existing recording without copying or modifying it:

```bash
python -B features.py ~/Desktop/speech-recogniser/data/train/Aaron/Aaron_train_01.wav
```

Expected output includes `MFCC shape: (100, 13)` and `All values finite: True`. The first number is the time dimension and the second is the coefficient dimension; these are not name predictions.

Check all 400 expected development recordings:

```bash
python -B check_features.py --data-dir ~/Desktop/speech-recogniser/data
```

To explicitly save a new report, add `--report results/feature_check_v2.json`. Without that option the checker only prints results. With a local dataset under this repository, omit `--data-dir`.

## Verification and limitations

Version 1 (21 October 2026): recording/data checks were run against 400 development and 100 test recordings. No new microphone recording was made.

Version 2 (22 October 2026):

- All 400 existing development recordings produced finite MFCC arrays of shape 100 by 13 with the baseline 26 Mel filters. See `results/feature_check_v2.json`.
- Compared one recording's MFCC array to the existing complete project's baseline output; the arrays matched exactly.
- Checked frame-tail preservation, filterbank shape/range, invalid sample rates, silence, empty and non-finite audio, overlong input and invalid filter counts.
- Syntax checks passed. No model was trained, no recognition accuracy was calculated, and the 100 test recordings were not used for this version's feature check.

Finite values and matching dimensions do not establish feature quality, correct spoken labels or recognition accuracy. Simple RMS trimming can be affected by noise; peak normalisation does not remove noise. Interpolation reduces absolute duration information. Training-data standardisation will be introduced with the model, not applied per file here.

Audio is still stored outside this repository. The default data location will report missing recordings until the owner records or restores a local dataset.

## Planned stages

1. Version 1: recording collection and dataset checks (commit `66e972d`).
2. Version 2: baseline MFCC extraction and feature checks (this version).
3. Later: introduce a fixed training/validation split and a baseline convolutional model.
4. Later: compare configurations and evaluate file recognition.

Model training, configuration comparison and recognition evaluation have not been added to this repository yet.
