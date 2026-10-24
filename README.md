# AVP Speech Recogniser - Version 4

## Scope

Version 4 compares seven configurations on the same 320/80 development split and selects a model using validation accuracy. A file-prediction interface, test-set evaluation and a demonstration menu are planned for Version 5.

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
| `dataset.py` | Load development features and split/standardise using training statistics only |
| `model.py` | Define a baseline Conv2D classifier |
| `train.py` | Train, save, reload and validate one baseline |
| `compare.py` | Compare seven controlled configurations and select a model |
| `models/config.json` | Selected Version 4 model settings and local asset paths |
| `models/baseline_config.json` | Separate baseline settings |
| `results/comparison/` | Seven experiment histories/curves, comparison table, fixed split and selection summary |
| `results/baseline_summary.json` | Actual training/validation results |
| `results/baseline_history.json` | Per-epoch learning history |
| `results/baseline_curves.png` | Training/validation loss and accuracy plots |
| `results/split.json` | Relative filenames in the fixed split |
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

The training folder holds 400 development recordings in total. Version 3 splits these into 320 training and 80 validation recordings (16 and four per name), using stratification and random seed 42. Test data will be kept separate. Audio remains local and is excluded from Git; no recordings are supplied by cloning this repository.

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

## Baseline model training (Version 3)

The new training command adapts the baseline architecture and splitting/normalisation approach from the existing complete project. It does not import that project's model or its stored accuracy figures.

```bash
conda activate speech
python -B train.py --data-dir ~/Desktop/speech-recogniser/data
```

With recordings restored under this repository's own `data/`, omit `--data-dir`. Dependencies now include TensorFlow, scikit-learn and Matplotlib. The command deliberately overwrites this repository's baseline model and result files when rerun; it does not modify the original complete project. From Version 4 it writes `models/baseline_config.json` and leaves the selected model configuration untouched.

- Input: 100 time steps by 13 MFCC coefficients by one channel, using 26 Mel bands.
- Network: Conv2D (eight 3 by 3 filters, ReLU), 2 by 2 max pooling, Flatten, Dense (32 units, ReLU), Dense (20 Softmax outputs).
- Optimiser/loss: Adam with learning rate 0.001, sparse categorical cross-entropy, batch size 16.
- Split: fixed stratified 80/20 development split; train and validation files are disjoint.
- Normalisation: per-feature mean and standard deviation computed only from the 320 training recordings; the same statistics transform validation data.
- Training: up to 45 epochs; stop after five epochs without validation-loss improvement and restore the weights with the lowest validation loss.

The actual Version 3 run completed 24 epochs and restored epoch 19. It correctly classified 67/80 validation recordings (83.75%). This is validation accuracy for the 26-band baseline, not test accuracy or a comparison of all configurations. The saved model and normalisation statistics were reloaded, and the validation predictions matched.

The loader only reads the expected 400 `train/` recordings; `test/` is not accessed by this training run. Previously viewed test results in the separate complete project do not become untouched test data simply because this is a new repository.

The baseline weights (`models/baseline.keras`) and normalisation array (`models/normalisation.npz`) remain local and are excluded from Git. The configuration and training reports are versioned. A clone needs the recordings restored and this baseline trained before it has model assets.

To inspect the plots on macOS:

```bash
open results/baseline_curves.png
```

Lower validation loss is the early-stopping criterion, so the restored epoch need not be the epoch with the highest validation accuracy. The training/validation gap suggests overfitting; a single split and random seed do not establish general robustness.

## Controlled comparison and model selection (Version 4)

Run the seven configurations using the same existing development recordings:

```bash
python -B compare.py --data-dir ~/Desktop/speech-recogniser/data
```

This command reruns the comparisons and overwrites `results/comparison/`, the selected model assets and `models/config.json`. It does not overwrite the Version 3 baseline history or its model assets. Do not rerun training merely to display existing results.

All experiments use random seed 42, the same 320/80 stratified split, training-only standardisation, Adam (learning rate 0.001), batch size 16 and at most 45 epochs. Each run restores the epoch with minimum validation loss after five epochs without improvement. Models are then compared by restored-model validation accuracy. Each alternative changes exactly one factor from the baseline:

| Configuration | Change from baseline | Validation result |
| --- | --- | --- |
| baseline | 26 Mel bands, eight filters, one block, no dropout | 67/80 (83.75%) |
| more_mel | 40 Mel bands | 72/80 (90%) |
| with_delta | Add central-difference feature changes | 65/80 (81.25%) |
| with_energy | Append log frame energy | 72/80 (90%) |
| more_filters | 16 convolutional filters | 68/80 (85%) |
| two_blocks | Two convolution/pooling blocks | 67/80 (83.75%) |
| with_dropout | Dropout rate 0.3 after Dense(32) | 67/80 (83.75%) |

These are results from the actual Version 4 run on 24 October 2026, not copied result files from the complete project. They are validation results, not test accuracy. A single seed/split does not show that a configuration always performs better.

The selected configuration is `more_mel`. `with_energy` ties at 90%, but the rule keeps the first configuration in experiment order when accuracy ties. This rule uses validation accuracy, not the test set or validation loss as a tie-breaker.

Feature widths are 13 for baseline/more_mel, 26 with delta, and 14 with energy. Energy is the logarithm of the sum of squared pre-emphasised frame samples before windowing; the optional delta is a simple central difference with repeated edge frames. This code does not claim to implement a noise-removal algorithm.

The selected model and its training-only statistics are local at `models/selected/best.keras` and `models/selected/normalisation.npz`. `models/config.json` records the corresponding settings and paths. Saved/reloaded validation predictions matched. Cloning the repository alone does not provide these binary assets; recordings and training are still needed to recreate them.

Open the saved results on macOS:

```bash
open results/comparison/experiments.csv
open results/comparison/more_mel_curves.png
```

The defaults of `extract_features` and `build_model` still give the original baseline configuration. Optional delta/energy or network changes are enabled only by the experiment parameters. File prediction and test-set accuracy are not implemented in this stage.

## Verification and limitations

Version 1 (21 October 2026): recording/data checks were run against 400 development and 100 test recordings. No new microphone recording was made.

Version 2 (22 October 2026):

- All 400 existing development recordings produced finite MFCC arrays of shape 100 by 13 with the baseline 26 Mel filters. See `results/feature_check_v2.json`.
- Compared one recording's MFCC array to the existing complete project's baseline output; the arrays matched exactly.
- Checked frame-tail preservation, filterbank shape/range, invalid sample rates, silence, empty and non-finite audio, overlong input and invalid filter counts.
- Syntax checks passed. No model was trained, no recognition accuracy was calculated, and the 100 test recordings were not used for this version's feature check.

Finite values and matching dimensions do not establish feature quality, correct spoken labels or recognition accuracy. Simple RMS trimming can be affected by noise; peak normalisation does not remove noise. Interpolation reduces absolute duration information. Version 3 adds training-only standardisation after the MFCC stage; it is not fitted separately to each file.

Version 3 (23 October 2026):

- Verified a disjoint, deterministic 320/80 split, with 16 training and four validation recordings per name.
- Changed only validation feature values in a software check; the fitted training mean and standard deviation stayed unchanged.
- Completed one real baseline training run and checked saved-model/normalisation reload consistency.
- Inspected the training-curve layout. No new microphone trial, configuration sweep or test evaluation was performed for this version.

Version 4 (24 October 2026):

- Verified one-factor experimental changes and exact feature parity with the original complete project's implementation for five feature configurations.
- Verified feature dimensions/finite values, model output shapes, the fixed disjoint split and training-only normalisation.
- Ran all seven comparisons and confirmed that every run used the same files in each subset.
- Reloaded the selected model and its saved normalisation statistics; validation predictions matched.
- Checked the comparison table, minimum-loss restored epochs and selected learning-curve layout.
- No test evaluation, microphone trial or systematic noise experiment was performed.

Audio is still stored outside this repository. The default data location will report missing recordings until the owner records or restores a local dataset.

## Five-version plan

1. Version 1: recording collection and dataset checks (`66e972d`).
2. Version 2: baseline MFCC extraction and feature checks (`ca33050`).
3. Version 3: fixed split and baseline convolutional model (`31c162c`).
4. Version 4: controlled configuration comparison and validation-based model selection (this version).
5. Version 5 (planned): 1-10 file recognition, test accuracy, confusion matrix and a short demonstration entry point.

Version 5 has not been added yet. The original complete project is separate and remains unchanged.
