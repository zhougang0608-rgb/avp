# Changelog

## Version 5 - 25 October 2026

- Add ordered 1-10 WAV recognition and shared selected-model preprocessing.
- Evaluate 100 existing test recordings using the unchanged Version 4 model: 27/100 (27%).
- Save per-file predictions, summary and a 20-class test confusion matrix.
- Add a short demo menu, optional microphone/replay/practice-log interface and English demonstration guide.
- Restore existing audio into ignored local data for demonstration; keep recordings, practice logs and binary model assets out of Git.
- Check inputs, configuration consistency, result consistency and simulated live/menu workflows. No real microphone test or model retraining.
- Complete the five-version plan while preserving earlier commits and documenting AI assistance and limitations.

## Version 4 - 24 October 2026

- Add optional delta/log-energy features and configurable convolutional filters, blocks and dropout.
- Add a seven-configuration comparison with a checked fixed split and training-only statistics.
- Actual selected result: more_mel, 72/80 (90%) validation accuracy; energy also reaches 90%, with ties keeping the first experiment.
- Save per-experiment histories/curves and the selected model/configuration; verify reload predictions.
- Keep baseline assets/reports separate; no test evaluation or microphone trial in this version.
- Finalise a five-version plan, leaving file recognition and evaluation for Version 5.

## Version 3 - 23 October 2026

- Add development-data loading and a fixed stratified 320/80 split, with training-only normalisation.
- Add one baseline 2D convolutional classifier using 26 Mel bands and 13 MFCC coefficients.
- Train with early stopping, save/reload the model and normalisation statistics, and record the split and learning curves.
- Actual validation result: 67/80 (83.75%), after 24 epochs with epoch 19 restored. Test recordings were not read.
- Track the configuration and reports; keep binary model assets local.
- Leave configuration comparison and file-based test evaluation for a later version.

## Version 2 - 22 October 2026

- Adapt the existing project's manually implemented baseline MFCC pipeline.
- Add frame length, frame hop, FFT size and fixed output-length settings.
- Add a single-file feature command and a development-recording feature checker.
- Store the actual 400-recording feature-check report: all arrays have shape 100 by 13 and finite values.
- Document the algorithm, commands, limitations and AI assistance.
- Leave neural-network training and recognition evaluation for later versions.

## Version 1 - 21 October 2026

- Initial baseline commit `66e972d`: name labels, recording collection, audio checks, settings, dependencies and documentation.
- Reuse the owner's existing recording code; establish a new repository without reconstructing earlier history.
