# Changelog

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
