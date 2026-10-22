# AI Assistance Record

## 21 October 2026 - Version 1

The owner asked OpenAI Codex to organise a new repository into stages, based on an existing speech-recognition project.

Codex reused the existing name list and recording script, simplified the shared settings, implemented basic dataset checks, prepared documentation and ignore rules, and checked the resulting files. The original recording code had also received AI assistance, as recorded in the existing project's AI usage record.

Existing audio was used for read-only checks. No new microphone recordings, MFCC extraction, neural-network training or recognition evaluation were performed for this version. This is a new staged baseline, not a reconstruction of earlier development history.

The owner is responsible for checking recordings and labels, understanding the code, and explaining the work independently during the assessment.

## 22 October 2026 - Version 2

The owner asked Codex to prepare the next stage. Codex adapted the existing project's baseline MFCC implementation into this repository, added the frame/FFT settings and a feature-checking command, and updated the documentation. Optional delta/energy experiments and neural-network code have not been included at this stage.

Codex ran feature checks on 400 existing development recordings, compared one feature array to the existing implementation, and checked frame boundaries, filterbank properties and invalid inputs. These were software/feature checks, not recognition-accuracy tests. No new microphone recording or model training was performed.

The owner should learn and explain each MFCC stage. The current commits document actual staging work on these dates, not a claim that the reused code was first developed on those dates.
