# Version 5 Demonstration Guide

## Start

On the owner's Mac:

```bash
cd ~/Documents/Codex/2026-10-05/qia/outputs/avp
conda activate speech
python -B demo.py
```

If the environment is not activated, use the existing interpreter directly:

```bash
/opt/anaconda3/envs/speech/bin/python -B demo.py
```

The local project has copies of the existing recordings and saved model assets. A fresh GitHub clone does not include those ignored assets; restore them or collect/train before demonstrating.

## File recognition

Choose `1`. Enter `Charlie` to recognise that name's first existing test file. The shortcut selects a file; it does not supply an answer to the model. The chosen file path is shown.

Choose `1` again and enter:

```text
Charlie Samuel Toby
```

The actual checked outputs are Charlie, Nicholas and Ali, respectively. Only the first is correct. Say:

> The program returns one prediction for each input file in the same order. These examples include correct and incorrect predictions. A high model score does not guarantee a correct prediction.

For arbitrary recordings or other examples, enter full WAV paths or paths relative to this project. Put quotes around paths containing spaces. The interface accepts 1-10 files; name shortcuts always use `test_01.wav`.

## Evaluation

Choose `2` to evaluate all 100 expected test files. This updates the test summary, predictions table and matrix. It does not train the model. Say:

> The selected model achieved 90% validation accuracy. On these 100 test recordings, it correctly recognised 27, giving 27% test accuracy. These are different datasets. The test recordings were previously inspected during debugging, so this is not an untouched final test set.

## Confusion matrix

Choose `3` to open the saved test matrix. Rows are true names; columns are predictions. The diagonal contains correct predictions. Examples:

- Charlie: five correct.
- Alex: four correct, one predicted as Harriet.
- Samuel: all five predicted as Nicholas.

Discuss recording conditions and overfitting as possible explanations, not established causes. Systematic noise experiments have not been completed.

## Experiment table and learning curves

Choose `4` to open the seven-configuration table. Choose `5` to open the selected model's curves. Say:

> Each alternative changes one factor from the baseline. The split and training settings are fixed. The 40-Mel-band and energy configurations both achieved 90% validation accuracy. The code keeps the first configuration in an accuracy tie, selecting the 40-band model.

Blue is training and orange is validation. Training accuracy near 100% with validation near 90% suggests overfitting. Early stopping uses validation loss, not test accuracy.

## Optional microphone recognition

Choose `6`. Press Enter, wait for the recording prompt, then say one name. Use `p` to replay and `q` to return to the menu. If entering a true name, spell it as listed in `Names.txt`.

Microphone WAV files and practice logs are saved locally in `live_recordings/` and excluded from Git. Entered true labels are logged for practice and do not update the model. The Version 5 microphone workflow was checked using simulated input from existing recordings, without using a real microphone or speaker; rehearse the hardware separately.

## Finish

Enter `q` at the main menu. Use the saved model for the assessment; retraining or rerunning the seven comparisons is not needed to show file recognition and stored results. Be ready to explain the code and answer questions independently.
