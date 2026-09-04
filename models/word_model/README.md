# Pretrained word-sign model

`word_model.tflite` is the ensemble model from Theo Viel & Christof Henkel's
6th-place solution to Kaggle's ["Google - Isolated Sign Language Recognition"](https://www.kaggle.com/competitions/asl-signs)
competition: https://github.com/TheoViel/kaggle_islr (MIT licensed — see
`THIRD_PARTY_LICENSE.txt`).

It recognizes **one of 250 isolated ASL signs** from the PopSign ASL vocabulary,
trained on MediaPipe Holistic landmarks (full body: face mesh + pose + both hands).

## Input / output contract (reverse-engineered from the source repo)

- Input tensor `inputs`, shape `[num_frames, 543, 3]`, `float32`.
  - Each frame is 543 raw MediaPipe Holistic landmarks in this exact order:
    `face (468) + left_hand (21) + pose (33) + right_hand (21)`.
  - Missing landmarks (hand/face not visible that frame) must be `NaN`, not `0`
    — the model's preprocessing treats `0` as a real coordinate.
- Output `outputs`, shape `[250]` — already-softmaxed probabilities.
- Run with `tf.lite.Interpreter(...).get_signature_runner("serving_default")`.

## Missing piece: the label list

The model outputs a class index (0-249), not a word. The word list
(`sign_to_prediction_index_map.json`) is part of the competition's dataset and
isn't redistributed in the solution repo, so it isn't bundled here either.

To get it: create a free Kaggle account, install the `kaggle` CLI
(`pip install kaggle`), accept the competition rules on the
[data page](https://www.kaggle.com/competitions/asl-signs/data), get an API
token from your Kaggle account settings, then run:

```powershell
kaggle competitions download -c asl-signs -f sign_to_prediction_index_map.json
```

Drop the downloaded file at `models/word_model/labels.json`. Until then,
`run_live.py` falls back to numbered placeholder labels (`class_0`, `class_1`, ...)
so the pipeline is still testable end-to-end.
