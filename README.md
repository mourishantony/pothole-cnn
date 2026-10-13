# Pothole & Road Damage Detection (CNN)

Classifier: EfficientNetV2-S (TensorFlow/Keras), 3 classes: crack / normal / pothole. Final model: `models/final.keras`.

| Test set | Accuracy | Macro-F1 |
|---|---|---|
| RDD2022 test crops (3,408) | 90.0% | 0.901 |
| Original dashcam test crops (1,150) | 77.4% | 0.757 |

## Run the website (GPU via WSL)
Double-click `start_app.bat`, then open http://127.0.0.1:5000 and upload a road video.
The video is split into tiles (30% of the road area), tiles are classified, neighbouring hits are merged into boxes.
Verdict = "pothole detected" if at least 3 sampled frames contain a pothole (>=2 agreeing tiles, prob >= 0.95).

## Pipeline
`download_data.py` -> `prepare_data.py` / `prepare_rdd.py` -> `train.py` (env: DATA_DIR, BACKBONE, TAG, RESUME, ...) -> `app.py`.
Training runs in WSL (`~/tfgpu`), since TensorFlow has no native-Windows GPU support.

## Limits
Trained on crops, not full frames; heavily cracked pavement can be flagged as pothole. Decision support only - needs human review.
Thresholds were tuned on unseen RDD2022 test frames (frame-level accuracy about 75%).
