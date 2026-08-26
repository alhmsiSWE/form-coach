Fahad Abdullah Alanazi
Saif Fawaz Alanazi
Faris Turki AlRasheed
Zayed abdulrahman Aldosari

# Form Coach — Exercise Form Analysis with Ultralytics YOLO

> **TODO before submitting:** fill in every `TODO` marker below.

A computer-vision application that watches a person exercising and reports how many
repetitions they completed and whether each rep reached full range of motion.

The system combines two models:

1. **Pose estimation** (`yolo26n-pose.pt`, pretrained) — extracts 17 body keypoints per
   frame, from which the joint angle driving the movement is computed.
2. **Exercise detector** (fine-tuned) — identifies which exercise is being performed, so
   the pipeline automatically selects the correct keypoint chain instead of hardcoding it.

Repetitions are counted with `ultralytics.solutions.AIGym`. The joint-angle time series
is the form signal: reps that fail to reach the depth of the others are flagged as
incomplete range of motion.

## Scope

- **Tasks used:** pose estimation, object detection, object tracking
- **Exercises supported:** pushups, squats
- **Input:** recorded video (side-on view, full body in frame)
- **Output:** annotated video, rep count, per-rep angle trace, CSV of angle data

## Pipeline

```
video ──> pose model ──> keypoints ──> joint angle ──> AIGym ──> rep count
            │                                                       │
            └──> exercise detector ──> keypoint chain selection ────┘
                                                                    │
                                             annotated video + CSV <┘
```

## Prerequisites

- Python 3.10+
- NVIDIA GPU recommended (CPU works but is slow)
- `pip install ultralytics opencv-python pandas matplotlib streamlit`
- ffmpeg (for H.264 re-encoding so output plays in notebooks)

## Model weights

Pretrained weights download automatically on first use. The fine-tuned detector is
produced by `notebooks/03_custom_training.ipynb`; trained weights are not committed
(see `.gitignore`).

TODO: state where your `best.pt` can be obtained or how to reproduce it.

## Dataset

See [`data/README.md`](data/README.md).

## How to run

```bash
git clone <TODO: your repo url>
cd form-coach
pip install -r requirements.txt
```

Then run the notebooks in order:

| Notebook | Purpose |
|---|---|
| `01_pose_inference.ipynb` | Pose inference, keypoints, joint-angle extraction |
| `02_solutions_pipeline.ipynb` | Rep counting and video analytics |
| `03_custom_training.ipynb` | Fine-tuning the exercise detector |
| `04_evaluation.ipynb` | Validation metrics and threshold selection |
| `05_export_deploy.ipynb` | ONNX export |

Web app:

```bash
streamlit run app/streamlit_app.py
```

## Expected output

TODO: describe what the user should see — annotated video with skeleton overlay,
rep counter, angle readout; plus the CSV and angle-trace plot.

## Results

TODO: fill in after evaluation.

| Metric | Value |
|---|---|
| mAP50 | |
| mAP50-95 | |
| Precision | |
| Recall | |
| Chosen confidence threshold | |

## Training program attribution

This project was completed under **TODO: exact program name**, delivered by
**SDAIA Academy** via Learning Space.

- Cohort / session dates: **TODO**
- SDAIA Academy on GitHub: https://github.com/SDAIAAcademy

## Author

TODO: your name and GitHub handle.
