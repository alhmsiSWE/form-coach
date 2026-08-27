## Team

- Fahad Abdullah Alanazi
- Saif Fawaz Alanazi
- Faris Turki AlRasheed
- Zayed Abdulrahman Aldosari

# Form Coach — Exercise Form Analysis with Ultralytics YOLO

A computer-vision application that watches a person exercising, counts their repetitions,
and exposes the joint angle behind each rep so incomplete range of motion is visible
rather than assumed.

Built for the SDAIA Academy capstone, **Computer Vision for Developers with Ultralytics**.

## What it does

The system runs two models over recorded video:

1. **Pose estimation** (`yolo26n-pose.pt`, pretrained) — extracts 17 body keypoints per
   frame. The angle at the driving joint (elbow for pushups, knee for squats) is computed
   from three of them and tracked across the clip.
2. **Object detector** (`yolo26n`, fine-tuned) — a single-class presence check confirming
   a subject is in frame before the angle series is trusted.

Repetitions are counted by `ultralytics.solutions.AIGym`, with the up/down thresholds
derived from the measured angle distribution rather than guessed. `model.track()` holds a
stable identity across frames, and `solutions.Heatmap` aggregates movement over the clip.

## Scope

- **Vision tasks used:** pose estimation, object detection, object tracking
- **Exercises supported:** pushups, squats
- **Input:** recorded video, side-on view, full body in frame
- **Output:** annotated video, rep count, per-frame angle CSV, angle-trace plot, ONNX model

**Known limitation:** the fine-tuned detector carries one class and acts as a presence
check. It does not classify which exercise is being performed — the keypoint chain is
selected by configuration (`EXERCISE` in notebook 01), not by the model.

## Pipeline

```
video ──> pose model ──> keypoints ──> joint angle ──> threshold sweep ──> AIGym ──> rep count
            │                                │                                        │
            └──> detector (presence check)   └──> angle CSV + trace plot              │
                                                                                       │
                                                        annotated video + heatmap <────┘
```

## Key components

| Component | Where | What it does |
|---|---|---|
| `joint_angle(a, b, c)` | notebooks 01, 02 | Angle in degrees at point `b` from three keypoints, via the dot product. The core form signal. |
| Frame sampling loop | notebook 01 | Processes `TARGET_FPS` frames per second instead of all of them, for a fast first pass. |
| `simulate_reps()` | notebook 02 | Replicates the `AIGym` two-state counter over the extracted angle array, so thresholds can be swept offline at no cost. |
| `solutions.AIGym` | notebook 02 | The real rep counter, inside an OpenCV capture → process → write loop. |
| `model.track(persist=True)` | notebook 02 | Stable identity across frames; without `persist` the tracker restarts on every call. |
| `solutions.Heatmap` | notebook 02 | Aggregates presence across the clip into a movement map. |
| `resolve_run_dir()` | notebook 03 | Reads the run directory from `results.save_dir` instead of hardcoding it — Ultralytics writes to `runs/detect/<project>/<name>/`. |
| `overfit_check()` | notebooks 03, 04 | Evaluates `split='train'` against `split='val'` as separate `val()` calls and reports the gap. |
| Threshold sweep | notebook 04 | Precision/recall across candidate confidence values, with the operating point chosen from cost asymmetry. |

## Configuration

Everything adjustable lives in a marked config cell near the top of each notebook.

| Setting | Notebook | Default | Meaning |
|---|---|---|---|
| `VIDEO_PATH` | 01, 02 | `../videos/pushups.mp4` | Input clip |
| `SIDE` | 01 | `left` | Which side of the body faces the camera. Far-side keypoints are occluded and their confidence collapses, so the wrong value makes the angle series noisy. Notebook 01 checks your choice against measured confidence and says so if it disagrees. |
| `EXERCISE` | 01 | `pushup` | `pushup` or `squat`. Selects the keypoint chain — shoulder/elbow/wrist or hip/knee/ankle. |
| `TARGET_FPS` | 01 | `5` | Frames per second sampled for the first pass |
| `MIN_CONF` | 01, 02 | `0.5` | Keypoints below this confidence are discarded before the angle is computed |
| `KPTS` | 02 | `[5, 7, 9]` | Keypoint indices, printed by notebook 01 — carry them across |
| `DOWN_ANGLE` / `UP_ANGLE` | 02 | `60` / `140` | `AIGym` rep-counting thresholds, chosen from the sweep table in section 4 |
| `MANUAL_COUNT` | 04 | — | Your own count of the reps in the clip; the ground truth for end-to-end evaluation |
| `ROBOFLOW_API_KEY` | 03, 04 | — | Entered at runtime via `getpass`, never written into the notebook or committed |

## Results

Fine-tuned detector, validation split (38 unseen images):

| Metric | Value |
|---|---|
| mAP50 | 0.585 |
| mAP50-95 | 0.375 |
| Precision | 0.620 |
| Recall | 0.526 |

Rep counting on `videos/pushups.mp4`: `AIGym` reported 3 reps, matching an independent
state-machine simulation over the angle series at thresholds 60°/140°.

### How training was run

Two runs were compared on the same dataset and split. The baseline used 25 epochs,
`imgsz=640`, `batch=16`, `patience=10` and default augmentation, with all layers
trainable. The second froze the backbone (`freeze=10`) and added rotation 10°, scale 0.6,
horizontal flip and mosaic. The baseline won on both mAP metrics (mAP50-95 0.375 against
0.325) and was selected; on 145 training images the frozen backbone left the detection
head too little capacity to adapt. The baseline early-stopped at epoch 22 with its best
weights from epoch 12, so the dataset — not training length — is the limiting factor.

Model selection was on validation mAP50-95, never training loss: a model can drive
training loss to zero by memorising and still be useless on unseen images.

### How evaluation was run

`model.val()` on the validation split for the headline metrics and the confusion matrix,
then `split='train'` against `split='val'` as two separate calls to measure the
overfitting gap honestly. The confidence threshold was chosen by sweeping candidate values
and reading the precision/recall trade-off, then pushed toward precision rather than the
F1 optimum: in this application a false positive lets a partial movement register as a
completed rep and misinforms the user about their form, while a false negative only costs
one repeated rep. Full numbers, curves and named failure modes are in
`notebooks/04_evaluation.ipynb`.

## Prerequisites

- Python 3.10+
- NVIDIA GPU recommended (CPU works, slower)
- ffmpeg, for H.264 re-encoding so output video plays inline in notebooks
- A free [Roboflow](https://roboflow.com) API key, to download the training dataset

```bash
pip install -r requirements.txt
```

## Dataset

[exercise-rep-identifier v2](https://universe.roboflow.com/student-t7wnl/exercise-rep-identifier/dataset/2),
Roboflow Universe, MIT licence. 145 train / 38 validation / 22 test images, one class.
Downloaded through the Roboflow SDK in notebook 03 — the API key is read via `getpass`,
never committed. Details and limitations in [`data/README.md`](data/README.md).

Demonstration footage in `videos/` was recorded by the team, so the pipeline is exercised
on real input rather than stock images.

## Model weights

Pretrained weights (`yolo26n-pose.pt`, `yolo26n.pt`) download automatically on first use.

Fine-tuned weights are **not committed** (excluded by `.gitignore`). To reproduce them,
run `notebooks/03_custom_training.ipynb` — the baseline run takes about two minutes on a
Colab T4 and writes `best.pt` into its run directory, then copies it to `weights/best.pt`.
`notebooks/04_evaluation.ipynb` finds that file in the same runtime, or reproduces the
baseline run itself if it is missing.

## How to run

```bash
git clone https://github.com/alhmsiSWE/form-coach.git
cd form-coach
pip install -r requirements.txt
```

Then run the notebooks in order. Each is self-contained and opens in Colab; select a GPU
runtime before running 03 or 04.

| Notebook | Purpose |
|---|---|
| `notebooks/01_pose_inference.ipynb` | Pose inference, keypoint extraction, joint-angle series |
| `notebooks/02_solutions_pipeline.ipynb` | Rep counting, tracking, heatmap — full OpenCV pipeline |
| `notebooks/03_custom_training.ipynb` | Fine-tuning the detector, two runs compared |
| `notebooks/04_evaluation.ipynb` | Validation metrics, threshold selection, ONNX export |

To run on your own footage: drop the clip in `videos/`, point `VIDEO_PATH` at it, set
`SIDE` and `EXERCISE`, and run notebook 01 — it prints the keypoint indices and the
suggested thresholds that notebook 02 needs.

## Expected output

Notebook 01 prints the keypoint tensor shape `(1, 17, 2)`, writes
`01_pose_annotated.mp4` with the skeleton and a live angle readout overlaid, and saves the
angle trace as a plot and a CSV. Each dip in the trace is one repetition; shallow dips are
reps that did not reach full range of motion.

Notebook 02 writes three videos — rep counter, tracking with a persistent ID, and the
heatmap — plus the threshold sweep table showing the rep count for every candidate
threshold pair.

Notebook 03 prints the two-run comparison, the training curves, the train/validation
overfitting gap, and a generated write-up of what happened.

Notebook 04 prints the validation metrics table, displays the confusion matrix and PR
curve, records the chosen operating threshold, and produces `best.onnx`.

## Repository structure

```
form-coach/
├── notebooks/          # 01–04, committed with output intact
│   ├── 01_pose_inference.ipynb
│   ├── 02_solutions_pipeline.ipynb
│   ├── 03_custom_training.ipynb
│   └── 04_evaluation.ipynb
├── videos/             # self-recorded footage (demo clip only)
├── data/               # dataset documentation
├── requirements.txt
├── .gitignore          # excludes weights, runs/, datasets and secrets
└── README.md
```

Model weights (`*.pt`, `*.onnx`), Ultralytics run directories (`runs/`), downloaded
datasets and any `.env` or key file are excluded from version control by `.gitignore`.

## Deployment

The selected model is exported to **ONNX** via `model.export(format="onnx")`, and notebook
04 runs inference through the exported file to confirm it works rather than only that it
was written. ONNX decouples inference from PyTorch, so the same file runs under
onnxruntime on CPU, in a browser, or on mobile — where a form-coaching app would
realistically live. TensorRT would be faster but ties deployment to NVIDIA hardware, and
inference is not the bottleneck here; the OpenCV capture loop is.

## Training program attribution

This project was completed under **Computer Vision for Developers with Ultralytics**, a
training program delivered by **SDAIA Academy** via Learning Space — a 5-day on-site
capstone, 30 training hours.

- **Cohort / session dates:**   23–27 August 2026
- **SDAIA Academy on GitHub:** https://github.com/SDAIAAcademy

ء

Repository: [github.com/alhmsiSWE/form-coach](https://github.com/alhmsiSWE/form-coach)
