# Dataset

## Source

- **Name:** exercise-rep-identifier (version 2)
- **URL:** https://universe.roboflow.com/student-t7wnl/exercise-rep-identifier/dataset/2
- **Platform:** Roboflow Universe
- **Workspace:** student-t7wnl
- **License:** MIT
- **Downloaded via:** the Roboflow Python SDK in `notebooks/03_custom_training.ipynb`.
  The API key is read with `getpass` at runtime and is never written into the notebook or
  committed to git.

## Contents

- **Task:** object detection
- **Classes:** 1 — `head`
- **Images:** 145 train / 38 validation / 22 test (205 total)
- **Format:** YOLOv8 (TXT annotations plus `data.yaml`)

## Why this dataset

It is a real public dataset in the exercise domain, licensed for reuse, and small enough
to fine-tune repeatedly inside the capstone window — a full run finishes in roughly two
minutes on a Colab T4, which made it possible to compare training configurations rather
than accept the first result.

**Its limitations are significant and shape how the model is used.** The dataset carries a
single class, so the fine-tuned detector confirms that a subject is present in frame; it
cannot identify which exercise is being performed. Exercise selection in the pipeline is
therefore configuration-driven, not model-driven. The validation split is 38 images, so
metrics move by several points when a handful of hard frames go the other way — the
numbers are indicative rather than precise.

A dataset with per-exercise classes (pushup / squat / plank) would let the detector select
the keypoint chain automatically. That is the clearest next step for this project.

## Preprocessing

None applied beyond the Roboflow export defaults. Images are resized to 640×640 by the
Ultralytics dataloader at training time.

Augmentation was varied deliberately between the two training runs:

| Run | Settings |
|---|---|
| run1_baseline | 25 epochs, `imgsz=640`, `batch=16`, Ultralytics default augmentation |
| run2_frozen_aug | same, plus `freeze=10`, `degrees=10.0`, `scale=0.6`, `fliplr=0.5`, `mosaic=1.0` |

## Not committed

The dataset is excluded by `.gitignore` (`datasets/`, `data/*/`). Re-download it by
running notebook 03 with a Roboflow API key.
