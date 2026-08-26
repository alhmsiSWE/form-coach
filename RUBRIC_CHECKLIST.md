# Form Coach — Rubric Checklist

Tick each line only when the artifact exists **and has real captured output**.

---

## Deliverable 1 — Core Vision Tasks & Inference (25 pts) · floor 10

- [ ] Pose model loaded via Python API (`YOLO("yolo26n-pose.pt")`)
- [ ] `model.predict()` run on **my own** video, not a stock image
- [ ] Keypoint tensor shape printed
- [ ] Annotated output video saved
- [ ] Notebook downloaded **with output intact**

→ `notebooks/01_pose_inference.ipynb`

---

## Deliverable 2 — Real-World Solution & Video Analytics (25 pts) · floor 10

- [ ] `solutions.AIGym` counting reps on my footage
- [ ] Real OpenCV pipeline: capture → process → write
- [ ] A second solution (Heatmap / Analytics / ObjectBlurrer)
- [ ] `model.track()` used somewhere
- [ ] Rep count printed and correct against a manual count

→ `notebooks/02_solutions_pipeline.ipynb`

**Manually count the reps in your own video and compare.** The gap between the two
is your best evaluation material — don't lose it.

---

## Deliverable 3 — Model Evaluation (25 pts) · floor 10

- [ ] `model.val()` run on the fine-tuned model
- [ ] mAP50 and mAP50-95 reported
- [ ] Confusion matrix saved and shown
- [ ] Failure modes named: where FPs happen vs FNs
- [ ] Confidence threshold justified using the Lab 4A cost framework
- [ ] Overfitting diagnosed **correctly** — Lab 4B's check is broken, don't copy it

→ `notebooks/04_evaluation.ipynb`

Threshold argument: a false positive counts a half-rep as real, so the app reports 10
when you did 7 and you go home thinking your form is fine. A false negative just means
repeating one rep. FP costs more → tune toward precision.

---

## Deliverable 4 — Custom Data & Training (15 pts) · no floor

- [ ] Real dataset, **not coco8** — URL written down in `data/README.md`
- [ ] `model.train()` actually run
- [ ] Real knobs touched: `epochs`, `imgsz`, `freeze`, augmentation
- [ ] At least two runs with different settings, compared
- [ ] Written up: what happened, what changed, why

→ `notebooks/03_custom_training.ipynb`

---

## Deliverable 5 — Deployment & Export (5 pts) · no floor

- [ ] `model.export(format="onnx")` run, file produced
- [ ] Target format named and briefly justified
- [ ] Small Streamlit/Gradio app

→ `notebooks/05_export_deploy.ipynb`, `app/streamlit_app.py`

---

## Deliverable 6 — Documentation & Evidence (5 pts) · no floor

- [ ] Every notebook committed **with output**
- [ ] README: what it does, prerequisites, setup, how to run, expected output
- [ ] Pipeline overview: models, dataset, how training and eval were run
- [ ] Training program name **and cohort/session dates**
- [ ] Link to https://github.com/SDAIAAcademy
- [ ] `.gitignore` excluding `runs/`, `*.pt`, `datasets/`, secrets
- [ ] Meaningful, incremental commit history — not one bulk upload

---

## Traps

1. **Unexecuted notebooks score nothing.** Download from Colab *after* running.
2. **coco8 doesn't count** as custom training.
3. **Keywords don't score** — only real `predict` / `track` / `val` / `train` / `export` calls.
4. **Lab 4B's overfitting check is broken.** It assigns `train_map` and `val_map` the same
   value, so it always reports no overfitting. Use `split='train'` vs `split='val'`.
5. **Lab 3's results path is wrong.** With `project=` set, output goes to
   `<project>/<name>/`, not `runs/detect/<project>/<name>/`.
6. **One bulk commit** loses easy points.

---

## Order of work

Deliverables 1–3 are 75 points and each has a 40% floor. Get a thin working version of
all three **first**, commit, and only then go deeper. Perfecting the training run while
the evaluation notebook doesn't exist is how this goes wrong.
