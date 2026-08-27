# Videos

Demonstration footage the pipeline is run against.

Large files are gitignored. Only the single demo clip below is committed, force-added:

```bash
git add -f videos/pushups.mp4
```

## Source

`pushups.mp4` is a free sample clip supplied as course material for **Computer Vision for
Developers with Ultralytics** (SDAIA Academy). It was **not recorded by the team**. It is
kept in the repository so the notebooks reproduce end to end without an external download;
check with the course before reusing it elsewhere.

## What the clip contains

One clip, one subject, no deliberately incomplete reps.

- A single person performing a **vertical pulling movement (pull-ups) on a gym rack**,
  filmed from **directly behind** the subject — the back of the torso faces the camera,
  both arms visible, full body in frame for the whole clip.
- Camera is static, held low (roughly floor level, angled slightly up), gym interior,
  even artificial lighting.
- **Four elbow flexion/extension cycles** in 9.4 s. The elbow angle sweeps roughly
  15°–167° (notebook 02, section 3). `AIGym` reports **3 reps** at the selected 60°/140°
  thresholds — the fourth extension completes as the clip ends. The threshold sweep in
  notebook 02 shows 4 at `up_angle=120`, which is where that gap comes from.

**Naming caveat:** the file is named `pushups.mp4` and the notebooks run with
`EXERCISE = "pushup"`. That selects the shoulder/elbow/wrist keypoint chain
(`KPTS = [5, 7, 9]`), which is the correct chain for the elbow-driven movement in this
clip, so the rep counting is valid — but the clip is not a pushup, and it is not the
side-on view the pipeline is designed around. A genuine side-on pushup or squat clip
would exercise the `SIDE` configuration as intended.

## Technical details

| Property | Value |
|---|---|
| Resolution | 1080 × 1920 (portrait) |
| Frame rate | 30 fps |
| Length | 283 frames, 9.43 s |
| Codec | H.264 video, AAC audio |
| File size | 7.4 MB |

Verified from the MP4 container headers and from the `cv2.VideoCapture` properties printed
by notebook 01 (`Resolution : 1080 x 1920`, `FPS : 30.0`, `Frames : 283`).

**Recording device, camera distance and camera height are not documented** because they
are not recoverable: the container carries a transcoder handler tag and no camera metadata,
and the clip did not originate with the team.

## Using your own footage

Drop the clip in this directory, point `VIDEO_PATH` at it in notebooks 01 and 02, and set
`SIDE` and `EXERCISE`. Record side-on with the full body in frame — notebook 01 checks
`SIDE` against measured keypoint confidence and tells you if the far side is occluded.
