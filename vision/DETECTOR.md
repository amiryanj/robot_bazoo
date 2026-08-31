> **UPDATE (2026-06-14): ball-only now.** The `heart_pink` class and the heart-detection
> path are retired — the gripper marker is AprilTags (finger ids 1,2), detected by ArUco,
> not a trained YOLO. `autolabel.py`/`detector_bench.py` are single-class (ball) and use a
> plain GDINO wrapper (`autolabel.Gdino`), not the deleted `HeartDetector`. The history
> below (2-class, heart augmentation, etc.) is kept for context.

# Scene detector: fast YOLO student, GDINO teacher

**Goal:** detect + localize *our* objects (mini basketball, heart markers) in 3D, with a
model fast enough for the real-time loop. RealSense depth gives the 3D once you know
*which pixels* — detection robustness is the whole game (plus `T_cam→base` from
`vision/HANDEYE.md` to land in arm coordinates).

## Why train our own

- Off-the-shelf basketball YOLOs (incl. our vendored `vision/models/basketball.pt` and
  the Roboflow Universe models) are **broadcast-domain**: full-size ball, court scenes.
  On our top-down mini ball against the white plate: conf 0.00–0.05. Unusable.
- No public heart-sticker/symbol detector exists (checked HF + Roboflow, 2026-06-11).
- **Grounding DINO works zero-shot on everything we ask** (ball 0.75+, hearts boxed
  reliably) but costs ~100 ms/frame on GPU and seconds on CPU — fine as a one-shot
  localizer or *teacher*, too slow as the in-loop *student*.

So: the standard distillation pattern. GDINO auto-labels our own frames; a `yolov8n`
(~3 M params, single-digit ms on the RTX 3000) learns the scene.

## Pipeline (all reusable as the dataset grows)

1. **Collect frames** — any mix of: saved captures under `outputs/vision/`, fresh
   RealSense grabs, exposure sweeps (`rs.option.exposure` sweep gives lighting
   diversity for free in a static scene).
2. **Auto-label**: `python vision/autolabel.py <frames_dir> <out_dir>` — GDINO
   "basketball." (conf ≥0.35, whole-table boxes rejected) → class `ball`; the gated
   pink-heart pipeline (retired) → class `heart_pink`. Writes YOLO-format
   `labels/` + an `overlays/` folder for **human spot-checking — always look** (the
   exposure sweep produced false hearts on a red cable clamp; caught because the static
   scene allows a consensus check across frames, and at the rest pose the heart isn't
   visible at all).
3. **Split + augment** (one-off script, kept in the 2026-06-11 dataset dir): split by
   base frame (no leakage), offline augment train only (h-flip, 90° rotation,
   brightness ×0.45/×1.6, blur) → ~200–400 images. Ultralytics adds mosaic/HSV on top
   during training.
4. **Train**: `yolov8n.pt`, 80 epochs, imgsz 640 — minutes on the 8 GB GPU.
5. **Benchmark**: `python vision/detector_bench.py <weights> <val_images>` — per-class
   agreement with GDINO (IoU>0.5) + ms/frame for both.

## Status 2026-06-11 (first iteration, overnight)

- Dataset: 45 base frames (24 from the day's sessions + 21 exposure-sweep), GDINO
  auto-labels: 35 ball / 19 heart_pink (after stripping 11 false sweep hearts).
  Train 204 / val 11 → `outputs/vision/dataset_2026-06-11/yolo/`.
- Weights: `vision/models/scene_yolov8n.pt` (see benchmark results in the repo
  history / commit message for the numbers of this iteration).
- **Known limits of v1** — single scene, single day, ball mostly in 2–3 spots, heart
  only at poses where it faced the camera. It will overfit to this scene; that's fine
  for the pick-place loop on this table, not for generalization. Next data session:
  move the ball + arm between captures, vary the holder, capture during teleop.
- **Tried and rejected (2026-06-11 night): copy-paste heart augmentation** — 68
  synthetic hearts (rotated/scaled/brightness-jittered crops pasted onto train frames)
  → zero gain on val heart recall (identical hits, slightly lower conf). The misses
  are poses the camera genuinely barely sees, not a sample-count problem. Fix is real
  varied-pose data: `vision/collect_marker_data.py` with the lights on. The val heart
  reference also inherits teacher false-positives (red clamp) — human-check val labels
  before reading the heart agreement number literally.

## 3D localization

Detection → 3D is already detector-agnostic: `ball_yolo.ball_from_box(box, conf,
depth, K, plane)` back-projects any box through the aligned depth (median of the
central quarter) and pushes the surface point in by one radius. `pick_ball.py` consumes
either detector through this one function; z comes from depth (works when the ball
sits on a holder), the RANSAC plane is a reference only — the scene has TWO planes
(white plate ≈1 cm above the wooden desk) and the fit may land on either.
