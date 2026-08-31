> **SUPERSEDED (2026-06-14).** The pink-heart marker is gone and the tool is
> retired (kept only for its `Realsense` class). Camera-pose calibration is now
> reference-anchored in `vision/cam_calib.py` (white-plate plane + desk tag id 13); recover
> after a camera move with `cam_calib.py recal`. Kept for historical context.

# Hand-eye calibration (eye-to-hand): how it works and why

**Result (2026-06-11):** `T_cam→base` solved at **4.6 mm RMS** over 16 poses →
`outputs/calib/handeye.json`. Camera sits 471 mm above the base origin, optical axis
7° off straight-down. Tool: `vision/handeye_calib.py` (deleted; see `vision/cam_calib.py`).

## The problem

The top-down Realsense gives ball positions in the **camera frame**; IK needs them in the
**arm-base frame**. They're related by one rigid transform (the camera and base are both
bolted down):

```
p_base = R · p_cam + t
```

Solving for `(R, t)` is *eye-to-hand* calibration (fixed camera watching a moving arm —
as opposed to *eye-in-hand*, camera on the wrist).

## The method

Put a marker on the arm, drive the arm to N poses, and at each pose measure the same
physical point two ways:

- **camera side:** detect the marker pixel `(u,v)`, back-project through the aligned
  depth → `p_cam` (median depth in a window, for robustness);
- **arm side:** read joint angles → MuJoCo FK of the `gripper` body → `(R_w, t_w)` in the
  base frame.

If the marker's offset `o` in that body frame were known, each pose would give
`R·p_cam + t = R_w·o + t_w` directly. We don't know `o` (it's a sticker, not a machined
feature), so we **solve for it jointly**:

```
min over (R, t, o):  Σᵢ ‖ R·p_camᵢ + t − (R_wᵢ·o + t_wᵢ) ‖²
```

9 unknowns (rotation vector 3, translation 3, offset 3), 3 equations per pose → well
over-determined with 10+ poses. `scipy.optimize.least_squares` (LM), seeded at 4 yaws of
a 180°-flip-about-X (a top-down camera is roughly that), keep the best basin. This is why
**pose diversity matters**: translation spread conditions `(R, t)`; orientation spread
(wrist_flex / wrist_roll variety) is what separates `o` from `t` — with identical
orientations the two are degenerate.

Sanity gates: a synthetic `--selftest` (recovers a known transform with 1 mm noise:
0.09° / 1.6 mm / RMS 1.4 mm), and on the real run the per-pose residual list — one pose
far above the rest = a mis-detection to drop.

## Why the marker is a sticker, and why detection is learned

- **Marker:** a pink heart sticker on a gripper finger. Anything rigid w.r.t. a body we
  can FK works, because the offset is solved, not measured. One constraint follows: the
  finger moves with the gripper joint, so the **gripper opening stays fixed during the
  run** (then the heart is rigid w.r.t. the wrist_roll body and the constant-offset model
  holds). No ArUco print, no machined target.
- **Detection = learned shape + in-box color.** Whole-frame HSV segmentation failed
  exactly the way the ball attempt failed before it (CLAUDE.md): the scene is full of
  color impostors (piano keys, wooden table, red arm plastic). Instead, **Grounding
  DINO** (`grounding-dino-tiny`, zero-shot prompt `"heart."`, threshold 0.12) proposes
  heart-shaped boxes — it happily boxes the basketball and a tire too — and a color gate
  *inside each box* picks the marker. Color classification within a verified shape is
  reliable; color segmentation across a cluttered frame is not.
- **The pink gate is two-ended.** Pink wraps around the HSV hue circle: the same sticker
  measured H 160–166 in cool light and H 0–2 in warm light. So the gate accepts both hue
  ends — and separates the sticker from the red arm plastic by **saturation** (pale
  sticker S 75–103 vs plastic S 136–180; cap at 130), not by hue, which is what actually
  survives lighting changes. Plus size (≤70 px, stickers are small top-down) and
  pink-fraction (≥0.10 of box area) gates. Every threshold here was measured on captured
  frames (`vision/heart_detect_test.py`), not guessed.
- **Detector runs in a background thread.** GDINO on CPU takes seconds per frame; inline
  it stalled the delta-based teleop to uselessness. Same pattern as `station.py`'s IMU
  and ball-detection threads: control loop at full rate, preview lags behind, capture
  uses the latest detection. (On CUDA it's near-real-time anyway.)

## Running it

```bash
conda activate lerobot
python vision/handeye_calib.py --selftest   # offline math check, no hardware
python vision/handeye_calib.py              # live: arm + realsense + gamepad
```

Drive with the gamepad; the Rerun window shows every heart candidate box (labelled
`conf pink=frac`) and a **green dot on the picked marker**. Typed commands: `c` capture,
`u` undo, `q` solve & save. Collect 10–16 poses: spread over the image, vary depth and
wrist orientation, keep the gripper opening fixed, only capture when the green dot is on
the heart. RMS under ~10 mm is good for ball-picking; inspect per-pose residuals for
outliers.

## Using the result

```python
import json, numpy as np
d = json.load(open("outputs/calib/handeye.json"))
R, t = np.array(d["R"]), np.array(d["t"])
p_base = R @ p_cam + t        # p_cam from e.g. vision/ball_yolo.py
```

Re-run the calibration whenever the camera or the arm base moves.
