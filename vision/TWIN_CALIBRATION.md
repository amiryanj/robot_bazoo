# Digital-twin calibration — phase summary & handoff (2026-06-14)

Goal of this phase: make the MuJoCo digital twin **metrically match the real arm** so
`scene_debug.py` is trustworthy — i.e. given encoder angles + camera detections, the twin
renders the true robot configuration and places detected objects at their real spots.
This is a **debugging/research aid, secondary** to the main VLA goal (see `CLAUDE.md`).

**Status: DONE and validated to ~2 mm.** All 5 arm joint offsets identified, the camera
pose yaw-corrected, both folded into the FK layer, and the whole chain validated end-to-end
at **~2.2 mm 3D RMS** across the workspace.

---

## The problem

LeRobot pins each joint's zero to an **eyeballed homing pose**, so the twin carried a
constant per-joint offset `δᵢ` between the encoder angle and the true kinematic (MuJoCo)
angle. Plus the wrist_roll horn sits ~−85° off the CAD zero. We identify all of these.

## What was done

1. **Joint-offset calibration as a GTSAM factor graph** (`vision/offset_calib.py`).
   Variables: 6 joint offsets δ, camera pose `T_cb` (Pose3), 2 finger-tag mounts. Factors:
   tag reprojection per sighting + priors. Marginal covariance quantifies observability.
   Diagram: `outputs/calib/factor_graph.html` (vis-network + MathJax).

2. **Two confounds had to be broken** (a tag-on-the-end-effector can't see them alone):
   - **wrist_roll (δ₅)** is confounded with the tag-mount rotation about the roll axis.
     Broken by **assumption B**: the tags are glued ~square to the fingers, so the mount
     rotation is known → tight rotation prior on the mounts (from `tag_calib.json`).
   - **shoulder_pan (δ₁)** is confounded with the **camera yaw**. The original handeye
     (heart-based) was itself FK-based, so it had the *same* ~10° yaw error baked in — it
     could not break it. **Broken by a base-rigid desk fiducial**: an ArUco tag laid flat
     on the plate with one edge on the pan=0 / y=0 line. Being rigid to the *base* (not the
     arm), it pins `T_cb`'s yaw + lateral independent of any joint → frees pan.
     (`vision/calib_overlay.py` visualizes/derives this; the desk-tag factor lives in
     `offset_calib.py`, fed by `outputs/calib/desk_tag.json`.)

3. **Promoted the yaw-corrected `T_cb`** into `handeye.json` (−10.08° yaw, 66 mm
   translation; old one backed up as `handeye_backup_*.json`).

4. **Folded the offsets into the FK layer** (`pick_ball.py`): the old wrist_roll-only
   `ROLL_DELTA_DEG` became `JOINT_OFFSETS` (all 5 arm joints), loaded from
   `offset_calib.json`, applied wherever angles feed MuJoCo (`Kin.fk`, `Twin.set`,
   `scene_debug.py`). **This does NOT touch lerobot control** — only the sim/twin reading
   of the reported angles. (Putting it in the lerobot calibration would shift every
   commanded pose physically, and wrist_roll −84° there would be catastrophic.)

5. **End-to-end validation — ball-as-marker sweep** (teleop, `vision/ball_cal_teleop.py`
   → `vision/ball_cal_analyze.py`). The ball is gripped (rigid to the gripper); recorded
   the 2 finger tags + detected ball at many poses; fit the constant ball↔tag geometry and
   measured the residual. Result: constant offset 7.6 mm (geometry), **residual ~2.2 mm 3D
   RMS, uniform across the workspace** (no spatial/depth bias). ~2 mm is the detection noise
   floor (sphere fit ±2 mm). `outputs/calib/ball_cal_residual.png`.

## Final numbers (`outputs/calib/offset_calib.json`)

| joint | offset (deg) | σ | note |
|---|---|---|---|
| shoulder_pan  | **−4.7** | 0.4 | desk-tag freed it (was −15.8 before) |
| shoulder_lift | **−6.1** | 0.8 | |
| elbow_flex    | **−6.1** | 0.6 | |
| wrist_flex    | **+0.1** | 0.6 | |
| wrist_roll    | **−84.2**| 2.7 | matches the independent tag_sweep value |
| gripper       | +0.8 | 3.6 | unobservable (jaw fixed); not applied |

Camera (`handeye.json`) yaw-corrected; full chain validated to ~2.2 mm.

---

## Files & artifacts

- `vision/offset_calib.py` — GTSAM solver (`--selftest` / `--collect` / solve / `--imu`).
- `vision/calib_overlay.py` — projects the base frame onto a camera frame; `--park` swings
  the arm to pan=−60 (clear of the desk-tag sightline); derives/visualizes the camera yaw.
- `vision/ball_cal_teleop.py` — teleop + Capture-button data collection (live Rerun view).
- `vision/ball_cal_analyze.py` — fits the constant offset, rejects outliers, residual + map.
- `outputs/calib/`: `offset_calib.json`, `handeye.json` (+backup), `tag_calib.json`,
  `desk_tag.json`, `offset_calib_samples.json`, `ball_cal_<ts>/` (raw color/depth/K),
  `factor_graph.html`, `ball_cal_residual.png`.

## Hardware/env notes (important for a new thread)

- **GTSAM env gotcha:** gtsam 4.2 (only PyPI build) is numpy-1 only and **segfaults with
  the lerobot env's numpy 2.x**. There is a dedicated conda env **`gtsam`** (numpy 1.26 +
  gtsam + mujoco + scipy). Run **collection in `lerobot`**, **solve in `gtsam`**. Do NOT
  downgrade numpy in lerobot.
- **Tags:** finger tags = `DICT_4X4_50` ids **1** (body `gripper` = wrist_roll link) & **2**
  (body `moving_jaw`). Desk tag = `DICT_ARUCO_MIP_36H12` id **8**, ~28 mm, ~32 px (detect by
  filtering id 8; "rightmost" gives false positives).
- **IMU not used** as a confound-breaker: its mount is only known via FK-Kabsch
  (`analyze_run.py`), which is circular with δ.

## Open loose ends (optional, none blocking)

- 3 ball detections in the sweep were gross outliers (YOLO box on background / depth holes).
  If they bite the pick pipeline, harden `ball_from_box` (reject low-inlier sphere fits).
- Could re-fit the sphere from the now-stored raw depth for a depth-bias study (the y axis
  was the noisiest at 1.5 mm RMS — still fine).

## Next (back on the critical path)

The twin is trustworthy now. Resume the main goal: scripted pick (`pick_ball.py`, now using
the corrected FK) → auto-record a LeRobot dataset → train ACT, then SmolVLA. See `CLAUDE.md`.
