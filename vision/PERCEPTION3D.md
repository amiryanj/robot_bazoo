# 3-D perception stack: design & plan

**Goal:** robust 3-D localization of workspace objects from the top-down RealSense,
designed so new objects are added by plugging parts together — not by writing a new
pipeline per object. Status 2026-06-12: layer built (`vision/cloud.py`), ball localizer
migrated onto it, benchmarked.

## Layering

```
2-D detector            3-D fitting                scene model            consumers
(what & where in img)   (where in space)           (persistent context)
─────────────────────   ────────────────────────   ────────────────────   ─────────────
yolov8n student    ─┐   cloud.deproject(box)       extract_planes()       pick_ball.py
GDINO (fallback /   ├─► cloud.fit_* with a    ──►  plate / desk planes ─► scene_debug.py
      teacher)     ─┘   geometric PRIOR            + boundaries           future: state
                        + quality record           (per camera setup)     machine, VLA
```

Principles, each one paid for by a real failure:

1. **Detection and metric localization are separate concerns.** The detector only
   answers "which pixels". All 3-D comes from depth + geometry. (Detectors got swapped
   twice already — YOLO→GDINO→student — and the 3-D side never changed.)
2. **Known dimensions are constraints, not estimates.** The Ø49 mm ball's radius is an
   input to the fit (3 unknowns instead of 4). Box-size radius estimates wobbled ±3 mm
   and fed straight into the centre error.
3. **Never trust a single statistic of raw depth.** Box-median depth was biased 1–3 cm
   toward the background by silhouette bleed (stereo mixes disparities at object
   edges; background is always *deeper*). Model fitting rejects those pixels because
   they don't satisfy the prior; benchmark: legacy z was 7–8 mm low and 13 mm off in y
   — about one finger-gap, i.e. the difference between grasp and miss.
4. **Every fitter returns a quality record** (`inliers`, `rms`) and consumers gate on
   it. A localization the fitter can't certify aborts the grasp instead of feeding it.
5. **The scene has structure worth modelling once, not per frame:** two horizontal
   planes (white plate ~1 cm above the wooden desk). Single-plane RANSAC fits a
   mongrel between them depending on pixel counts — `extract_planes` (sequential
   RANSAC) separates them and reports support + extent for each.

## What exists (vision/cloud.py — nothing ball-specific)

- `deproject(depth, K, box=None)` — depth → camera-frame cloud, cached pixel grids.
- `fit_plane`, `extract_planes(max_planes)` — RANSAC + sequential peeling; per plane:
  normal, offset, inlier count, centroid, extent.
- `fit_sphere_known_r(points, r)` — RANSAC (hypotheses: surface point + r along its
  ray) + Gauss-Newton refine; returns centre + inliers + rms.
- `nearest_depth_center(points, r)` — cheap fallback: object top = nearest depth
  percentile (bleed-immune), centre = top + r along the ray.

`ball_yolo.ball_from_box` is now: detector box → padded cloud crop → sphere fit →
quality gate → fallback. Consumers (`pick_ball`, `scene_debug`) unchanged in shape.

## Benchmark protocol (vision/bench_localize.py)

Replay saved sessions where the ball was physically static; report per-method spread
(precision) and mean (compare against known resting geometry for trueness).
2026-06-12 results, 9-pose session, ball on the ~30 mm spool (true centre ≈ +12 mm):

| method | std xyz (mm) | mean z | z error |
|---|---|---|---|
| legacy box-median | 0.1 / 1.1 / 2.6 | +5 | −7 mm (bleed bias) |
| sphere fit (r known) | 1.5 / 3.2 / 1.9 | +13 | ≈ +1 mm |

Plus a 13 mm mean y disagreement — visually settled by projecting both centres onto
the image: the sphere-fit centre sits on the ball, the legacy centre near its edge.
Sessions for replay must save `depth_*.npy` and intrinsics (the exposure-sweep set
saved no K — lesson encoded in `collect_marker_data.py`, which saves both).

## Adding a new object (recipe)

1. Detector: add a prompt to GDINO (slow path) and/or a class to the student via
   `autolabel.py` → retrain (minutes).
2. Prior: pick/write the fitter — sphere exists; cylinder (the spool!) and box are the
   obvious next two, each ~40 lines in `cloud.py` following the same contract
   (`points, known dims → dict(pose, inliers, rms) | None`).
3. Consumer: detector box → `deproject` → fitter → gate on quality → `R_cb @ p + t_cb`.

## Roadmap

- [ ] **Joint-mapping layer** (blocked on the ArUco tags being printed): one shared
      real-degrees↔model-qpos conversion used by FK/IK/viewers, with per-joint
      offset/sign measured by a tag sweep — settles the wrist_roll question with a
      number and re-certifies the other joints.
- [ ] **World-anchor tag** (Javad's design, 2026-06-12): a fixed ArUco (tag id 2,
      48 mm) glued to a static spot. Live T_cam→tag per frame + one-time T_tag→base
      → camera moves/bumps stop invalidating anything; plane map and calibration key
      to the tag, not to camera stillness. For tag work bump the COLOR stream to
      1280×720 (doubles px/tag; independent stream) but KEEP DEPTH at 640×480 —
      min-Z grows with depth resolution (~0.5 m at 720p = table at the blind edge).
- [ ] **Plane persistence**: save `extract_planes` output (keyed to the world tag /
      handeye timestamp) to `outputs/calib/scene_planes.json`; consumers load instead
      of refitting.
- [ ] **3-D coarse proposals** (detector-free): voxel-downsample → subtract known
      planes → Euclidean clustering → "blobs above the table" as object candidates,
      verified by the fine fitters. Lighting-independent (measured: depth quality in
      the pitch-dark night session equals daylight — the D455's IR projector brings
      its own texture). Complements the 2-D detector for unknown objects.
- [ ] **Scene model as a contract** (Javad's design, 2026-06-12): a periodic "scene
      scan" writes `outputs/calib/scene_model.json` — planes classified as support
      surfaces (|n·ẑ|>0.95 in base frame, with height + boundary hull) + object cards
      (class, pose, footprint, resting plane). Producers: GDINO (names) → SAM masks
      (FastSAM/MobileSAM ship in our ultralytics — no new deps) → mask-cropped cloud →
      fitters. Consumers: the MuJoCo twin / scene_debug render whatever the file says
      (debug value now); later, captured scenes become **sim assets for domain-
      randomized VLA episode generation** — the bridge from perception to the
      project's actual goal. Note: SAM is class-agnostic — it segments, GDINO names;
      they're a pair, not alternatives. **Measured 2026-06-12** (FastSAM-s via our
      ultralytics, box-prompt API): mask-cropped cloud = 93% sphere-fit inliers vs 51%
      for box+pad, centre shift only 1–2 mm — the known-radius fit already rejects
      clutter, so the ball path stays mask-free (simpler); masks become the default
      crop for future objects whose fitters can't self-clean as decisively.
- [ ] Cylinder fitter (spool/holder) — makes "ball on holder" a modelled fact instead
      of a special case, and is the second test of the recipe above.
- [ ] Plane *boundaries* (hull polygons in plane coordinates) — workspace limits and
      placement targets for the pick-place state machine.
- [ ] Wrist-cam integration during grasp (after the USB bandwidth fix) — close-range
      verification of the final 5 cm, where the top-down depth is weakest (D455
      min-Z ≈ 0.4 m).
