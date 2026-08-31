# Tag teleop: driving the SO-101 with a tagged joystick

**Goal.** Move the arm by moving your hand. A plain USB webcam watches a **3-face tag box**
(a 32 mm half-cube, one `DICT_4X4_50` tag per face, ids 7/11/15, 28 mm squares) carried on a
hand-held Nintendo Switch Pro Controller; the box's 6-D pose drives the TCP position + wrist
roll, and the gamepad's own buttons carry the clutch and the gripper.

Why a box and not flat tags: a single planar tag has **two** valid PnP solutions (the planar
two-fold ambiguity), and the tracker can sit on the mirrored one for a whole stretch --
self-consistent, and wrong. Two visible faces are not coplanar, so the pose is unique. The
box shows 2+ faces in **75.5%** of frames; the two loose tags it replaced managed 0.4%.

This exists because there is no leader arm and stick teleop is too clumsy for
pick-and-place (see CLAUDE.md, "Data strategy"). It is a **demo-collection input device**,
not the end goal.

## Layers (each file has one job, and imports the one below)

```
teleop_tag.py        the loop: clutch, mapping, safety rails, IK, Rerun, CSV
  ├─ pad_imu.py      the controller's OWN 6-axis IMU  (evdev) + its extrinsic calibration
  ├─ vision/tag_body.py   rigid multi-tag body model -> one body pose from any visible tag
  │    │                  (also VirtualCam: renders a tag body, so the loop runs with
  │    │                   no camera and no hand -- `--source virtual`)
  │    └─ vision/tag_pose.py   camera + ArUco detector + single-tag PnP  (the base layer)
  ├─ vision/box_tags.py   builds the box model: ident -> capture -> fit -> show
  ├─ pick_ball.py    Kin (MuJoCo FK/IK), Twin, read_angles, move_to, MOTOR_NAMES
  ├─ gamepad_utils.py  button_index, filtered_axis_value, graceful_shutdown, REST_POSE
  ├─ station.py      JoystickManager (hot-plug)
  └─ sim_backend.py  SimRobot, for --sim
```

Nothing here re-implements a layer below it. `pick_ball.Kin` is the single IK; the single
ArUco wrapper is `tag_pose.detect`; the single landing path is `gamepad_utils.graceful_shutdown`.

## The mapping

**Clutched relative**, like lifting a mouse. Hold **L** to engage: the current hand pose
and the current TCP are latched as an anchor, and from then on
`tcp = anchor_tcp + gain * M @ (hand - anchor_hand)`. Release L and the arm holds while you
reposition your hand. `--gain` is the ratio of arm motion to hand motion (1.0 = 1 cm of
hand gives 1 cm of arm).

`M` is set by `--axes` (default `-z,x,-y`): camera **z** (depth) drives base **x**, camera
**x** drives base **y**, camera **y** drives base **z**. The x sign is inverted because the
camera faces you and therefore mirrors you.

**wrist_roll is free.** `Kin._ik_pass` never writes `dq[4]`, so roll costs the position
solution nothing — it is the one hand-orientation DoF a 5-DoF arm can follow without
trading away position. It is driven by the twist of the controller about the camera's
viewing axis, as a rotation vector (no gimbal), gyro-predicted and vision-corrected on
`TWIST_TAU`.

## Safety rails (in the order they act)

| rail | constant | what it stops |
|---|---|---|
| reprojection gate | `MAX_REPROJ` 4.0 px | sloppy / ambiguous body poses |
| outlier drop + EMA | `JUMP_MAX`, `TAU` | single-frame perception glitches |
| TCP speed cap | `MAX_TCP_SPEED` | the target outrunning the arm |
| leash | `LEASH` 40 mm | target windup when the arm falls behind |
| joint jump gate | `MAX_JOINT_JUMP` 12° | IK branch flips reaching the servos |
| damping ladder | `DAMP_LADDER` | the jump gate deadlocking (see below) |
| blind timeout | `BLIND_MAX_S` 0.4 s | commanding from stale/dead-reckoned pose |
| landing | `graceful_shutdown` | the arm collapsing on exit |

The smoothing is **dt-aware** (`alpha = 1 - exp(-dt/tau)`). Rates expressed per *tick* are
wrong on a loop whose rate swings 3–17 Hz, which this one does.

**Why the damping ladder exists.** A bare jump refusal deadlocks: with `ang` unchanged the
next tick re-solves from the same state, gets the same answer, and refuses again — 94
consecutive refusals / 7.4 s with the arm at a standstill (2026-08-20 log). Walking up the
damping ladder asks the same question with a shorter step instead of giving up.

## Sensor fusion: tags global, IMU local

The tags give **absolute** pose but go blind; the controller's IMU gives **relative** motion
at ~77 Hz of new information but drifts. A **linear Kalman filter** fuses them —
9 states: position, velocity, **accelerometer bias**.

Linear is enough on purpose. Orientation is deliberately kept *out* of the state (it comes
from the tags / gyro propagation), so the remaining model is exactly the double integrator
`p' = v, v' = a - b` — linear, no EKF/UKF/ESKF needed. The bias state is what matters: a
constant bias *integrates* into position while white noise averages out, and without it the
filter was 16 mm wrong while reporting 2.9 mm confidence.

**The gyro was reading 12% high.** Before any of this could work, `hid_nintendo`'s reported
scale of 14247 units/deg/s turned out to be wrong -- it should be ~16250. Found twice,
independently: vision (tag rotation vs gyro, ratio 0.876) and gravity alone (still-to-still
accelerometer transitions, no camera, 0.875). `pad_imu.py gyrocal` measures scale and bias
together against gravity, in 60 s, with no camera. They must be solved jointly: fitting
either alone absorbs part of the other (scale alone left 10.8 deg of residual, both left
1.7 deg). Uncorrected, this alone moved `X` by tens of degrees.

`pad_imu.py` calibrates the one fixed rotation `X` (`v_body = X @ v_imu`):
- `align` — from a hand-waving recording, pairing **rotation increments** over 0.35 s
  (not finite-differenced angular velocity, which amplifies 20 Hz pose noise ~25×).
- `check` — gravity-based, `R_k @ X @ a_k = g_cam`, alternating solve + Huber IRLS,
  with bootstrap uncertainty and a **coverage** score (gravity in one orientation pins
  only 2 of 3 DoF, so coverage must be reported, not assumed).

`align` also refuses a bias measured while the pad is moving (it checks the stillness sd),
and shows **live axis coverage** while recording: `X` is only observable in the directions
you actually rotate about, and the pose count cannot show you that. The three characters
are the eigenvalues of `sum(w_hat w_hat^T)` -- `[#..]` means one axis only (underdetermined),
`[###]` means well spread.

State 2026-08-31, after the box and the gyro fix:

| | before | after |
|---|---|---|
| run-to-run agreement of `X` | 43° | **1.34° avg, 1.81° worst** (4 runs) |
| live camera-vs-gyro gap, 2+ tags | — | **1.37°** |
| gravity residual | 14.9° | **1.5°** |

### How far the IMU can be trusted

`teleop_tag.py --selftest` drives the real filter with a known hand path and realistic
errors -- 1.4° extrinsic, accel bias, gyro bias, corner noise, dropouts -- and reports error
against **gap length**, because a single number hides the shape:

| blind for | p90 error | filter's own sigma |
|---|---|---|
| with vision | 2.3 mm (median) | — |
| 0.00–0.15 s | 8.2 mm | 9.1 mm |
| 0.15–0.30 s | 27.3 mm | 26.6 mm |
| 0.30–0.50 s | 77.2 mm | 53.1 mm |

The growth is physics, not a bug: a `d`-degree tilt leaks `sin(d)*9.81` m/s² into the
measured acceleration, and that integrates as `½at²` -- so the error must roughly quadruple
when the gap doubles, and it does. **Practical limit: trust the IMU for ~0.15 s of
blindness, marginally to 0.3 s, not beyond.** The filter's sigma tracks the true error
closely, so `bridge_mm` is a sound stopping rule -- and the selftest asserts that honesty
rather than asserting a number, since an optimistic sigma is the failure that hurts.

## Measured performance (2026-08-21, `--sim`)

Per-stage tick timing is logged to `teleop.csv` (`ms_grab`, `ms_detect`, `ms_ik`,
`ms_twin`, `ms_rerun`, `ms_rest`). Guessing at this from the outside cost a full round of
debugging; do not do it again.

| stage | before | after | note |
|---|---|---|---|
| idle `ms_ik` | 168 ms | **0.03 ms** | IK was solving a frozen target and discarding it |
| engaged `ms_ik` | 198 ms | ~43 ms | fixed-point break, see below |
| `ms_twin` | 8.3 ms/tick | ~half | `TWIN_HZ` 10; `viewer.sync()` blocks on the render thread |
| `ms_detect` | 28.8 ms | unchanged | **now the largest single cost** — full 1280×720 + subpix |

Hardware is not the constraint: i9-13900H, 14 cores, 24 MB L3, 33 GFLOP/s single-thread
numpy. One IK iteration is 0.252 ms; the loop was doing ~600–900 of them per tick for a job
that needs ~30.

## Findings (2026-08-21)

**1. IK ran while the clutch was released.** 76–88% of all solves, at 135–230 ms each, with
the answer thrown away. The arm is *holding* while idle — there is nothing to solve. Roll
and gripper are direct joint commands and need no IK.

**2. An unreachable frozen target can never be rescued by damping.** After a disengage the
target stays wherever it was, often past the reach limit; the ladder then walked all five
rungs to the same failure (409 ms measured). `HOPELESS` (30 mm) short-circuits it: that
much error is unreachability, not discontinuity.

**3. `_ik_pass` ran provably-dead iterations.** With a joint pinned at its limit the update
is clipped back every iteration, so FK, Jacobian, step and clip all repeat identically. On
the observed pinned pose, iterations **50–150 of every pass were literal no-ops** with the
cost frozen at 0.343078. Breaking on that fixed point is lossless by construction — verified
bit-identical over 60 solves (max joint difference 0.0000°, perr difference 0.000 mm) —
and 4.4× faster on the pinned pose (188.5 → 43.0 ms).

**4. `wrist_flex` ratchets toward its 95° limit** (all four logged runs: 79.9°, 82.1°,
52.2°, 95.0°). It is a *symptom*: `Kin.ik` defaults `approach_dir` to a grasp-oriented
"radially outward, pitched down" preference inherited from `pick_ball.py`. As the operator
drives the TCP inward and upward the arm folds, and holding that pitch forces wrist_flex to
saturate. A teleop-appropriate orientation preference is the open fix.

**5. The drift you feel is a ratchet, caused by tag blindness — not by the code.**
*(Fixed 2026-08-31 by the tag box; kept because the measurement is how it was found.)*
Measured over one 95 s session:

| hand motion | net displacement | path |
|---|---|---|
| while **ENGAGED** | [+203, −120, **+277**] mm | 1573 mm |
| while **IDLE** | [+94, +112, **−280**] mm | 1594 mm |

The outward and return strokes cancel almost exactly, but **half the hand motion (1594 of
3167 mm) happened while disengaged and was discarded**. The disengages were not deliberate:
28 blind gaps, mean 0.80 s, max 5.5 s, **68% longer than the 0.4 s timeout**. The operator
never sees it happen mid-stroke.

Root cause: **`tags == 2` occurred in 0% of every run logged** — the second tag was never
once seen, so exactly one hand orientation was trackable. A tag-placement problem, not a
timing one. **Fixed** by the 3-face box: 2+ tags now appear in 75.5% of frames. If a gap
still catches you mid-stroke, press **A** to ease back to the ready pose.

## Known issues / next

- [x] ~~**Tag coverage** — the top blocker.~~ Fixed by the 3-face box: 0.4% → 75.5%.
- [x] ~~**Tag id collision**~~ — the box uses ids 7/11/15, clear of the finger tags (1, 2)
      and the desk anchor (13).
- [ ] **Not yet re-tested on the real arm since the box.** Everything above is measured,
      but the last real-arm run predates the new marker and the gyro fix. Do a `--sim` run
      with the clutch first.
- [ ] Teleop-appropriate `approach_dir` so `wrist_flex` stops saturating (finding 4).
- [ ] `ms_detect` 28.8 ms — detect at half resolution, refine at full. Note the camera
      itself costs 33 ms/frame at 30 fps, so this is not the whole story: measured
      grab 33.4 ms, detect 11.0 ms, jpeg-for-Rerun 4.5 ms at 1280×720.
- [ ] Rerun retains the whole recording by design (~700 B per scalar point × ~36 series).
      Not a leak in our code or MuJoCo — verified RSS-flat with logging off.

## Running it

```bash
conda activate lerobot
# no hardware at all -- rendered tag box, MuJoCo arm:
python teleop_tag.py --sim --source virtual
# real webcam + box, MuJoCo arm:
python teleop_tag.py --sim --gain 1.0 --max-speed 0.20
# real arm:
python teleop_tag.py --gain 1.0 --max-speed 0.20
# check the fusion maths, no camera / IMU / arm:
python teleop_tag.py --selftest
```

**Setting up the box from scratch** (only needed once, or after re-printing tags):

```bash
python vision/box_tags.py ident   --source 9   # which tag ids are on it?
python vision/box_tags.py capture --source 9   # 45 s, roll it through the edges
python vision/box_tags.py fit                  # -> outputs/calib/box_body.json
python vision/box_tags.py show                 # eyeball the model in 3-D
sg input -c "python pad_imu.py gyrocal"        # gyro scale + bias (no camera)
sg input -c "python pad_imu.py align --source 9 --seconds 60"   # IMU -> box rotation
sg input -c "python pad_imu.py view --source 9 --reanchor 1"    # check: ~1.4 deg
```

`box_tags.py window` shows the detection in a plain window if you need to aim the camera
(the OpenCV here is the headless build, so there is no `cv2.imshow`; it uses pygame).

The IMU needs the `input` group: prefix with `sg input -c "..."` (or
`sudo usermod -aG input $USER` once, then re-login).

Controls: **A** ready pose · hold **L** engage · **right stick Y** gripper ·
**Home** pause · **B** quit. Logs land in `outputs/teleop/<timestamp>/teleop.csv`.
