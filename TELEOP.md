# Tag teleop: driving the SO-101 with a tagged joystick

**Goal.** Move the arm by moving your hand. A plain USB webcam watches AprilTags stuck on
a hand-held Nintendo Switch Pro Controller; the controller's 6-D pose drives the TCP
position + wrist roll, and the gamepad's own buttons carry the clutch and the gripper.

This exists because there is no leader arm and stick teleop is too clumsy for
pick-and-place (see CLAUDE.md, "Data strategy"). It is a **demo-collection input device**,
not the end goal.

## Layers (each file has one job, and imports the one below)

```
teleop_tag.py        the loop: clutch, mapping, safety rails, IK, Rerun, CSV
  ├─ pad_imu.py      the controller's OWN 6-axis IMU  (evdev) + its extrinsic calibration
  ├─ vision/tag_body.py   rigid multi-tag body model -> one body pose from any visible tag
  │    └─ vision/tag_pose.py   camera + ArUco detector + single-tag PnP  (the base layer)
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

`pad_imu.py` calibrates the one fixed rotation `X` (`v_body = X @ v_imu`):
- `align` — from a hand-waving recording, pairing **rotation increments** over 0.35 s
  (not finite-differenced angular velocity, which amplifies 20 Hz pose noise ~25×).
- `check` — gravity-based, `R_k @ X @ a_k = g_cam`, alternating solve + Huber IRLS,
  with bootstrap uncertainty and a **coverage** score (gravity in one orientation pins
  only 2 of 3 DoF, so coverage must be reported, not assumed).

State 2026-08-21: **X uncertain to 2.19°** from 35 accumulated poses; per-pose scatter
4.06° is irreducible AprilTag noise. Successive solves disagree by 1.7° — below the noise
floor, i.e. converged. `check` appends across sessions, so more poses tighten it.

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

**5. The drift you feel is a ratchet, and it is caused by tag blindness — not by the code.**
Measured over one 95 s session:

| hand motion | net displacement | path |
|---|---|---|
| while **ENGAGED** | [+203, −120, **+277**] mm | 1573 mm |
| while **IDLE** | [+94, +112, **−280**] mm | 1594 mm |

The outward and return strokes cancel almost exactly, but **half the hand motion (1594 of
3167 mm) happened while disengaged and was discarded**. The disengages were not deliberate:
28 blind gaps, mean 0.80 s, max 5.5 s, **68% longer than the 0.4 s timeout**. The operator
never sees it happen mid-stroke.

Root cause: **`tags == 2` has occurred in 0% of every run logged** — the second tag has
never once been seen, so exactly one hand orientation is trackable. This is a tag-placement
problem, not a timing one. Immediate remedy: press **A** to ease back to the ready pose.

## Known issues / next

- [ ] **Tag coverage** — tags on more faces of the controller. This is the top blocker.
- [ ] `ms_detect` 28.8 ms — detect at half resolution, refine at full.
- [ ] Teleop-appropriate `approach_dir` so `wrist_flex` stops saturating.
- [ ] **Tag id collision:** the desk anchor tag (`vision/cam_calib.py`, `DICT_4X4_50` id 13)
      and the joystick reference tag (`outputs/calib/joystick_body.json`, `DICT_4X4_50`
      id 13, same 27.4 mm) are **the same dictionary, id and size**. Harmless in normal use
      (different cameras), but `cam_calib.py recal` will mis-anchor if the controller is
      lying in the Realsense's view. Renumber the joystick tags when they are reprinted.
- [ ] Rerun retains the whole recording by design (~700 B per scalar point × ~36 series).
      Not a leak in our code or MuJoCo — verified RSS-flat with logging off.

## Running it

```bash
conda activate lerobot
# sim (no hardware):
python teleop_tag.py --sim --gain 1.0 --max-speed 0.20
# real arm:
python teleop_tag.py --gain 1.0 --max-speed 0.20
```

The IMU needs the `input` group: prefix with `sg input -c "..."` (or
`sudo usermod -aG input $USER` once, then re-login).

Controls: **A** ready pose · hold **L** engage · **right stick Y** gripper ·
**Home** pause · **B** quit. Logs land in `outputs/teleop/<timestamp>/teleop.csv`.
