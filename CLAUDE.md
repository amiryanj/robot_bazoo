# robot_bazoo — SO-101 arm + LeRobot

Personal control/diagnostics/tuning stack for an SO-101 follower arm on LeRobot 0.4.5.
Repo: `github.com/amiryanj/robot_bazoo`. See [README.md](README.md) for the short tour.

## Working principles

How to approach any change here — from
[andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills):

1. **Think before coding** — don't assume, don't hide confusion, surface tradeoffs.
   State assumptions explicitly; present alternatives instead of silently picking one;
   mention the simpler approach and push back when warranted; stop and name confusion
   rather than ploughing ahead.
2. **Simplicity first** — minimum code that solves the problem, nothing speculative.
   No unrequested features, abstractions, flexibility, or error handling for impossible
   cases. Test: *"would a senior engineer call this overcomplicated?"* If yes, simplify.
3. **Surgical changes** — touch only what you must; clean up only your own mess. Don't
   refactor, reformat, or improve adjacent working code; don't delete pre-existing dead
   code; match existing style. Every changed line should trace to the request.
4. **Goal-driven execution** — define success criteria, loop until verified. Turn tasks
   into measurable checks; for multi-step work, state a brief plan with a verification
   step for each. (Here, "verified" often means: ran it in `--sim`, metrics improved.)

## Conventions (read first)

These are the rules that keep a fresh session from breaking things:

- **Activate, don't `conda run`.** `conda activate lerobot` then run scripts directly.
  `conda run` breaks stdin for interactive scripts.
- **Always pass `--robot.id=so101`** to lerobot CLI commands (loads the calibration).
- **Don't re-run `lerobot-setup-motors`** (motor IDs 1–6 are assigned) and don't
  recalibrate unless the calibration file is missing — both are already done.
- **The `P_Coefficient=32` fix is load-bearing — do not revert it.** lerobot's
  `configure()` originally halved it to 16, which left the arm unable to lift against
  gravity. The fix lives in `patches/lerobot_local.patch` (see below).
- **Power: 7.4–7.5 V required.** 5 V trips STS3215 under-voltage protection
  (`RxPacketError: Input voltage error`). Tune/test only at correct voltage.
- **`lerobot/` and `SO-ARM100/` are external git repos** (gitignored here). After a
  fresh clone, reapply local fixes: `git -C lerobot apply ../patches/lerobot_local.patch`.
- **On shutdown the arm collapses unless eased down.** Scripts call
  `graceful_shutdown()` (interpolates to `REST_POSE` in `gamepad_utils.py`) before
  torque-off; `shoulder_lift` is the one that drops hardest.
- **EEPROM wear:** P/I/D are EEPROM registers; `calibrate.py autotune` writes them
  once per trial. A few hundred trials is fine; don't loop for hours.

## Environment

- Conda env `lerobot` (Python 3.10), LeRobot 0.4.5 editable install from `lerobot/`.
- Extra deps installed in-env: `control`, `optuna`, `scipy` (tuning); `mujoco==3.8.1`,
  `placo==0.9.23`; `feetech-servo-sdk==1.0.0` (imports as `scservo_sdk`);
  `pyrealsense2==2.56.5.9235`; `opencv-python-headless==4.12.0`;
  `torchvision==0.20.1+cu124` (runtime-compatible with torch 2.5.1).

## Hardware

**SO-101 arm** — `/dev/ttyACM0` (CH343 USB-serial), Feetech STS3215 (model 777),
baud 1,000,000. Motors / IDs: 1 `shoulder_pan`, 2 `shoulder_lift`, 3 `elbow_flex`,
4 `wrist_flex`, 5 `wrist_roll`, 6 `gripper`. Calibration:
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101.json` (id `so101`).
Positions are in **degrees** (`use_degrees=True`), gripper in 0–100.

**Power** — 7.4–7.5 V / 5A+ (SPS-3010 bench supply). Connected and working; arm holds
and lifts at 7.4 V.

**Cameras** — Realsense D455 serial `117222251972` (`intelrealsense`); wrist webcam
index `15` (`opencv`, 640×480 @ **25** fps — 30 raises RuntimeError). Known issue: both
on the same USB hub stall (bandwidth); each works alone. Scripts connect cameras
fault-tolerantly and accept `--no-realsense` / `--no-wrist`.

**Gamepad** — Nintendo Switch Pro Controller (pygame name `"Pro Controller"`). Profiles
auto-detected in `gamepad_utils.py`.

## Scripts

**Control & diagnostics (real arm):**
- `teleop_gamepad.py` — gamepad joint-velocity teleop + pygame panel + Rerun.
  `--twin` adds the MuJoCo 3-D viewer (off by default). Mapping: L-stick = pan/lift,
  R-stick = elbow/wrist-roll, L/R = wrist-flex, ZL/ZR = gripper.
- `command_log.py` — drive joints by gamepad **or** typed commands while logging
  load/current per motor (pygame bars + Rerun + CSV). For studying torque draw.
- `diagnose_motors.py` — startup register health report + live sensor stream to
  Rerun + CSV. Flags: `--torque N`, `--fix-pgain`, `--clear-overload`.
- `gamepad_utils.py` — shared constants, controller profiles, `DeltaSmoother`,
  `graceful_shutdown`, `REST_POSE`, pygame draw helpers. Imported by the above.

**Sim & data:**
- `sim_collect.py` — gamepad teleop inside MuJoCo, records episodes as a LeRobot
  dataset (for ACT training without the real arm). Scene: `scene_sim.xml` (ball + cams).

**Controller tuning (Feetech STS3215):**
- `servo_tuning.py` — library: Lock-aware PID read/write (keeps torque on),
  `Acceleration` profiling, high-rate single-joint capture, trajectory generators
  (step / trapezoid / sine / chirp), step metrics, 2nd-order fit, FRF/resonance, cost.
- `calibrate.py` — CLI: `step` / `chirp` / `profile` / `autotune` (Optuna closed-loop
  PID search). `--sim` runs the whole loop on the MuJoCo twin; `--rate` caps samples.
- `sim_backend.py` — MuJoCo `SimBus`/`SimRobot` mimicking the Feetech bus, so
  `calibrate.py --sim` can smoke-test tuning with no hardware. Maps P/D → actuator
  kp/kv, I → integral torque, Acceleration → setpoint rate-limit.

## Tuning workflow

Method: clean step response → measure rise / overshoot / settling / steady-state error
→ tune **D (damping) → P (speed) → I (steady-state)**. Chirp gives the frequency
response (resonance). Acceleration profiling (`profile`) attacks inertia/backlash
separately from the gains. shoulder_pan carries the most inertia (start there).

```bash
conda activate lerobot
# validate the loop with no hardware/noise:
python calibrate.py --sim autotune --joint shoulder_pan --size 25 --trials 40
# on the real arm (at 7.5 V):
python calibrate.py step     --joint shoulder_pan --size 25
python calibrate.py autotune --joint shoulder_pan --size 25 --trials 40
python calibrate.py chirp    --joint shoulder_pan --f0 0.5 --f1 15
```
Outputs (CSV + PNG) land in `outputs/tuning/<timestamp>_*/`. Restore factory gains with
`--p 32 --i 0 --d 32`. Future: a wrist accelerometer (IMU) for true resonance ID /
input shaping — encoder-only feedback caps the chirp at ~20–30 Hz.

## Local lerobot patches

`patches/lerobot_local.patch` holds two edits to the gitignored `lerobot/` install:
1. `so_follower.py`: `P_Coefficient=32` (lift torque) + per-camera fault-tolerant connect.
2. `camera_realsense.py`: read-loop `stop_event` fix.

Reapply after a fresh lerobot clone: `git -C lerobot apply ../patches/lerobot_local.patch`.

## Status & TODO

- [x] Motors confirmed, calibrated; arm lifts at 7.4 V (P_Coefficient=32)
- [x] Gamepad teleop, MuJoCo twin, Rerun diagnostics, current logging
- [x] Tuning toolkit built; analysis validated on synthetic data; full loop smoke-tested in `--sim`
- [x] Project under git, pushed to GitHub
- [ ] **Run tuning on the real arm** (not done yet — start shoulder_pan at 7.5 V)
- [ ] Realsense + wrist cam USB bandwidth (works alone, stalls together)
- [ ] Collect sim episodes → cloud ACT training (plan exists)
- [ ] Buy + mount wrist IMU for resonance/input-shaping

## Robot config (cameras)

```python
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

config = SOFollowerRobotConfig(
    port="/dev/ttyACM0", id="so101",
    cameras={
        "realsense": RealSenseCameraConfig(serial_number_or_name="117222251972", fps=30, width=640, height=480),
        "wrist":     OpenCVCameraConfig(index_or_path=15, fps=25, width=640, height=480),
    },
)
```
