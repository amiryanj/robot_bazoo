# robot_bazoo — SO-101 arm + LeRobot

Personal control/diagnostics/tuning stack for an SO-101 follower arm on LeRobot 0.4.5.
Repo: `github.com/amiryanj/robot_bazoo`. See [README.md](README.md) for the short tour.

## Primary goal — end-to-end VLA (don't get distracted)

The **main direction** is an autonomous, end-to-end AI bot: run a vision-language-action
policy on this laptop and fine-tune it on task data. Everything else (servo tuning, IMU
resonance, etc.) is secondary — useful but **off the critical path**; don't rabbit-hole.

**Reference task:** detect a mini-basketball (sphere) with the top-down Realsense, pick it,
rotate a bit, place it, repeat.

**Data strategy:** no leader arm and joystick teleop is too clumsy for pick-and-place, so
**collect demos with a scripted/deterministic policy** (perception → IK "moveTo" → grasp
state machine), then fine-tune the VLA on those episodes. Build in **randomization**
(ball position, start pose, trajectory noise) from the start — clean scripted data alone
gives a policy that can't recover from mistakes (covariate shift).

**Critical path:** (1) ball detection + 3D point from depth → (2) **hand-eye calibration**
(camera→base — the one unavoidable prerequisite) → (3) scripted pick-place state machine
(IK via `placo`, already installed) → (4) auto-record LeRobot dataset in a loop →
(5) train **ACT first** (tiny, trains locally for fast iteration), then **SmolVLA** (~450M,
laptop-class, adds language). Big VLAs (OpenVLA-7B, Pi0) likely need cloud training.

**Inference vs training:** we *can* run bigger VLAs locally for **inference** — OpenVLA-7B
in 4-bit (~6–7 GB) and Pi0-class fit the 8 GB GPU. The limit is **speed, not memory**:
a 7B runs ~1–5 Hz closed-loop on this laptop (fine for slow pick-place, bad for reactive
control). ACT/SmolVLA run real-time. So big models are a comparison point; small-and-fast
is the practical target for the bot.

**Perception (decided):** ball detection uses a **deep detector (YOLO)**, not classical
CV. Tried color+depth first (`vision/ball.py`) — the **RANSAC table-plane fit works well**
and stays as a reusable primitive, but **color-only ball detection is brittle**: the wooden
table reads orange in HSV and swamps the mask. A 3D-extent filter helps, but a small
trained/zero-shot YOLO is the robust path. Plan: YOLO 2-D box → back-project box centre
through depth → 3-D ball point; keep `fit_table_plane` for the workspace reference.

**Hardware gotchas for this goal:** GPU is an **RTX 3000 Ada Laptop (~8 GB VRAM)** — fits
ACT/SmolVLA, not big-VLA training. The **two-camera USB stall** is now on the critical path
(VLAs want the wrist cam during grasp); solve it (powered hub / separate controller) or
collect top-down-only first.

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

**SO-101 arm** — `/dev/ttyACM1` (CH343 USB-serial, vendor id `1a86`), Feetech STS3215
(model 777), baud 1,000,000. Motors / IDs: 1 `shoulder_pan`, 2 `shoulder_lift`,
3 `elbow_flex`, 4 `wrist_flex`, 5 `wrist_roll`, 6 `gripper`. Calibration:
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101.json` (id `so101`).
Positions are in **degrees** (`use_degrees=True`), gripper in 0–100.
Note: the arm moved to `ttyACM1` once the ESP32-C3 IMU claimed `ttyACM0` — `station.py`
defaults the arm to `ttyACM1` (override with `--port`); the IMU reader finds the C3 by
USB vendor id `303a`, so enumeration order doesn't matter.

**Wrist IMU** — ADXL345 accelerometer on an ESP32-C3 Super Mini, mounted on `wrist_roll`.
`/dev/ttyACM0` (native USB, vendor id `303a`), 800 Hz, binary stream at 460800 baud,
scale `0.038246` m/s²/LSB. Firmware + readers in `ESP32/` (see `ESP32/CLAUDE.md`).
`station.py` reads it in a background thread; `imu_serial.py` is the shared parser.

**Power** — 7.4–7.5 V / 5A+ (SPS-3010 bench supply). Connected and working; arm holds
and lifts at 7.4 V.

**Cameras** — Realsense D455 serial `117222251972` (`intelrealsense`), **mounted top-down
looking at the workspace** (the scene cam for ball detection; depth gives metric 3D);
wrist webcam index `15` (`opencv`, 640×480 @ **25** fps — 30 raises RuntimeError). Known
issue: both on the same USB hub stall (bandwidth); each works alone. Scripts connect
cameras fault-tolerantly and accept `--no-realsense` / `--no-wrist`.

**Gamepad** — Nintendo Switch Pro Controller (pygame name `"Pro Controller"`). Profiles
auto-detected in `gamepad_utils.py`.

**Compute** — laptop with an **NVIDIA RTX 3000 Ada Laptop GPU (~8 GB VRAM)**. Fits ACT /
SmolVLA training+inference; big-VLA (7B) training needs the cloud.

## Scripts

**Control & diagnostics (real arm):**
- `station.py` — **the single entry point** (replaced `teleop_gamepad.py`,
  `command_log.py`, `diagnose_motors.py`). Connects the arm, auto-opens **one** Rerun
  window (`spawn=True`) with a default blueprint, and streams everything onto one
  `time` timeline: motors (pos/load/current/voltage/status), the gamepad (sticks as
  2-D points + button strips), the wrist IMU (background thread, see ESP32), and
  optionally cameras + a MuJoCo twin. Gamepad mapping unchanged: L-stick = pan/lift,
  R-stick = elbow/wrist-roll, L/R = wrist-flex, ZL/ZR = gripper. Also takes typed
  commands (`<joint> <deg>`, `all <deg>`, `hold`, `torque off/on`, `q`).
  - Joystick is **hot-pluggable** — start with none, plug in mid-run, unplug and the
    arm holds (pygame runs headless via `SDL_VIDEODRIVER=dummy`, input only).
  - Flags: `--observe` (read-only, no teleop), `--health` (startup register report),
    `--torque N` / `--fix-pgain` / `--clear-overload` (EEPROM fixes), `--cameras`,
    `--twin`, `--no-imu`, `--no-log`, `--port`, `--rate`.
  - Logs to `outputs/logs/<timestamp>/`: `station.csv` (motors @ `--rate`),
    `imu.csv` (IMU @ 800 Hz), `summary.txt` (header + `t0_unix_epoch` + health report).
    Both CSVs share one `time_s` origin (`t0`) so motor↔IMU rows merge directly. The
    IMU goes to Rerun + `imu.csv`, **not** into `station.csv`.
- `gamepad_utils.py` — shared constants, controller profiles, `DeltaSmoother`,
  `graceful_shutdown`, `REST_POSE`, pygame draw helpers. Imported by `station.py`
  and `sim_collect.py`.

**Sim & data:**
- `sim_collect.py` — gamepad teleop inside MuJoCo, records episodes as a LeRobot
  dataset (for ACT training without the real arm). Scene: `scene_sim.xml` (ball + cams).

**Vision (perception, WIP):**
- `vision/capture_frame.py` — grab one aligned color+depth+intrinsics frame from the
  top-down Realsense → `outputs/vision/<ts>/`. For offline detector dev (no arm).
- `vision/ball.py` — `fit_table_plane` (RANSAC, **works**) + `localize_ball` (color+depth,
  **brittle** — wood reads orange). Plane fit stays; ball detection moving to YOLO.

**Analysis:**
- `analyze_run.py` — offline analysis of a `station.py` run: gravity-removed wrist
  vibration (spectrum + correlation with joint motion) and IMU-frame localization via
  MuJoCo FK + Kabsch. `python analyze_run.py <run_dir>`.

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
`--p 32 --i 0 --d 32`. The wrist IMU (now mounted) gives true resonance ID / input
shaping beyond the encoder-only chirp ceiling (~20–30 Hz); capture motor+IMU together
with `station.py` and merge the two CSVs on `time_s`. `calibrate.py` does not yet read
the IMU.

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
- [x] Wrist IMU (ADXL345 on ESP32-C3) mounted on `wrist_roll`, streaming at 800 Hz
- [x] Unified cockpit `station.py` — motors + IMU + gamepad in one Rerun window,
      synced CSV logging (`station.csv` + `imu.csv` on a shared clock)
- [ ] **Run tuning on the real arm** (not done yet — start shoulder_pan at 7.5 V)
- [ ] Use synced IMU + motor logs to calibrate servo coefficients / ID resonance
- [ ] Revisit `graceful_shutdown` — Ctrl-C rest-pose move was abrupt/noisy (tune
      `REST_POSE` / duration, maybe slower easing)
- [ ] Realsense + wrist cam USB bandwidth (works alone, stalls together)
- [ ] Collect sim episodes → cloud ACT training (plan exists)

## Robot config (cameras)

```python
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

config = SOFollowerRobotConfig(
    port="/dev/ttyACM1", id="so101",   # ttyACM0 is the ESP32-C3 IMU
    cameras={
        "realsense": RealSenseCameraConfig(serial_number_or_name="117222251972", fps=30, width=640, height=480),
        "wrist":     OpenCVCameraConfig(index_or_path=15, fps=25, width=640, height=480),
    },
)
```
