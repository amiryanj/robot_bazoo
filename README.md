# robot_bazoo

My working setup for an [SO-101](https://github.com/TheRobotStudio/SO-ARM100) follower arm,
built on top of [LeRobot](https://github.com/huggingface/lerobot). A single "station" cockpit
(gamepad teleop + motor/IMU/camera telemetry in one Rerun window), a MuJoCo digital twin, sim
data collection, and a controller-tuning toolkit for the Feetech STS3215 servos.

## Hardware

- SO-101 follower, 6× Feetech STS3215 @ 1 Mbaud — `/dev/ttyACM1` (the ESP32-C3 IMU
  grabs `/dev/ttyACM0`; scripts pick each by USB vendor id, so order doesn't matter)
- Needs 7.4–7.5 V supply (5 V trips under-voltage protection)
- Wrist IMU: ADXL345 on an ESP32-C3, mounted on `wrist_roll`, 800 Hz (see [ESP32/](ESP32/))
- Optional: Realsense D455 + wrist webcam, a Switch Pro controller

## Setup

```bash
conda activate lerobot                 # Python 3.10, lerobot 0.4.5 editable install
git -C lerobot apply ../patches/lerobot_local.patch   # P-gain + camera fixes (see below)
```

`lerobot/` and `SO-ARM100/` are external repos and aren't tracked here — clone them yourself.
The patch restores `P_Coefficient=32` (so the arm holds against gravity) and makes camera
connect/read fault-tolerant.

## Scripts

| Script | What it does |
|---|---|
| `station.py` | **Main entry point.** Gamepad teleop + typed commands while motors, the wrist IMU, the controller, and (opt) cameras stream to one Rerun window + CSV. Flags: `--observe` (read-only), `--health`, `--cameras`, `--twin`, `--no-imu`, `--no-log`. |
| `gamepad_utils.py` | Shared controller profiles, smoothing, graceful shutdown |
| `sim_collect.py` | Collect episodes in MuJoCo as a LeRobot dataset |
| `servo_tuning.py` | Tuning toolbox: PID r/w, trajectories, step/FRF analysis |
| `calibrate.py` | `step` / `chirp` / `profile` / `autotune` CLI for the servos |
| `ESP32/` | ESP32-C3 + ADXL345 IMU firmware and host-side readers |

```bash
python station.py                  # full cockpit: teleop + motors + IMU → Rerun + CSV
python station.py --observe        # read-only telemetry (safe, no motion)
python calibrate.py autotune --joint shoulder_pan --size 25 --trials 40
```

Each `station.py` run writes `outputs/logs/<timestamp>/` with `station.csv` (motors),
`imu.csv` (IMU @ 800 Hz), and `summary.txt`. Both CSVs share one `time_s` clock, so they
merge directly for motor↔IMU analysis.

Detailed hardware notes, register quirks, and status live in [CLAUDE.md](CLAUDE.md).

## TODO

Live worklist; longer status/history lives in [CLAUDE.md](CLAUDE.md).

**Primary: end-to-end VLA** (main direction — see CLAUDE.md)
- [x] Realsense capture tool (`vision/capture_frame.py`) + RANSAC table-plane fit.
- [ ] **Ball detection via YOLO** (color+depth was brittle — wood reads orange);
      2-D box → back-project through depth → 3-D point.
- [ ] Hand-eye calibration (camera→base) — the one unavoidable prerequisite.
- [ ] Scripted pick→rotate→place state machine (IK via `placo`), with randomization.
- [ ] Auto-record LeRobot dataset in a loop → train **ACT**, then **SmolVLA**.
- [ ] Fix two-camera USB stall (wrist cam for grasp) or collect top-down-only first.

**Next (at the bench, arm at 7.5 V)**
- [ ] Commanded **chirp on elbow_flex** (`calibrate.py` + `station.py` logging) → confirm
      the **~9.6 Hz** structural mode found in teleop (hand-wiggle, not clean yet).
- [ ] **Wrist pose-sweep** (wave wrist_roll + wrist_flex through full range, rest still) →
      pin the IMU mounting rotation below ~1° (currently 2.9°, see `analyze_run.py`).
- [ ] **Tune D per joint** against the mode — start big: shoulder_pan, shoulder_lift,
      elbow_flex; wrist/gripper are easy.

**Calibration infra**
- [ ] Integrate IMU into `calibrate.py autotune` — add a wrist-vibration cost term
      (per-trial IMU thread + 9.6 Hz energy). Scaffold ready to write; **verify on hardware**.
- [ ] Use IMU→link rotation to subtract gravity + rigid-body accel → pure joint oscillation.

**Fixes / cleanup**
- [ ] `graceful_shutdown` Ctrl-C move was abrupt/noisy — slower easing / tune REST_POSE.
- [ ] Handle the ±100% `load_pct` single-sample spikes (direction-reversal glitch) when
      parsing motor loads for calibration.
- [ ] Realsense + wrist cam USB bandwidth (works alone, stalls together).

**Findings (2026-06-10, run 00-48-25)**
- Wrist vibration ~9.6 Hz (mode) + 19.3 Hz (harmonic); RMS 0.5 m/s², peak 8.
- Vibration tracks joint **speed** (r=0.60), not current — mostly **elbow** (0.59), then pan/lift.
- IMU mounting solved: IMU +X≈link+Z, +Y≈link+X, +Z≈link+Y (residual 2.9°).
