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
