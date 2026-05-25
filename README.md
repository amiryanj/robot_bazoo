# robot_bazoo

My working setup for an [SO-101](https://github.com/TheRobotStudio/SO-ARM100) follower arm,
built on top of [LeRobot](https://github.com/huggingface/lerobot). Gamepad teleop, a MuJoCo
digital twin, motor diagnostics, sim data collection, and a controller-tuning toolkit for the
Feetech STS3215 servos.

## Hardware

- SO-101 follower, 6× Feetech STS3215 on `/dev/ttyACM0` @ 1 Mbaud
- Needs 7.4–7.5 V supply (5 V trips under-voltage protection)
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
| `teleop_gamepad.py` | Gamepad teleop; `--twin` adds the MuJoCo viewer |
| `gamepad_utils.py` | Shared controller profiles, smoothing, graceful shutdown |
| `diagnose_motors.py` | Live per-motor state → Rerun + CSV, with a startup health report |
| `command_log.py` | Drive joints (pad or text) while logging load/current |
| `sim_collect.py` | Collect episodes in MuJoCo as a LeRobot dataset |
| `servo_tuning.py` | Tuning toolbox: PID r/w, trajectories, step/FRF analysis |
| `calibrate.py` | `step` / `chirp` / `profile` / `autotune` CLI for the servos |

```bash
python calibrate.py autotune --joint shoulder_pan --size 25 --trials 40
```

Detailed hardware notes, register quirks, and status live in [CLAUDE.md](CLAUDE.md).
