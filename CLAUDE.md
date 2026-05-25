# LeRobot Setup — SO-101 Arm

## Environment

- Conda env: `lerobot` (Python 3.10)
- LeRobot version: 0.4.5 (editable install from `lerobot/`)
- Activate with: `conda activate lerobot`
- **Always use `conda activate lerobot` then run commands directly — do NOT use `conda run` for interactive scripts (stdin breaks)**

## Hardware

### SO-101 Arm
- Serial port: `/dev/ttyACM0` (QinHeng CH343 USB-serial chip)
- Motor protocol: Feetech STS3215 (model ID 777), `scservo_sdk` module name
- Baud rate: 1,000,000
- Motors: IDs 1–6, all confirmed responding
  - ID 1: `shoulder_pan`
  - ID 2: `shoulder_lift`
  - ID 3: `elbow_flex`
  - ID 4: `wrist_flex`
  - ID 5: `wrist_roll`
  - ID 6: `gripper`
- Motor setup (ID assignment): **already done** — do not re-run `lerobot-setup-motors`
- Calibration: **done** — `~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101.json`
  - Always pass `--robot.id=so101` to all lerobot commands

### Power Supply
- ⚠️ **5V 4A is WRONG** — triggers STS3215 under-voltage protection (`RxPacketError: Input voltage error`)
- **Required: 7.4–7.5V / 5A+** (SPS-3010 bench supply ordered, set to 7.5V / 5A limit)
- At correct voltage: arm holds position, load reads non-zero, no RxPacketError

### Cameras (not yet configured)
- **External Realsense D455**: `librealsense2` v2.56.5 system-wide; `pyrealsense2==2.56.5.9235` in lerobot env
  - Serial number: `117222251972`
  - Config type: `intelrealsense`
- **Wrist webcam** (`USB CAMERA`): config type `opencv`
  - Video index: `15` (`/dev/video15`), confirmed 640×480 @ 25 fps (not 30 — raises RuntimeError)

### Bluetooth Gamepad
- Nintendo Switch Pro Controller, paired on `/dev/input/event19`
- pygame detects it as `"Pro Controller"`

## Scripts

### `teleop_gamepad.py` — Main teleoperation script
```bash
python teleop_gamepad.py
```
Opens three things simultaneously:
1. **Pygame debug panel** — live joystick axes + buttons + joint position bars
2. **MuJoCo 3-D digital twin** — real-time arm visualization (`SO-ARM100/Simulation/SO101/scene.xml`)
3. **Robot control** — direct joint velocity mode (no IK)

Control mapping (Nintendo Switch Pro Controller):
| Input | Joint |
|---|---|
| Left stick L/R | `shoulder_pan` (base rotation) |
| Left stick U/D | `shoulder_lift` (raise/lower) |
| Right stick U/D | `elbow_flex` (reach) |
| Right stick L/R | `wrist_roll` (spin wrist) |
| L / R buttons | `wrist_flex` (tilt wrist) |
| ZL / ZR | gripper close / open |

Key design decisions:
- `SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1` — joystick works even when pygame window is not focused
- Hold position = re-send current servo readings every frame (no drift, no IK)
- Tune `JOINT_SPEED` dict at top of file to adjust sensitivity per joint

### `diagnose_motors.py` — Live motor diagnostics via Rerun
```bash
python diagnose_motors.py               # read-only, current torque limits unchanged
python diagnose_motors.py --torque 800  # set Max_Torque_Limit to 800 for all motors
```
Opens Rerun viewer automatically. Streams per motor:
- `motors/<name>/position_deg` — calibrated joint angle
- `motors/<name>/load_pct` — torque load (-100% to +100%)
- `motors/<name>/temperature_c` — motor temperature
- `motors/<name>/voltage_v` — supply voltage (should be ~7.4V)
- `motors/<name>/current_raw` — raw current reading

Watch `voltage_v` — if it reads ~5V, the power supply is insufficient.
Watch `load_pct` hitting ±100% — motor is saturating (torque-limited).

## Installed Packages (non-standard)

| Package | Version | Why |
|---|---|---|
| `feetech-servo-sdk` | 1.0.0 | Feetech STS3215 motor protocol (imports as `scservo_sdk`) |
| `pyrealsense2` | 2.56.5.9235 | Python bindings matching system librealsense2 v2.56.5 |
| `opencv-python-headless` | 4.12.0 | LeRobot camera support (`opencv-python 4.8.1.78` removed — conflict) |
| `torchvision` | 0.20.1+cu124 | Compatible with torch 2.5.1 at runtime (despite lerobot requiring >=0.21) |
| `mujoco` | 3.8.1 | Digital twin visualization |
| `placo` | 0.9.23 | IK solver (installed as dependency, available if needed) |

## Key LeRobot Commands

```bash
lerobot-find-port                                    # identify serial port
lerobot-setup-motors --robot.type=so101_follower ... # ALREADY DONE, skip
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101
lerobot-find-cameras                                 # detect camera indices
lerobot-teleoperate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=so101
lerobot-record ...
```

## Status & TODO

- [x] Motor IDs 1–6 confirmed on `/dev/ttyACM0`
- [x] Joint calibration done (`so101.json`)
- [x] All packages installed
- [x] Gamepad teleoperation working (`teleop_gamepad.py`)
- [x] MuJoCo digital twin working
- [x] Rerun motor diagnostics working (`diagnose_motors.py`)
- [ ] **Power supply** — SPS-3010 ordered, set to 7.5V/5A when it arrives
- [x] Realsense D455 serial: `117222251972`
- [x] Wrist webcam index: `15` (640×480 confirmed)
- [x] Full robot config ready (see below)
- [ ] Test full teleoperation at correct voltage

## Robot Config (ready to use)

```python
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

config = SOFollowerRobotConfig(
    port="/dev/ttyACM0",
    id="so101",
    cameras={
        "realsense": RealSenseCameraConfig(
            serial_number_or_name="117222251972",
            fps=30,
            width=640,
            height=480,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=15,
            fps=25,
            width=640,
            height=480,
        ),
    },
)
```
