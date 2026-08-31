"""Where the hardware is, and where output goes. One place, so it is edited once.

`/dev/ttyACM1` was written into 14 files, the Realsense serial into 3, the tag-box webcam
index into 5. Every one of those is a fact about THIS bench, not about the code, and when
a device moves they all have to agree.

Deliberately flat constants, not a config framework: the values are a handful of strings
and paths, they change rarely, and every script already takes a CLI flag to override.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── devices ───────────────────────────────────────────────────────────────────────────
ARM_PORT = "/dev/ttyACM1"      # CH343 arm controller. The ESP32-C3 wrist IMU owns
                               # ttyACM0, and both are found by USB vendor id, so
                               # enumeration order does not matter -- this is the default,
                               # not a guarantee.
ROBOT_ID = "so101"             # selects the lerobot calibration file; always pass it
RS_SERIAL = "117222251972"     # Realsense D455, mounted top-down over the workspace
HAND_CAM = "9"                 # /dev/video9, the USB webcam watching the tag box
WRIST_CAM = 15                 # wrist webcam (640x480 @ 25 fps; 30 raises RuntimeError)

# ── outputs ───────────────────────────────────────────────────────────────────────────
OUT = ROOT / "outputs"
CALIB = OUT / "calib"          # handeye.json, box_body.json, pad_imu.json, ...
LOGS = OUT / "logs"            # station.py runs
TUNING = OUT / "tuning"        # calibrate.py runs
TELEOP = OUT / "teleop"        # teleop_tag.py runs

# ── external repos (gitignored, cloned beside this one) ───────────────────────────────
SO_ARM = ROOT / "SO-ARM100"
SCENE_XML = str(SO_ARM / "Simulation/SO101/scene.xml")        # the twin / FK model
SCENE_SIM_XML = str(SO_ARM / "Simulation/SO101/scene_sim.xml")  # + ball and cameras
VISION_OUT = OUT / "vision"        # capture_frame.py dumps, detector datasets
DATASETS = ROOT / "datasets"       # LeRobot datasets recorded here

# ── calibration files everything reads ────────────────────────────────────────────────
GAINS = ROOT / "config/gains.json"        # servo P/I/D, re-applied on every connect
BOX_BODY = CALIB / "box_body.json"        # the 3-face tag box (vision/box_tags.py fit)
PAD_IMU = CALIB / "pad_imu.json"          # X, gyro bias + scale (pad_imu.py)
HANDEYE = CALIB / "handeye.json"          # T_cam->base (vision/cam_calib.py)
