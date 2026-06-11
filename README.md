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

## Architecture

### Hardware / control stack

The PID loops run **onboard each servo** (the laptop is never inside them) — host scripts
stream goal positions and read telemetry. Gains are not trusted to EEPROM: the patched
lerobot `configure()` re-applies `outputs/tuning/gains.json` on every connect.

```mermaid
flowchart LR
    pad["Switch Pro gamepad"] --> st

    subgraph laptop["Laptop (conda env, Rerun, RTX 3000 Ada)"]
        st["station.py<br/>50 Hz command tick<br/>10 Hz telemetry poll"]
        logs["Rerun window +<br/>outputs/logs/ CSVs<br/>(shared time_s clock)"]
        st --> logs
    end

    subgraph arm["SO-101 arm  (7.4-7.5 V bench supply)"]
        servos["6x STS3215 @ 1 Mbaud<br/>onboard PID per servo<br/>(gains.json: P=32 I=0 D=64<br/>on pan/lift/elbow)"]
        adxl["ADXL345 accelerometer<br/>800 Hz, on wrist_roll"]
    end

    st <-->|"USB ttyACM1 (CH343)<br/>goals down / pos·load·current up"| servos
    adxl -->|"I2C 400 kHz, FIFO"| esp["ESP32-C3"]
    esp -->|"USB ttyACM0<br/>460800 baud binary"| st
    cams["Realsense D455 (top-down)<br/>+ wrist webcam"] -->|USB| st
```

### Software stack

One shared layer per concern — one IMU parser, one gamepad layer, one gains source,
one tuning library — imported by thin entry-point CLIs:

```mermaid
flowchart TD
    subgraph entry["Entry points"]
        station["station.py<br/>teleop cockpit"]
        calibrate["calibrate.py<br/>step / chirp / profile / autotune"]
        simc["sim_collect.py<br/>record sim episodes"]
        he["vision/handeye_calib.py<br/>T_cam-to-base solver"]
        pb["pick_ball.py (WIP)<br/>scripted pick-and-place"]
    end

    subgraph shared["Shared libraries"]
        gp["gamepad_utils.py<br/>stick to deg/s (expo), smoothing,<br/>joint limits, rest-pose shutdown"]
        stl["servo_tuning.py<br/>PID r/w, trajectories, step/FRF<br/>metrics, ImuRecorder, vibration cost"]
        imus["ESP32/imu_serial.py<br/>800 Hz binary parser"]
        vis["vision/ball_yolo.py<br/>YOLO + depth to 3-D point,<br/>fit_table_plane"]
        sb["sim_backend.py<br/>MuJoCo SimBus / SimRobot"]
    end

    subgraph extern["External + config"]
        lr["lerobot 0.4.5 (patched)<br/>SOFollower, cameras"]
        gj["outputs/tuning/gains.json<br/>single source of servo gains,<br/>applied on every connect"]
        mj["MuJoCo SO-101 scene<br/>(SO-ARM100, digital twin / FK)"]
    end

    station --> gp & imus & lr
    calibrate --> stl & gp & lr & sb
    simc --> gp & mj
    he --> gp & lr & mj
    pb --> vis & lr
    stl --> imus
    lr --> gj
    sb --> mj
```

## Setup

```bash
conda activate lerobot                 # Python 3.10, lerobot 0.4.5 editable install
git -C lerobot apply ../patches/lerobot_local.patch   # P-gain + camera fixes (see below)
```

`lerobot/` and `SO-ARM100/` are external repos and aren't tracked here — clone them yourself.
The patch makes `configure()` apply per-motor gains from `outputs/tuning/gains.json` on every
connect (default P=32 so the arm holds against gravity) and makes camera connect/read
fault-tolerant.

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
- [x] **Ball detection via YOLO** (`vision/ball_yolo.py`): pretrained basketball
      `best.pt` (`vision/models/basketball.pt`) → 2-D box → back-project box centre
      through depth → 3-D point (cam frame); keeps `fit_table_plane`. Clean single
      box @0.36 conf where color was brittle (wood reads orange). Zero-shot COCO
      (`orange`@0.09) and YOLO-World (@0.18) were too weak. Fine-tune on our scene
      later if conf wobbles at table edges / under occlusion (model = ready auto-labeler).
- [x] **Hand-eye calibration done** (`vision/handeye_calib.py`): **4.6 mm RMS over
      16 poses** → `outputs/calib/handeye.json`. Marker = the **pink heart sticker** on a
      gripper finger, detected with **Grounding DINO** ("heart" zero-shot) + an in-box
      pink gate (hue wraps: gate both HSV ends, cap saturation to reject the red arm).
      Per pose: pink pixel + depth → 3-D (cam frame), joint angles + MuJoCo FK →
      3-D (base frame); least-squares jointly fits T_cam→base **and** the marker offset.
      How/why: [vision/HANDEYE.md](vision/HANDEYE.md).
- [~] **Scripted pick (WIP)** — `pick_ball.py`: GDINO ball detection + handeye +
      multi-seed MuJoCo IK + twin preview + staged grasp behind a confirm. First real
      grasp still pending (z-from-depth fix is in, untested on hardware).
- [~] **Fast scene detector** — GDINO-as-teacher auto-labeling (`vision/autolabel.py`)
      → yolov8n student + benchmark (`vision/detector_bench.py`). v1 trained on 45
      auto-labeled frames; see [vision/DETECTOR.md](vision/DETECTOR.md).
- [ ] Scripted pick→rotate→place state machine, with randomization.
- [ ] Auto-record LeRobot dataset in a loop → train **ACT**, then **SmolVLA**.
- [ ] Fix two-camera USB stall (wrist cam for grasp) or collect top-down-only first.

**Done 2026-06-11 — servo tuning + teleop (see CLAUDE.md for details)**
- [x] **IMU integrated into `calibrate.py`** — every hardware capture records the wrist
      IMU; `step`/`profile` report after-stop decay/residual/ring-frequency, `autotune`
      penalizes measured wrist ringing in its cost. Verified on hardware.
- [x] **Tuned the big joints** — after-stop oscillation was an underdamped loop
      (factory D=32): now **P=32 I=0 D=64** on pan/lift/elbow via `gains.json`
      (applied at every connect). I>0 limit-cycles against stiction — keep I=0.
      Irreducible ~0.5 m/s² holding buzz at gravity poses = torque dither, not PID.
- [x] **Teleop overhaul** — 50 Hz command tick decoupled from telemetry polling,
      `JOINT_SPEED` in deg/s (~2.5× faster), expo stick curve.

**Next (at the bench, arm at 7.5 V)**
- [ ] Commanded **chirp on elbow_flex** (`calibrate.py` + `station.py` logging) → confirm
      the **~9.6 Hz** structural mode found in teleop (hand-wiggle, not clean yet).
- [ ] **Wrist pose-sweep** (wave wrist_roll + wrist_flex through full range, rest still) →
      pin the IMU mounting rotation below ~1° (currently 2.9°, see `analyze_run.py`).
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
