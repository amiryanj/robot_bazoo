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

**Critical path:** (1) ball detection + 3D point from depth (**done** — `vision/ball_yolo.py`)
→ (2) **hand-eye calibration** (**done** — `vision/cam_calib.py`, reference-anchored
T_cam→base from the white-plate plane + desk tag → `outputs/calib/handeye.json`; recover
after any camera move with `cam_calib.py recal`, see below) → (3) scripted pick-place state machine
(IK via `placo`, already installed) → (4) auto-record LeRobot dataset in a loop →
(5) train **ACT first** (tiny, trains locally for fast iteration), then **SmolVLA** (~450M,
laptop-class, adds language). Big VLAs (OpenVLA-7B, Pi0) likely need cloud training.

**Inference vs training:** we *can* run bigger VLAs locally for **inference** — OpenVLA-7B
in 4-bit (~6–7 GB) and Pi0-class fit the 8 GB GPU. The limit is **speed, not memory**:
a 7B runs ~1–5 Hz closed-loop on this laptop (fine for slow pick-place, bad for reactive
control). ACT/SmolVLA run real-time. So big models are a comparison point; small-and-fast
is the practical target for the bot.

**Perception (done):** ball detection uses a **deep detector (YOLO)**, not classical
CV. Tried color+depth first (`vision/ball.py`) — the **RANSAC table-plane fit works well**
and stays as a reusable primitive, but **color-only ball detection is brittle**: the wooden
table reads orange in HSV and swamps the mask. Implemented in `vision/ball_yolo.py`: a
**pretrained basketball YOLO** (`vision/models/basketball.pt`, ~6 MB, from
[avishah3/AI-Basketball-Shot-Detection-Tracker], classes `Basketball`/`Hoop`) → 2-D box →
back-project box centre through depth → 3-D ball point (cam frame); `fit_table_plane` pushes
the surface point in by one radius to the centre. Despite the broadcast→tabletop domain gap
it gives a **clean single box @0.36** (a basketball is a sphere — same from any angle), vs
zero-shot COCO `orange`@0.09 and YOLO-World @0.18, which were unusable. **Next-step plan if
conf wobbles** (table edges, arm occlusion): fine-tune `yolov8n` on our own top-down scene,
using this model as a free auto-labeler (+ a Roboflow ball dataset to augment).

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
- **The servo-gain fix in `configure()` is load-bearing — do not revert it.** The
  patched `so_follower.py` applies per-motor gains from `config/gains.json`
  on every connect (currently P=32 I=0 **D=200** on shoulder_pan/lift/elbow — tuned
  2026-06-11 against wrist-IMU ringing; default P=32 I=0 D=32 elsewhere). Stock
  lerobot wrote P=16, which left the arm unable to lift against gravity. Lives in
  `patches/lerobot_local.patch` (see below). Note this also means **EEPROM gains are
  reset from gains.json on every connect** — to keep a tuning result, put it in that
  file.
- **Power: 7.4–7.5 V required.** 5 V trips STS3215 under-voltage protection
  (`RxPacketError: Input voltage error`). Tune/test only at correct voltage.
- **`lerobot/` and `SO-ARM100/` are external git repos** (gitignored here). After a
  fresh clone, reapply local fixes: `git -C lerobot apply ../patches/lerobot_local.patch`.
- **On shutdown the arm collapses unless eased down.** Scripts call
  `graceful_shutdown()` (interpolates to `REST_POSE` in `gamepad_utils.py`) before
  torque-off; `shoulder_lift` is the one that drops hardest.
- **EEPROM wear:** P/I/D are EEPROM registers; `calibrate.py autotune` writes them
  once per trial. A few hundred trials is fine; don't loop for hours.
- **One process per serial port.** Two scripts on `/dev/ttyACM1` (e.g. `station.py`
  twice, or a background test + a user run) interleave packets and both fail with
  `TxRxResult: Incorrect status packet!` / `no status packet`. Seeing those errors →
  first check `pgrep -af station.py` before suspecting hardware.

## Environment

- Conda env `lerobot` (Python 3.10), LeRobot 0.4.5 editable install from `lerobot/`.
- Extra deps installed in-env: `control`, `optuna`, `scipy` (tuning); `mujoco==3.8.1`,
  `placo==0.9.23`; `feetech-servo-sdk==1.0.0` (imports as `scservo_sdk`);
  `pyrealsense2==2.56.5.9235`; `opencv-python-headless==4.12.0`;
  `torchvision==0.20.1+cu124` (runtime-compatible with torch 2.5.1);
  `transformers==4.49.0` (Grounding DINO for the ball detector's GDINO fallback — **pin it**: transformers
  5.x pulls `huggingface-hub` 1.x which breaks lerobot's `<0.36.0` pin; keep hub at 0.35.x).

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
A third, plain USB webcam at **`/dev/video9`** (1280×720, no intrinsics — K guessed from a
70° FOV) watches the hand-held tag joystick for `teleop_tag.py`; it is unrelated to the
scene/wrist cams above.

**Gamepad** — Nintendo Switch Pro Controller (pygame name `"Pro Controller"`). Profiles
auto-detected in `gamepad_utils.py`. It also carries a **factory-calibrated 6-axis IMU**
that `hid_nintendo` exposes as a SEPARATE evdev device `Pro Controller (IMU)` — accel
4096 units/g, gyro 14247 units per °/s, bursty ~201 Hz (new info ~77 Hz). Read it with
python-evdev, **not** pygame/SDL, and **not** via `evdev.list_devices()` (which filters to
read-write devices) — glob `/dev/input/event*`. Needs the `input` group: prefix commands
with `sg input -c "..."`. Driver + calibration in `pad_imu.py`.

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
  Teleop ticks at **50 Hz** (`CMD_RATE_HZ`) decoupled from telemetry polling
  (`--rate`, 10 Hz); stick mapping is expo (`v∝s·|s|`) with `JOINT_SPEED` in
  **deg/s** (pan 60, lift/elbow 50 — user-validated 2026-06-11: fast + smooth,
  stops clean with D=64). Active servo gains are logged in each run's `summary.txt`.
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
- `record_pick.py` — **critical-path step 4**: record real-arm pick demos into a LeRobot
  dataset, driven by the scripted policy (`pick_ball.run_grasp`), one episode per cycle,
  appending across sessions. Always lands via `graceful_shutdown` on exit.

**Teleop by tagged joystick (see [TELEOP.md](TELEOP.md)):**
- `teleop_tag.py` — **the tag-teleop entry point.** A webcam tracks AprilTags on the
  hand-held Pro Controller; its motion drives TCP position + wrist roll, while the
  gamepad's own buttons carry the clutch (hold **L**) and gripper. Mapping is **clutched
  relative** (mouse-lift): `tcp = anchor_tcp + gain * M @ (hand - anchor_hand)`.
  Reuses every existing layer — `pick_ball.Kin`/`Twin`, `tag_body`/`tag_pose`,
  `gamepad_utils`, `station.JoystickManager`, `sim_backend.SimRobot`, `pad_imu.PadIMU`.
  Flags: `--sim`, `--gain`, `--max-speed`, `--axes`, `--cage`, `--view`, `--no-twin`,
  `--no-imu`, `--no-bridge`, `--no-rr-images`, `--dry-run`. Logs
  `outputs/teleop/<ts>/teleop.csv` including **per-stage tick timing**
  (`ms_grab`/`ms_detect`/`ms_ik`/`ms_twin`/`ms_rerun`/`ms_rest`) — read those before
  theorising about where the loop went slow.
- `pad_imu.py` — the controller's own 6-axis IMU as a teleop sensor: evdev reader
  (`PadIMU`), the IMU→tag-body extrinsic `X`, and a 9-state linear KF (position,
  velocity, **accel bias**) that bridges tag dropouts. Subcommands `probe`, `bias`,
  `align`, `check`, `view`, `solo`, `gyrocal`, `selftest`. Calibration →
  `outputs/calib/pad_imu.json` (X, bias, gyro scale). `gyrocal` measures the gyro's
  scale + bias against gravity with NO camera; `align` now refuses a bias measured
  while the pad is moving, and shows live axis coverage (X is only observable in the
  directions you actually rotate about).
- `vision/tag_pose.py` — **the base tag layer**: camera sources (`Webcam`, `RS`,
  `open_source`), the ArUco detector wrapper (`make_detector`, `detect`) and single-tag
  `SOLVEPNP_IPPE_SQUARE` PnP. Everything tag-shaped imports this.
- `vision/tag_body.py` — rigid **multi-tag body model** (bundle-adjusted) so any one
  visible tag yields the same body pose. Body frame = reference tag id 13. Used by the
  OLD two-tag joystick; the box below supersedes it but reuses `load_model`/`body_pose`.
- `vision/box_tags.py` — **the 3-face tag box** on the hand-held joystick (ids 7/11/15,
  `DICT_4X4_50`, 28 mm squares on a 32 mm half-box). Modes: `ident` (scan dictionaries,
  report ids + co-visibility), `debug` (live Rerun: green = decoded, red = quad found but
  NOT decoded), `window` (same in a pygame window — cv2 here is HEADLESS, no `imshow`),
  `capture` + `fit` (measure the real face geometry → `box_body.json`), `show` (3-D of the
  fitted model vs the ideal cube). `fit` resolves the per-pair flip by LOOP CLOSURE, not
  by vote count: both IPPE candidates cluster at ~25% each and are indistinguishable by
  count, but only the true set satisfies T(a→b)·T(b→c) = T(a→c) (0.25 mm vs 3.72 mm).
  Measured: faces land on three orthogonal axes to <1°, and sit 1.17 mm proud of an ideal
  corner (panel thickness — lateral +7.3%, axial +0.4%, so the 28 mm tag size is right).

**Vision (perception, WIP):**
- `vision/capture_frame.py` — grab one aligned color+depth+intrinsics frame from the
  top-down Realsense → `outputs/vision/<ts>/`. For offline detector dev (no arm).
- `vision/ball.py` — `fit_table_plane` (RANSAC, **works**, reused) + `localize_ball`
  (color+depth, **brittle** — wood reads orange; superseded by the YOLO localizer).
- `vision/cloud.py` — **the generic 3-D layer** (see `vision/PERCEPTION3D.md`):
  depth→cloud deprojection, multi-plane extraction (sequential RANSAC — separates the
  white plate from the wooden desk), and geometric-prior fitters with quality records
  (`fit_sphere_known_r`, `nearest_depth_center`). Nothing object-specific lives here.
- `vision/ball_yolo.py` — **the ball localizer**: 2-D box (student YOLO or GDINO) →
  `ball_from_box` → known-radius (Ø49 mm) RANSAC sphere fit on the box's point cloud
  + inliers/rms quality gate. Replaced the box-median-depth estimator, which was
  biased 1-3 cm toward the background by silhouette bleed (benchmark in
  `vision/bench_localize.py`: legacy was 7 mm low in z, 13 mm off in y — a missed
  grasp; sphere fit lands ±2 mm of known resting geometry). Run
  `python vision/ball_yolo.py [<capture_dir>]` to verify; writes `yolo_overlay.png`.
- `vision/cam_calib.py` — **THE camera-pose calibration path** (T_cam→base, base frame =
  robot base). Reference-anchored: the flat **white-plate plane** gives level (pitch/roll)
  + height (z); the **fixed desk ArUco tag** (`DICT_4X4_50` **id 13** — the finger tags are
  ids 1,2) gives x/y + yaw. Geometry uses no PnP — corner rays intersect the fitted plate
  plane. Subcommands (no arm motion): `level` (correct a tilted hand-eye from the plate
  prior), `anchor` (pin the desk tag in the base frame, once → `desk_anchor.json`), `recal`
  (single-frame T_cam→base from desk tag + plate after touching the camera; tag averaged
  ~16 frames → 0.3°/2 mm), `check` (self-check guard: table tilt < 2° AND ball sits on the
  plate). Run `check` after anything that touches the camera; `level`/`recal` to fix it.
  Supersedes the retired pink-heart `handeye_calib.py` (kept only for its `Realsense` class)
  and the gripper-tag `tag_handeye.py` (deleted — clustered poses left a ~10° tilt).
- `vision/scene_align.py` — **the alignment proof**: headless MuJoCo render overlaying, in
  the base frame, the robot at live joints + the camera's coloured point cloud + the ball +
  a z=0 reference + the measured plate + the camera as a pinhole frustum. If the cloud's
  table is flat at the plate and the ball sits on it, T_cam→base is right.
- `vision/scene_debug.py` — **the 3-D truth window**: live MuJoCo viewer with the arm
  (`--hold` keeps torque on for honest FK), the detected ball, the finger tags (cyan,
  FK-predicted) vs the live desk tag (magenta, `DICT_4X4_50` id 13), plus the workspace
  point cloud (Rerun) and the fitted plane. Use it first whenever localization looks wrong.
- `vision/autolabel.py` / `vision/detector_bench.py` / `vision/collect_marker_data.py` —
  the detector pipeline (see `vision/DETECTOR.md`): GDINO-teacher auto-labeling and the
  student-vs-teacher benchmark. **Now ball-only** — the old `heart_pink` class is retired
  (the gripper marker is AprilTags, detected by ArUco, not a trained YOLO).
  Student weights: `vision/models/scene_yolov8n.pt` (ball 7/7 vs teacher, 11 ms warm).

**Analysis:**
- `analyze_run.py` — offline analysis of a `station.py` run: gravity-removed wrist
  vibration (spectrum + correlation with joint motion) and IMU-frame localization via
  MuJoCo FK + Kabsch. `python analyze_run.py <run_dir>`.

**Controller tuning (Feetech STS3215):**
- `servo_tuning.py` — library: Lock-aware PID read/write (keeps torque on),
  `Acceleration` profiling, high-rate single-joint capture, trajectory generators
  (step / trapezoid / sine / chirp), step metrics, 2nd-order fit, FRF/resonance, cost.
  Also `ImuRecorder` (background wrist-IMU capture on the same `perf_counter` clock
  as `run_trajectory`) + `vibration_metrics` (gravity-removed wrist vibration →
  after-stop decay time, residual RMS, ring frequency) + `vibration_cost`.
- `calibrate.py` — CLI: `step` / `chirp` / `profile` / `autotune` (Optuna closed-loop
  PID search). `--sim` runs the whole loop on the MuJoCo twin; `--rate` caps samples.
  On hardware it records the wrist IMU per capture (opt out with `--no-imu`): `step`
  prints/plots after-stop ringing (`imu.png`/`imu.csv`), and `autotune`'s cost
  penalizes measured wrist ringing, not just encoder error.
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
`--p 32 --i 0 --d 32` (one run only — the next connect reapplies `gains.json`).
`calibrate.py` reads the IMU directly during every hardware capture; for resonance ID
beyond the encoder ceiling you can still merge `station.py`'s synced CSVs on `time_s`.

**Real-arm findings (2026-06-11, 7.8 V):** the after-stop oscillation seen in teleop is
an **underdamped servo loop** (factory D=32): 5–11 % overshoot and a 1–2 s wrist
ring-down at ~3–6 Hz on the big joints, worst on gravity-loaded `shoulder_lift`/
`elbow_flex`. Fix: **P=32 I=0 D=200** (now in `gains.json`) → overshoot ≈0 %, ring-down
dies into the noise floor, elbow micro-hunting gone (residual 0.45→0.20 m/s²), rise only
~35–90 ms slower. Two non-fixes learned: **I>0 causes limit-cycle hunting** (integrator
vs stiction — Optuna picks it to please steady-state cost; keep I=0), and there is an
irreducible **~0.5 m/s² holding buzz** at gravity-loaded poses (torque dither, present
with zero motion at any gains — don't chase it with PID).

## Local lerobot patches

`patches/lerobot_local.patch` holds two edits to the gitignored `lerobot/` install:
1. `so_follower.py`: per-motor P/I/D loaded from `config/gains.json` on connect
   (default P=32 I=0 D=32 — P must stay 32 for lift torque) + per-camera fault-tolerant
   connect.
2. `camera_realsense.py`: read-loop `stop_event` fix.

Reapply after a fresh lerobot clone: `git -C lerobot apply ../patches/lerobot_local.patch`.
`gains.json` lives in the tracked `config/` dir (alongside the gamepad layout
override) — edit it (not EEPROM) to change standing gains.

## Status & TODO

- [x] Motors confirmed, calibrated; arm lifts at 7.4 V (P_Coefficient=32)
- [x] Gamepad teleop, MuJoCo twin, Rerun diagnostics, current logging
- [x] Tuning toolkit built; analysis validated on synthetic data; full loop smoke-tested in `--sim`
- [x] Project under git, pushed to GitHub
- [x] Wrist IMU (ADXL345 on ESP32-C3) mounted on `wrist_roll`, streaming at 800 Hz
- [x] Unified cockpit `station.py` — motors + IMU + gamepad in one Rerun window,
      synced CSV logging (`station.csv` + `imu.csv` on a shared clock)
- [x] **YOLO ball detection → 3D point** (`vision/ball_yolo.py`, pretrained model)
- [x] **Hand-eye calibration done — reference-anchored** (`vision/cam_calib.py`, 2026-06-14):
      T_cam→base from the white-plate plane (level + z) + the fixed desk tag (`DICT_4X4_50`
      id 13, x/y + yaw) → `outputs/calib/handeye.json` + `desk_anchor.json`. After any camera
      move: `cam_calib.py recal` (single frame, no arm, 0.3°/2 mm) then `check`. Replaced the
      pink-heart `handeye_calib.py` (heart gone; file kept only for its `Realsense` class) and
      the gripper-tag `tag_handeye.py` (deleted — left a ~10° tilt). Verify with `scene_align.py`.
- [x] **Real-arm tuning done (2026-06-11)** — `calibrate.py` now records the wrist IMU
      per capture; diagnosed the after-stop oscillation (underdamped loop + gravity)
      and fixed it with **D=200** on pan/lift/elbow, persisted via `gains.json`
      (applied on every connect by the patched `configure()`). See Tuning workflow.
- [~] **Scripted pick (WIP)** — `pick_ball.py`: GDINO "basketball." detection (the
      vendored basketball.pt scores ~0 on the mini ball vs white plate), depth →
      handeye → base frame (z from depth — ball may sit on a holder; the scene has
      TWO planes: white ~1 cm plate on the wooden desk, RANSAC may fit either),
      multi-seed MuJoCo IK (fingers-down is a soft preference, position dominates),
      twin preview, staged grasp, graceful landing on every exit. **First real grasp
      pending.** Geometry context 2026-06-11: cam ~50–70 cm top-down; base origin is
      ~3 cm above the plate (plate ≈ −29 mm, desk ≈ −43 mm in base coords).
- [~] **Fast scene detector (v1 trained overnight)** — GDINO teacher auto-labels
      (`vision/autolabel.py`), yolov8n student, benchmark vs teacher
      (`vision/detector_bench.py`). Dataset 45 frames / 204 augmented under
      `outputs/vision/dataset_2026-06-11/`. See `vision/DETECTOR.md`. v1 is
      single-scene — collect varied data with Javad before trusting it broadly.
      `pick_ball.py` now uses the student first (11 ms) with GDINO fallback. The pipeline
      is **ball-only now** — the `heart_pink` class is retired (gripper marker is AprilTags).
- [x] **Pick choreography validated in sim (2026-06-11 night)** — full sequence
      (plan → above → descend → contact-close → lift → replace → land) ran against
      `sim_backend.SimRobot`: plan 0.4 mm, sim TCP tracks within ~7 mm. Only physical
      contact remains untested — first real grasp happens with Javad present.
- [ ] Resonance ID / input shaping from chirp + IMU (encoder-only ceiling ~20–30 Hz)
- [x] **`graceful_shutdown` soft landing (2026-06-12)** — old `REST_POSE` (lift 40)
      left the gripper 87 mm up; torque-off dropped it with a clunk. Now two-stage:
      full-torque ease to a raised rest (~14 mm above the measured min-energy pose
      from station.csv cold starts), then `Torque_Limit` (RAM, no EEPROM wear)
      clamped to 150/1000 for a compliant float-down to `SETTLE_POSE` — stalls
      gently on contact, error zeroed, limit restored. Verified on hardware:
      settles ±2.6 deg of equilibrium, TL back at 1000. The STS3215 has no real
      torque/impedance mode; RAM `Torque_Limit` is the compliance lever.
- [~] **Tag teleop (WIP)** — `teleop_tag.py`: webcam-tracked tag joystick → clutched
      relative TCP control, with the controller's own IMU fused in by a 9-state linear KF.
      Full write-up in [TELEOP.md](TELEOP.md). Works on the real arm. **2026-08-21 loop
      profiling**: idle IK was solving a frozen target and discarding it (168 → 0.03 ms);
      `_ik_pass` ran provably-dead iterations once a joint pinned (50–150 of every pass
      were no-ops — fixed-point break is bit-identical and 4.4× faster); `ms_detect`
      (28.8 ms, full 1280×720 + subpix) is now the largest single cost. Hardware is NOT
      the constraint (i9-13900H, 33 GFLOP/s single-thread).
      **Tag coverage — SOLVED 2026-08-31** by replacing the two loose tags with a
      3-face tag box (below): `tags >= 2` went from 0.4% to 75.5% of frames.
- [x] **3-face tag box + IMU calibration (2026-08-31)** — the hand-held marker is now a
      32 mm half-box with one tag per face, `DICT_4X4_50` ids **7, 11, 15**, 28 mm black
      square. Built and measured with `vision/box_tags.py` (`ident` → `capture` → `fit` →
      `show`). Model → `outputs/calib/box_body.json` (same schema as `tag_body.load_model`;
      `pad_imu.py --model` points at it by default). Two or more faces are visible 75.5% of
      the time, which makes the point set non-planar and kills the planar two-fold
      ambiguity that used to flip the old single-tag pose.
      **Also fixed a real sensor error: the Pro Controller gyro reads ~12% HIGH.** The
      `hid_nintendo` resolution of 14247 units/°/s should be ~16200. Found twice,
      independently — vision (tag rotation vs gyro, ratio 0.876) and gravity alone
      (still→still accelerometer transitions, no camera, 0.882). Correcting scale AND
      bias together takes the gravity residual from 14.9° to 1.5°. Re-measure per
      controller with `pad_imu.py gyrocal` (no camera needed, 60 s). With that fixed,
      four `align` runs agree to **1.34° average / 1.81° worst** (before: 43°), and the
      live camera-vs-gyro gap is **1.37°** with 2+ tags.
- [x] **Tag id collision — gone.** The joystick tags are now ids 7/11/15, so nothing
      clashes with the finger tags (1, 2) or the desk anchor (13).
- [ ] Realsense + wrist cam USB bandwidth (works alone, stalls together)
- [ ] Collect sim episodes → cloud ACT training (plan exists)

Refactor backlog (small, not urgent — the layering is mostly right: one IMU parser,
one gamepad layer, one gains source, one tuning library):
- [ ] Shared constants (`PORT`, output roots) are duplicated in `station.py` /
      `calibrate.py` — hoist into one small config module / settings file
- [ ] Two IMU sinks duplicate the align-to-host-clock logic (`station.imu_loop`,
      `servo_tuning.ImuRecorder`) — unify into one recorder with pluggable sinks
- [ ] `gains.json`: add named per-mode profiles (teleop / point-to-point / VLA)
      when the scripted pick-place needs them; trapezoid profile for auto-mode
      moves lives in `servo_tuning.trapezoid_ref` (already written, unused)

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
