#!/usr/bin/env python
"""
SO-101 station — one command, one cockpit.

Connects the arm, opens a single Rerun viewer, and streams everything onto one
timeline: motor state (position / load / current / voltage / status), the gamepad
(stick positions + buttons), the wrist IMU (ADXL345 on the ESP32-C3), and — when
asked — the cameras and a MuJoCo 3-D twin. Gamepad teleop and typed commands drive
the arm; the joystick may be plugged/unplugged mid-run.

Replaces the old trio: teleop_gamepad.py, command_log.py, diagnose_motors.py.

Usage:
    python station.py                  # teleop + motors + IMU → Rerun + CSV
    python station.py --observe        # read-only (no teleop), for diagnostics
    python station.py --health         # + startup register report
    python station.py --cameras        # + realsense & wrist images in Rerun
    python station.py --twin           # + MuJoCo 3-D viewer
    python station.py --no-imu         # skip the ADXL thread
    python station.py --observe --health --torque 1000 --fix-pgain   # restore EEPROM

Typed commands (while running):
    <joint> <deg>   move one joint (e.g. 'elbow_flex 45')
    all <deg>       move all joints
    hold            resend current positions (clears overload)
    torque off/on   limp / hold
    q / quit        exit
"""

import argparse
import csv
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Headless gamepad: read the joystick without opening a pygame window.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

import numpy as np
import pygame
import rerun as rr
import rerun.blueprint as rrb

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

from gamepad_utils import (
    MOTOR_NAMES, JOINT_LIMITS,
    detect_profile, get_joint_deltas, apply_deltas, is_neutral,
    graceful_shutdown, DeltaSmoother,
)

PORT     = "/dev/ttyACM1"  # CH343 arm controller (ESP32-C3 IMU takes ttyACM0)
ROBOT_ID = "so101"
LOG_DIR  = Path("/home/javad/workspace/lerobot_all/outputs/logs")
SCENE_XML = "/home/javad/workspace/lerobot_all/SO-ARM100/Simulation/SO101/scene.xml"

# ── Motor register decoders ────────────────────────────────────────────────────

def decode_load(raw: int) -> float:
    mag = (raw & 0x3FF) / 10.0
    return -mag if (raw >> 10) & 1 else mag

def decode_current_mA(raw: int) -> float:
    return raw * 6.5  # 1 unit = 6.5 mA per STS3215 datasheet

def decode_voltage(raw: int) -> float:
    return raw / 10.0

def decode_status(raw: int) -> list[str]:
    flags = []
    if raw & (1 << 0): flags.append("VOLT")
    if raw & (1 << 2): flags.append("TEMP")
    if raw & (1 << 3): flags.append("STALL")
    if raw & (1 << 4): flags.append("LOAD")
    if raw & (1 << 5): flags.append("OVERLOAD")
    if raw & (1 << 6): flags.append("ENC")
    return flags if flags else ["OK"]

# ── Startup register health report (was diagnose_motors.py) ─────────────────────

CONFIG_REGISTERS = [
    ("Max_Torque_Limit",  "max torque (0-1000 = 0-100%)",                       1000),
    ("Torque_Enable",     "1=hold, 0=limp",                                     1),
    ("P_Coefficient",     "proportional gain (factory=32, lerobot halves to 16)", 32),
    ("I_Coefficient",     "integral gain (factory=0)",                          0),
    ("D_Coefficient",     "derivative gain (factory=32)",                       32),
    ("CW_Dead_Zone",      "CW deadband counts (factory=1, ~0.088°)",            1),
    ("CCW_Dead_Zone",     "CCW deadband counts (factory=1, ~0.088°)",           1),
    ("Overload_Torque",   "overload threshold % (factory=80)",                  80),
    ("Protection_Time",   "overload trip time ms (factory=200)",                200),
    ("Status",            "fault bitmask (0=OK)",                               0),
]


def health_report(bus, file=None) -> None:
    """Read config registers for every motor, print a report, flag anomalies."""
    lines = ["=" * 72, "MOTOR CONFIGURATION REPORT", "=" * 72]
    issues = []
    for motor in MOTOR_NAMES:
        lines.append(f"\n  {motor}\n  {'─' * 40}")
        for reg, desc, expected in CONFIG_REGISTERS:
            try:
                val = bus.read(reg, motor, normalize=False)
            except Exception as e:
                lines.append(f"    {reg:<26} ERR:{e}")
                continue
            if reg == "Status":
                flags = decode_status(val)
                warn = "  ⚠ OVERLOAD ACTIVE" if "OVERLOAD" in flags else ""
                lines.append(f"    {reg:<26} {val:>5}   [{', '.join(flags)}]{warn}")
                if "OVERLOAD" in flags:
                    issues.append(f"{motor}: overload flag set — 'hold' or --clear-overload to clear")
            elif val != expected:
                lines.append(f"    {reg:<26} {val:>5}   ← expected {expected}  ({desc})")
                if reg == "Max_Torque_Limit" and val < 600:
                    issues.append(f"{motor}: Max_Torque_Limit={val} LOW — use --torque 1000")
                elif reg == "P_Coefficient" and val < 32:
                    issues.append(f"{motor}: P_Coefficient={val} — use --fix-pgain")
            else:
                lines.append(f"    {reg:<26} {val:>5}   ({desc})")
    lines.append("\n" + "=" * 72)
    lines += (["ISSUES DETECTED:"] + [f"  ⚠  {i}" for i in issues]
              if issues else ["No configuration issues detected."])
    lines.append("=" * 72)
    text = "\n".join(lines)
    print(text)
    if file:
        file.write(text + "\n")


def apply_fixes(bus, robot, args, file=None) -> None:
    if args.torque is not None:
        print(f"\nSetting Max_Torque_Limit → {args.torque} for all motors (EEPROM)...")
        for name in MOTOR_NAMES:
            bus.write("Max_Torque_Limit", name, args.torque, normalize=False)
        print("Done. Power-cycle the arm so EEPROM reloads into RAM Torque_Limit.")
        if file: file.write(f"ACTION: Max_Torque_Limit={args.torque} for all motors.\n")
    if args.fix_pgain:
        print("\nRestoring P_Coefficient → 32 for all motors...")
        for name in MOTOR_NAMES:
            bus.write("P_Coefficient", name, 32, normalize=False)
        print("Done.")
        if file: file.write("ACTION: P_Coefficient restored to 32.\n")
    if args.clear_overload:
        print("\nClearing overload flags (sending hold)...")
        obs = robot.get_observation()
        for name in MOTOR_NAMES:
            bus.write("Goal_Position", name, int(obs.get(f"{name}.pos", 0.0)), normalize=False)
        print("Done.")
        if file: file.write("ACTION: Goal_Position reset to clear overload.\n")

# ── Hot-pluggable joystick ──────────────────────────────────────────────────────

class JoystickManager:
    """Open the joystick on demand and survive plug/unplug while running.

    pygame posts JOYDEVICEADDED for joysticks already present when the subsystem
    initialises, so handle_event() covers both startup and live hot-plug.
    """

    def __init__(self):
        self.joystick = None
        self.profile = None

    def _open(self, index: int) -> None:
        js = pygame.joystick.Joystick(index)
        js.init()
        self.joystick = js
        self.profile = detect_profile(js)
        print(f"  Joystick connected: {js.get_name()}")

    def handle_event(self, event) -> None:
        if event.type == pygame.JOYDEVICEADDED and self.joystick is None:
            self._open(event.device_index)
        elif event.type == pygame.JOYDEVICEREMOVED and self.joystick is not None:
            print("  Joystick disconnected — arm holding last goal.")
            self.joystick = None
            self.profile = None

    @property
    def connected(self) -> bool:
        return self.joystick is not None

# ── IMU thread (ADXL345 on the ESP32-C3) ────────────────────────────────────────

def imu_loop(t0: float, stop: threading.Event, log_dir: Path = None,
             entity: str = "imu/wrist_roll") -> None:
    """Stream IMU samples onto the shared Rerun timeline (and imu.csv). Best-effort:
    if the C3 isn't present the thread warns and exits, leaving the arm running.

    time_s is built on the SAME t0 as the motor loop, so imu.csv and station.csv are
    directly syncable; t_micros keeps the raw 800 Hz firmware clock for fine timing."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ESP32"))
    try:
        from imu_serial import stream_samples, SCALE
        gen = stream_samples()
    except Exception as e:
        print(f"  IMU off ({e}) — continuing without it.")
        return

    imu_f = writer = None
    if log_dir is not None:
        imu_f = open(log_dir / "imu.csv", "w", newline="")
        writer = csv.writer(imu_f)
        writer.writerow(["time_s", "t_micros", "x_raw", "y_raw", "z_raw"])

    print(f"  IMU streaming → {entity}" + ("  + imu.csv" if writer else ""))
    imu_t0_us = None
    base_s = 0.0
    n = 0
    try:
        for (t_us, x_raw, y_raw, z_raw) in gen:
            if stop.is_set():
                break
            if imu_t0_us is None:
                imu_t0_us = t_us
                base_s = time.perf_counter() - t0   # align to the shared timeline
            t_rel = base_s + (t_us - imu_t0_us) / 1e6
            x, y, z = x_raw * SCALE, y_raw * SCALE, z_raw * SCALE
            rr.set_time("time", duration=t_rel)
            rr.log(f"{entity}/accel_x",   rr.Scalars(x))
            rr.log(f"{entity}/accel_y",   rr.Scalars(y))
            rr.log(f"{entity}/accel_z",   rr.Scalars(z))
            rr.log(f"{entity}/magnitude", rr.Scalars(float(np.sqrt(x * x + y * y + z * z))))
            if writer:
                writer.writerow([f"{t_rel:.4f}", t_us, x_raw, y_raw, z_raw])
                n += 1
                if n % 400 == 0:            # ~0.5 s of samples — flush for crash safety
                    imu_f.flush()
    finally:
        if imu_f:
            imu_f.close()

# ── Typed commands ──────────────────────────────────────────────────────────────

def _input_thread(q: queue.Queue) -> None:
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            break
        if line.strip():
            q.put(line.strip())


def process_command(cmd: str, goal_pos: dict, robot, bus) -> bool:
    """Apply a typed command. Returns True to exit."""
    parts = cmd.lower().split()
    if not parts:
        return False
    if parts[0] in ("q", "quit", "exit"):
        return True
    if parts[0] in ("?", "help"):
        print("  <joint> <deg> | all <deg> | hold | torque off|on | q")
        print(f"  joints: {', '.join(MOTOR_NAMES)}")
    elif parts[0] == "hold":
        obs = robot.get_observation()
        for n in MOTOR_NAMES:
            goal_pos[n] = obs.get(f"{n}.pos", goal_pos[n])
        robot.send_action({f"{n}.pos": goal_pos[n] for n in MOTOR_NAMES})
        print("  Holding current positions.")
    elif parts[0] == "torque" and len(parts) >= 2:
        if parts[1] == "off":
            bus.disable_torque(); print("  Torque DISABLED.")
        elif parts[1] == "on":
            bus.enable_torque()
            robot.send_action({f"{n}.pos": goal_pos[n] for n in MOTOR_NAMES})
            print("  Torque ENABLED.")
    elif parts[0] == "all" and len(parts) >= 2:
        try:
            deg = float(parts[1])
            for n in MOTOR_NAMES:
                lo, hi = JOINT_LIMITS[n]
                goal_pos[n] = max(lo, min(hi, deg))
            robot.send_action({f"{n}.pos": goal_pos[n] for n in MOTOR_NAMES})
            print(f"  All joints → {deg:.1f}° (clipped)")
        except ValueError:
            print(f"  Bad angle: {parts[1]}")
    elif parts[0] in MOTOR_NAMES and len(parts) >= 2:
        name = parts[0]
        try:
            deg = float(parts[1])
            lo, hi = JOINT_LIMITS[name]
            goal_pos[name] = max(lo, min(hi, deg))
            robot.send_action({f"{n}.pos": goal_pos[n] for n in MOTOR_NAMES})
            print(f"  {name} → {goal_pos[name]:.1f}°")
        except ValueError:
            print(f"  Bad angle: {parts[1]}")
    else:
        print(f"  Unknown: '{cmd}'. Type 'help'.")
    return False

# ── Rerun logging ───────────────────────────────────────────────────────────────

def log_controller(jm: JoystickManager) -> None:
    js = jm.joystick
    def axis(i):
        try:    return js.get_axis(i)
        except Exception: return 0.0
    rr.log("controller/left_stick",  rr.Points2D([[axis(0), axis(1)]], radii=0.06))
    rr.log("controller/right_stick", rr.Points2D([[axis(2), axis(3)]], radii=0.06))
    for key, idx in jm.profile["shoulder"].items():
        try:    pressed = float(js.get_button(idx))
        except Exception: pressed = 0.0
        rr.log(f"controller/buttons/{key}", rr.Scalars(pressed))


def log_stick_bounds() -> None:
    """Static [-1,1] frame so the stick views have stable axes."""
    box = rr.Boxes2D(centers=[[0.0, 0.0]], half_sizes=[[1.0, 1.0]])
    rr.log("controller/left_stick/frame", box, static=True)
    rr.log("controller/right_stick/frame", box, static=True)


def build_blueprint(args) -> rrb.Blueprint:
    """Default layout so every stream gets a panel without hand-arranging the UI."""
    views = [
        rrb.TimeSeriesView(origin="motors",          name="Motors"),
        rrb.TimeSeriesView(origin="imu/wrist_roll",  name="IMU (wrist_roll)"),
        rrb.Spatial2DView(origin="controller",       name="Controller"),
        rrb.TimeSeriesView(origin="diagnostics",      name="Loop rate (Hz)"),
    ]
    if args.cameras:
        views.append(rrb.Spatial2DView(origin="cameras", name="Cameras"))
    return rrb.Blueprint(rrb.Grid(*views), collapse_panels=True)

# ── Cameras / twin (lazy) ───────────────────────────────────────────────────────

def make_robot_config(args):
    cameras = {}
    if args.cameras:
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        if not args.no_realsense:
            cameras["realsense"] = RealSenseCameraConfig(
                serial_number_or_name="117222251972", fps=15, width=640, height=480)
        if not args.no_wrist:
            cameras["wrist"] = OpenCVCameraConfig(
                index_or_path=15, fps=25, width=640, height=480)
    return SOFollowerRobotConfig(port=args.port, id=ROBOT_ID, cameras=cameras)

def detection_loop(detector, ball_class, t0, stop, frame_slot, conf=0.25):
    """Run YOLO ball detection OFF the control loop, in its own thread, so teleop
    stays at full rate. Reads the most recent realsense frame from frame_slot[0]
    (RGB) and logs the 2-D box to Rerun. Detection runs as fast as the GPU allows,
    decoupled from the motor poll rate."""
    import cv2
    last = None
    while not stop.is_set():
        frame = frame_slot[0]
        if frame is None or frame is last:      # nothing new since last inference
            time.sleep(0.01)
            continue
        last = frame
        try:
            res = detector(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), conf=conf, verbose=False)[0]
            best = None
            for c, p, b in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy):
                if res.names[int(c)] == ball_class and (best is None or float(p) > best[0]):
                    best = (float(p), [float(v) for v in b])
            rr.set_time("time", duration=time.perf_counter() - t0)
            if best is not None:
                score, (x1, y1, x2, y2) = best
                rr.log("cameras/realsense/ball", rr.Boxes2D(
                    array=[[x1, y1, x2 - x1, y2 - y1]],
                    array_format=rr.Box2DFormat.XYWH, labels=[f"ball {score:.2f}"]))
            else:
                rr.log("cameras/realsense/ball", rr.Clear(recursive=False))
        except Exception as e:
            print(f"  Ball detection error ({e}); detector thread stopping, teleop continues.")
            return

# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SO-101 station — unified cockpit.")
    parser.add_argument("--port", default=PORT, help="Arm serial port (default: /dev/ttyACM1)")
    parser.add_argument("--rate", type=float, default=10, help="Motor poll rate Hz (default: 10)")
    parser.add_argument("--observe", action="store_true", help="Read-only: no teleop, no typed motion.")
    parser.add_argument("--cameras", action="store_true", help="Stream realsense + wrist images to Rerun.")
    parser.add_argument("--no-realsense", action="store_true", help="With --cameras, skip the realsense.")
    parser.add_argument("--no-wrist", action="store_true", help="With --cameras, skip the wrist cam.")
    parser.add_argument("--detect-ball", action="store_true",
                        help="With --cameras, run YOLO ball detection on the realsense feed (2-D box in Rerun).")
    parser.add_argument("--twin", action="store_true", help="Launch the MuJoCo 3-D twin viewer.")
    parser.add_argument("--no-imu", action="store_true", help="Skip the ADXL345 IMU thread.")
    parser.add_argument("--no-log", action="store_true", help="Skip the CSV log.")
    parser.add_argument("--health", action="store_true", help="Print the startup register report.")
    parser.add_argument("--torque", type=int, default=None, help="Set Max_Torque_Limit (0-1000) for all motors.")
    parser.add_argument("--fix-pgain", action="store_true", help="Restore P_Coefficient=32 for all motors.")
    parser.add_argument("--clear-overload", action="store_true", help="Clear overload flags (send hold).")
    args = parser.parse_args()

    # ── Connect arm ─────────────────────────────────────────────────────────────
    robot = SOFollower(make_robot_config(args))
    print("Connecting arm" + (" + cameras" if args.cameras else "") + "...")
    robot.connect()
    bus = robot.bus

    # ── Log dir ─────────────────────────────────────────────────────────────────
    log_dir = csv_path = None
    summary_f = None
    if not args.no_log:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = LOG_DIR / ts
        log_dir.mkdir(parents=True, exist_ok=True)
        csv_path = log_dir / "station.csv"
        summary_f = open(log_dir / "summary.txt", "w")
        summary_f.write(f"Run: {datetime.now().isoformat()}  Port: {args.port}\n\n")
        print(f"Logging to: {log_dir}")

    # ── Health report + fixes ───────────────────────────────────────────────────
    if args.health:
        health_report(bus, file=summary_f)
    if args.torque is not None or args.fix_pgain or args.clear_overload:
        apply_fixes(bus, robot, args, file=summary_f)
    if summary_f:
        summary_f.flush()

    obs = robot.get_observation()
    goal_pos = {n: obs.get(f"{n}.pos", 0.0) for n in MOTOR_NAMES}

    # ── Ball detector (optional) ─────────────────────────────────────────────────
    # Load + warm up BEFORE Rerun spawns its GPU viewer, so torch's CUDA init can't
    # race Rerun's Vulkan init ("device busy"). Fault-tolerant: any failure just
    # disables detection — vision is auxiliary and must never block arm control.
    detector = None
    if args.detect_ball and args.cameras and not args.no_realsense:
        try:
            import cv2  # noqa: F401  (used in the loop)
            sys.path.insert(0, str(Path(__file__).resolve().parent / "vision"))
            from ball_yolo import load_model, BALL_CLASS
            print("Loading ball detector...")
            detector = load_model()
            detector(np.zeros((480, 640, 3), np.uint8), verbose=False)  # warm up CUDA now
            print("Ball detector ready.")
        except Exception as e:
            print(f"  Ball detector unavailable ({e}); continuing without detection.")
            detector = None

    # ── Rerun (auto-opens a viewer with a default layout) ───────────────────────
    rr.init("so101_station", spawn=True)
    rr.send_blueprint(build_blueprint(args))
    log_stick_bounds()

    # ── IMU thread ──────────────────────────────────────────────────────────────
    # t0 is the shared monotonic origin for every time_s in this run; wall0 anchors
    # it to absolute time so logs can also be synced across runs / external tools.
    t0 = time.perf_counter()
    wall0 = time.time()
    if summary_f:
        summary_f.write(f"t0_unix_epoch: {wall0:.6f}  "
                        f"({datetime.fromtimestamp(wall0).isoformat()})\n")
        summary_f.flush()
    imu_stop = threading.Event()
    imu_t = None
    if not args.no_imu:
        imu_t = threading.Thread(target=imu_loop, args=(t0, imu_stop, log_dir), daemon=True)
        imu_t.start()

    # Ball detection runs in its own thread so YOLO never throttles the control loop.
    detect_stop = threading.Event()
    detect_t = None
    frame_slot = [None]                         # latest realsense RGB frame (shared)
    if detector is not None:
        detect_t = threading.Thread(target=detection_loop,
                                    args=(detector, BALL_CLASS, t0, detect_stop, frame_slot),
                                    daemon=True)
        detect_t.start()

    # ── Joystick (headless) ─────────────────────────────────────────────────────
    pygame.init()
    jm = JoystickManager()
    smoother = DeltaSmoother(alpha=0.45)

    # ── Twin (lazy import) ──────────────────────────────────────────────────────
    twin = mj_model = mj_data = None
    if args.twin:
        import math
        import mujoco
        import mujoco.viewer
        os.chdir(os.path.dirname(SCENE_XML))
        mj_model = mujoco.MjModel.from_xml_path(SCENE_XML)
        mj_data = mujoco.MjData(mj_model)
        twin = mujoco.viewer.launch_passive(mj_model, mj_data)

    # ── Typed-command thread ────────────────────────────────────────────────────
    cmd_queue: queue.Queue = queue.Queue()
    if not args.observe:
        threading.Thread(target=_input_thread, args=(cmd_queue,), daemon=True).start()

    mode = "observe" if args.observe else "teleop"
    print(f"\nStation running ({mode}) at {args.rate} Hz. "
          + ("Type 'help' for commands. " if not args.observe else "")
          + "Ctrl-C to stop.\n")

    dt = 1.0 / args.rate
    prev_loop = None
    error_counts = {n: 0 for n in MOTOR_NAMES}
    writer = None
    csv_f = None
    if csv_path:
        csv_f = open(csv_path, "w", newline="")
        writer = csv.DictWriter(csv_f, fieldnames=[
            "time_s", "motor", "position_deg", "goal_deg",
            "load_pct", "current_mA", "voltage_v", "temperature_c",
            "moving", "status_flags"])
        writer.writeheader()

    try:
        while True if not twin else twin.is_running():
            loop_start = time.perf_counter()
            t_rel = loop_start - t0

            # ── pygame / joystick events (hot-plug) ──────────────────────────────
            for event in pygame.event.get():
                jm.handle_event(event)

            # ── typed commands ───────────────────────────────────────────────────
            should_exit = False
            while not cmd_queue.empty():
                if process_command(cmd_queue.get_nowait(), goal_pos, robot, bus):
                    should_exit = True
                    break
            if should_exit:
                break

            # ── joystick → action ────────────────────────────────────────────────
            if jm.connected and not args.observe:
                deltas = smoother(get_joint_deltas(jm.joystick, jm.profile))
                if not is_neutral(deltas):
                    goal_pos = apply_deltas(goal_pos, deltas)
                    robot.send_action({f"{n}.pos": goal_pos[n] for n in MOTOR_NAMES})

            # ── read motors → Rerun + CSV ────────────────────────────────────────
            rr.set_time("time", duration=t_rel)
            if prev_loop is not None:
                rr.log("diagnostics/loop_hz", rr.Scalars(1.0 / max(loop_start - prev_loop, 1e-6)))
            prev_loop = loop_start
            obs = robot.get_observation()

            for name in MOTOR_NAMES:
                pos_deg = obs.get(f"{name}.pos", 0.0)
                try:
                    raw_load   = bus.read("Present_Load",        name, normalize=False)
                    raw_temp   = bus.read("Present_Temperature", name, normalize=False)
                    raw_volt   = bus.read("Present_Voltage",     name, normalize=False)
                    raw_cur    = bus.read("Present_Current",     name, normalize=False)
                    raw_status = bus.read("Status",              name, normalize=False)
                    raw_moving = bus.read("Moving",              name, normalize=False)
                except Exception as e:
                    error_counts[name] += 1
                    if error_counts[name] == 1:
                        print(f"  Read error {name}: {e}")
                    elif error_counts[name] % 30 == 0 and not args.observe:
                        try:
                            robot.send_action({f"{n}.pos": goal_pos[n] for n in MOTOR_NAMES})
                        except Exception:
                            pass
                    continue
                error_counts[name] = 0

                load_pct = decode_load(raw_load)
                temp_c   = float(raw_temp)
                volt_v   = decode_voltage(raw_volt)
                cur_mA   = decode_current_mA(raw_cur)
                status   = decode_status(raw_status)

                base = f"motors/{name}"
                rr.log(f"{base}/position_deg",  rr.Scalars(pos_deg))
                rr.log(f"{base}/goal_deg",      rr.Scalars(goal_pos[name]))
                rr.log(f"{base}/load_pct",      rr.Scalars(load_pct))
                rr.log(f"{base}/current_mA",    rr.Scalars(cur_mA))
                rr.log(f"{base}/voltage_v",     rr.Scalars(volt_v))
                rr.log(f"{base}/temperature_c", rr.Scalars(temp_c))
                rr.log(f"{base}/moving",        rr.Scalars(float(raw_moving)))
                if raw_status != 0:
                    rr.log(f"{base}/STATUS_FAULT", rr.Scalars(float(raw_status)))

                if writer:
                    writer.writerow({
                        "time_s": f"{t_rel:.3f}", "motor": name,
                        "position_deg": f"{pos_deg:.2f}", "goal_deg": f"{goal_pos[name]:.2f}",
                        "load_pct": f"{load_pct:.1f}", "current_mA": f"{cur_mA:.1f}",
                        "voltage_v": f"{volt_v:.2f}", "temperature_c": f"{temp_c:.0f}",
                        "moving": raw_moving, "status_flags": "|".join(status)})

            if writer:
                csv_f.flush()

            # ── controller + cameras + twin ──────────────────────────────────────
            if jm.connected:
                log_controller(jm)

            if args.cameras:
                for cam in ("realsense", "wrist"):
                    frame = obs.get(cam)
                    if isinstance(frame, np.ndarray):
                        rr.log(f"cameras/{cam}", rr.Image(frame))
                        if detector is not None and cam == "realsense":
                            frame_slot[0] = frame      # hand off to the detector thread

            if twin:
                import math
                import mujoco
                for name in MOTOR_NAMES:
                    if f"{name}.pos" in obs:
                        mj_data.joint(name).qpos[0] = math.radians(obs[f"{name}.pos"])
                mujoco.mj_forward(mj_model, mj_data)
                twin.sync()

            time.sleep(max(dt - (time.perf_counter() - loop_start), 0.0))

    except KeyboardInterrupt:
        pass
    except Exception as e:
        # Any unexpected loop error: report it, but still fall through to finally
        # so the arm always gets a smooth landing (never left torque-on / collapsing).
        print(f"\nUnexpected error: {e!r}\nLanding the arm safely...")
    finally:
        imu_stop.set()
        if imu_t:
            imu_t.join(timeout=1.0)   # let the IMU thread flush + close imu.csv
        detect_stop.set()
        if detect_t:
            detect_t.join(timeout=1.0)
        if twin:
            twin.close()
        graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception as e:
            print(f"  Disconnect warning (torque disable likely succeeded): {e}")
        pygame.quit()
        if csv_f:
            csv_f.close()
        if summary_f:
            summary_f.close()
        if log_dir:
            print(f"Logs saved to {log_dir}")


if __name__ == "__main__":
    main()
