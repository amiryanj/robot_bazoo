#!/usr/bin/env python
"""
SO-101 motor diagnostic streamer — logs live motor state to Rerun + CSV + text summary.

On startup: reads ALL config registers and prints a health report, highlighting anything
that looks wrong (low Max_Torque_Limit, overload flag, wrong P gain, etc.)

Then streams live data to:
  - Rerun viewer (spawns automatically)
  - outputs/logs/<timestamp>/stream.csv   (all sensors, every poll)
  - outputs/logs/<timestamp>/summary.txt  (startup register dump + recommendations)

Usage:
    python diagnose_motors.py                    # read-only
    python diagnose_motors.py --torque 1000      # restore Max_Torque_Limit to 100%
    python diagnose_motors.py --fix-pgain        # restore P_Coefficient to factory 32
    python diagnose_motors.py --torque 1000 --fix-pgain  # both fixes

Known issues this script helps diagnose:
  - Max_Torque_Limit written low by a previous run (persists in EEPROM)
  - Overload protection triggered (motor drops to 20% torque after 200ms overload)
  - P_Coefficient halved by lerobot configure() — reduces corrective torque
  - Deadband (CW/CCW_Dead_Zone) too large — motor ignores small gravity sag
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import rerun as rr

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

PORT      = "/dev/ttyACM0"
ROBOT_ID  = "so101"
RATE_HZ   = 10
LOG_DIR   = Path("/home/javad/workspace/lerobot_all/outputs/logs")

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]

# ── Decoders ──────────────────────────────────────────────────────────────────

def decode_load(raw: int) -> float:
    magnitude = (raw & 0x3FF) / 10.0
    direction = (raw >> 10) & 1
    return -magnitude if direction else magnitude

def decode_voltage(raw: int) -> float:
    return raw / 10.0

def decode_current_mA(raw: int) -> float:
    return raw * 6.5  # 1 unit = 6.5 mA per STS3215 datasheet

def decode_status(raw: int) -> list[str]:
    flags = []
    if raw & (1 << 0): flags.append("VOLTAGE_ERR")
    if raw & (1 << 2): flags.append("TEMP_ERR")
    if raw & (1 << 3): flags.append("STALL")
    if raw & (1 << 4): flags.append("LOAD_ERR")
    if raw & (1 << 5): flags.append("OVERLOAD")   # ← most relevant
    if raw & (1 << 6): flags.append("ENCODER_ERR")
    return flags if flags else ["OK"]

# ── Startup register dump ─────────────────────────────────────────────────────

CONFIG_REGISTERS = [
    ("Max_Torque_Limit",  "EEPROM", "max torque (0-1000 = 0-100%)", 1000),
    ("Torque_Enable",     "RAM",   "1=hold, 0=limp",                 1),
    ("Torque_Limit",      "RAM",   "active torque cap (reloads from Max_Torque_Limit)", None),
    ("P_Coefficient",     "EEPROM","proportional gain (factory=32, lerobot=16)",         32),
    ("I_Coefficient",     "EEPROM","integral gain (factory=0)",                          0),
    ("D_Coefficient",     "EEPROM","derivative gain (factory=32)",                       32),
    ("CW_Dead_Zone",      "EEPROM","CW deadband counts (factory=1, ~0.088°)",            1),
    ("CCW_Dead_Zone",     "EEPROM","CCW deadband counts (factory=1, ~0.088°)",           1),
    ("Overload_Torque",   "EEPROM","overload threshold % (factory=80)",                  80),
    ("Protection_Time",   "EEPROM","overload trip time ms (factory=200)",                200),
    ("Status",            "RAM",   "fault bitmask (0=OK)",                               0),
]

def read_config(bus, name: str) -> dict:
    result = {}
    for reg, mem, desc, expected in CONFIG_REGISTERS:
        try:
            val = bus.read(reg, name, normalize=False)
            result[reg] = val
        except Exception as e:
            result[reg] = f"ERR:{e}"
    return result


def print_config_report(configs: dict, file=None) -> list[str]:
    issues = []
    lines = []
    lines.append("=" * 72)
    lines.append("MOTOR CONFIGURATION REPORT")
    lines.append("=" * 72)

    for motor in MOTOR_NAMES:
        cfg = configs[motor]
        lines.append(f"\n  {motor}")
        lines.append(f"  {'─' * 40}")
        for reg, mem, desc, expected in CONFIG_REGISTERS:
            val = cfg.get(reg, "?")
            if isinstance(val, int):
                if reg == "Status":
                    flags = decode_status(val)
                    flagstr = ", ".join(flags)
                    warn = "  ⚠ OVERLOAD PROTECTION ACTIVE" if "OVERLOAD" in flags else ""
                    lines.append(f"    {reg:<26} {val:>5}   [{flagstr}]{warn}")
                    if "OVERLOAD" in flags:
                        issues.append(f"{motor}: overload flag set — send new Goal_Position to clear")
                elif expected is not None and val != expected:
                    lines.append(f"    {reg:<26} {val:>5}   ← expected {expected}  ({desc})")
                    if reg == "Max_Torque_Limit" and val < 600:
                        issues.append(f"{motor}: Max_Torque_Limit={val} is LOW — use --torque 1000 to restore")
                    elif reg == "P_Coefficient" and val < 32:
                        issues.append(f"{motor}: P_Coefficient={val} (lerobot halves it to 16 on connect) — use --fix-pgain")
                else:
                    lines.append(f"    {reg:<26} {val:>5}   ({desc})")
            else:
                lines.append(f"    {reg:<26} {val}")

    lines.append("\n" + "=" * 72)
    if issues:
        lines.append("ISSUES DETECTED:")
        for i in issues:
            lines.append(f"  ⚠  {i}")
    else:
        lines.append("No configuration issues detected.")
    lines.append("=" * 72)

    text = "\n".join(lines)
    print(text)
    if file:
        file.write(text + "\n")
    return issues


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_log_dir() -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = LOG_DIR / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


CSV_FIELDS = [
    "time_s", "motor",
    "position_deg", "goal_position_deg",
    "position_error_deg",
    "load_pct", "current_mA", "voltage_v", "temperature_c",
    "moving", "status_flags",
]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--torque",     type=int, default=None,
                        help="Set Max_Torque_Limit (0-1000) for ALL motors. Writes to EEPROM.")
    parser.add_argument("--fix-pgain",  action="store_true",
                        help="Restore P_Coefficient to factory default (32) for all motors.")
    parser.add_argument("--clear-overload", action="store_true",
                        help="Send Goal_Position=Present_Position to clear overload flags.")
    args = parser.parse_args()

    # ── Connect ───────────────────────────────────────────────────────────────
    robot = SOFollower(SOFollowerRobotConfig(port=PORT, id=ROBOT_ID))
    robot.connect()
    bus = robot.bus

    # ── Log dir ───────────────────────────────────────────────────────────────
    log_dir = setup_log_dir()
    summary_path = log_dir / "summary.txt"
    csv_path     = log_dir / "stream.csv"
    print(f"\nLogging to: {log_dir}")

    with open(summary_path, "w") as summary_f:
        # ── Config report ─────────────────────────────────────────────────────
        summary_f.write(f"Run: {datetime.now().isoformat()}\n")
        summary_f.write(f"Port: {PORT}  Robot ID: {ROBOT_ID}\n\n")

        print("\nReading configuration registers...")
        configs = {name: read_config(bus, name) for name in MOTOR_NAMES}
        issues  = print_config_report(configs, file=summary_f)

        # ── Apply fixes if requested ──────────────────────────────────────────
        if args.torque is not None:
            print(f"\nSetting Max_Torque_Limit → {args.torque} ({args.torque/10:.0f}%) for all motors...")
            for name in MOTOR_NAMES:
                bus.write("Max_Torque_Limit", name, args.torque, normalize=False)
            print("Done. Power-cycle the arm for EEPROM to reload into RAM Torque_Limit.")
            summary_f.write(f"\nACTION: Max_Torque_Limit set to {args.torque} for all motors.\n")

        if args.fix_pgain:
            print("\nRestoring P_Coefficient → 32 for all motors...")
            for name in MOTOR_NAMES:
                bus.write("P_Coefficient", name, 32, normalize=False)
            print("Done.")
            summary_f.write("\nACTION: P_Coefficient restored to 32 for all motors.\n")

        if args.clear_overload:
            print("\nClearing overload flags (sending hold commands)...")
            obs = robot.get_observation()
            for name in MOTOR_NAMES:
                pos = obs.get(f"{name}.pos", 0.0)
                bus.write("Goal_Position", name, int(pos), normalize=False)
            print("Done.")
            summary_f.write("\nACTION: Goal_Position reset to clear overload flags.\n")

    # ── Rerun ─────────────────────────────────────────────────────────────────
    rr.init("so101_motor_diagnostics")
    rr.connect_grpc()

    print(f"\nStreaming at {RATE_HZ} Hz → Rerun + {csv_path.name}")
    print("Move the arm or command joints. Watch 'load_pct' and 'current_mA'.")
    print("Ctrl-C to stop.\n")

    dt = 1.0 / RATE_HZ
    t0 = time.perf_counter()

    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        try:
            while True:
                loop_start = time.perf_counter()
                t_rel = loop_start - t0
                rr.set_time("time", timestamp=t_rel)

                obs = robot.get_observation()
                console_parts = []

                for name in MOTOR_NAMES:
                    pos_deg = obs.get(f"{name}.pos", 0.0)

                    try:
                        raw_load   = bus.read("Present_Load",        name, normalize=False)
                        raw_temp   = bus.read("Present_Temperature",  name, normalize=False)
                        raw_volt   = bus.read("Present_Voltage",      name, normalize=False)
                        raw_cur    = bus.read("Present_Current",      name, normalize=False)
                        raw_goal   = bus.read("Goal_Position",        name, normalize=False)
                        raw_status = bus.read("Status",               name, normalize=False)
                        raw_moving = bus.read("Moving",               name, normalize=False)
                    except Exception as e:
                        print(f"  Read error {name}: {e}")
                        continue

                    load_pct   = decode_load(raw_load)
                    temp_c     = float(raw_temp)
                    volt_v     = decode_voltage(raw_volt)
                    cur_mA     = decode_current_mA(raw_cur)
                    status_str = "|".join(decode_status(raw_status))
                    # Goal_Position is raw ticks — convert approximately to degrees
                    # (calibration maps 0-4095 ticks to the joint range; rough estimate only)
                    goal_deg   = raw_goal * (360.0 / 4096.0)
                    pos_err    = pos_deg - goal_deg

                    # ── Rerun ─────────────────────────────────────────────────
                    base = f"motors/{name}"
                    rr.log(f"{base}/position_deg",  rr.Scalars(pos_deg))
                    rr.log(f"{base}/load_pct",       rr.Scalars(load_pct))
                    rr.log(f"{base}/temperature_c",  rr.Scalars(temp_c))
                    rr.log(f"{base}/voltage_v",      rr.Scalars(volt_v))
                    rr.log(f"{base}/current_mA",     rr.Scalars(cur_mA))
                    rr.log(f"{base}/moving",         rr.Scalars(float(raw_moving)))
                    if raw_status != 0:
                        rr.log(f"{base}/STATUS_FAULT", rr.Scalars(float(raw_status)))

                    # ── CSV ───────────────────────────────────────────────────
                    writer.writerow({
                        "time_s":           f"{t_rel:.3f}",
                        "motor":            name,
                        "position_deg":     f"{pos_deg:.2f}",
                        "goal_position_deg":f"{goal_deg:.2f}",
                        "position_error_deg":f"{pos_err:.2f}",
                        "load_pct":         f"{load_pct:.1f}",
                        "current_mA":       f"{cur_mA:.1f}",
                        "voltage_v":        f"{volt_v:.2f}",
                        "temperature_c":    f"{temp_c:.0f}",
                        "moving":           raw_moving,
                        "status_flags":     status_str,
                    })
                    csv_f.flush()

                    overload_warn = " ⚠ OVERLOAD" if "OVERLOAD" in status_str else ""
                    console_parts.append(
                        f"{name[:6]}:{load_pct:+5.0f}% {cur_mA:5.0f}mA{overload_warn}"
                    )

                print(f"\r  t={t_rel:6.1f}s  {'  |  '.join(console_parts)}", end="", flush=True)

                elapsed = time.perf_counter() - loop_start
                time.sleep(max(dt - elapsed, 0.0))

        except KeyboardInterrupt:
            print(f"\n\nStopped. Logs saved to {log_dir}")
        finally:
            robot.disconnect()


if __name__ == "__main__":
    main()
