#!/usr/bin/env python
"""
SO-101 interactive joint commander + current/load logger.

Gamepad drives the arm while per-motor current, load, voltage, and status
stream to a pygame panel, Rerun, and a CSV file.

Joystick mapping: same as teleop_gamepad.py.
Text commands (type in terminal alongside streaming):
  elbow_flex 45        jump to exact angle
  shoulder_lift -30    jump to exact angle
  all 0                all joints to 0
  hold                 resend current positions (clears overload flags)
  torque off / on      limp / hold
  help                 show command list
  q / quit             exit

Usage:
    python command_log.py
    python command_log.py --no-rerun
    python command_log.py --rate 20
"""

import argparse
import csv
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import os
os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

import pygame
import rerun as rr

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

from gamepad_utils import (
    MOTOR_NAMES, JOINT_LIMITS,
    detect_profile, get_joint_deltas, apply_deltas, is_neutral,
    graceful_shutdown, DeltaSmoother,
    BLACK, WHITE, GRAY, GREEN, RED, YELLOW, CYAN, BLUE, ORANGE,
    draw_stick, draw_button, draw_controller,
)

PORT     = "/dev/ttyACM0"
ROBOT_ID = "so101"
LOG_DIR  = Path("/home/javad/workspace/lerobot_all/outputs/logs")

# ── Decoders ──────────────────────────────────────────────────────────────────

def decode_load(raw: int) -> float:
    mag = (raw & 0x3FF) / 10.0
    return -mag if (raw >> 10) & 1 else mag

def decode_current_mA(raw: int) -> float:
    return raw * 6.5

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

# ── Background input thread ────────────────────────────────────────────────────

def _input_thread(q: queue.Queue):
    while True:
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            break
        line = line.strip()
        if line:
            q.put(line)

# ── Text command processing ────────────────────────────────────────────────────

HELP = (
    "Text commands:\n"
    "  <joint> <deg>   move one joint  (e.g. 'elbow_flex 45')\n"
    "  all <deg>       move all joints to same angle\n"
    "  hold            resend current positions (clears overload)\n"
    "  torque off/on   limp / hold\n"
    "  q / quit        exit\n"
    f"  joints: {', '.join(MOTOR_NAMES)}"
)


def process_command(cmd: str, goal_pos: dict, robot, bus) -> bool:
    """Returns True if should exit."""
    parts = cmd.lower().split()
    if not parts:
        return False

    if parts[0] in ("q", "quit", "exit"):
        return True

    if parts[0] in ("?", "help"):
        print(HELP)

    elif parts[0] == "hold":
        obs = robot.get_observation()
        for n in MOTOR_NAMES:
            goal_pos[n] = obs.get(f"{n}.pos", goal_pos[n])
        robot.send_action({f"{n}.pos": goal_pos[n] for n in MOTOR_NAMES})
        print("  Holding current positions.")

    elif parts[0] == "torque":
        if len(parts) < 2:
            print("  Usage: torque off|on")
        elif parts[1] == "off":
            bus.disable_torque()
            print("  Torque DISABLED.")
        elif parts[1] == "on":
            bus.enable_torque()
            robot.send_action({f"{n}.pos": goal_pos[n] for n in MOTOR_NAMES})
            print("  Torque ENABLED.")

    elif parts[0] == "all":
        if len(parts) < 2:
            print("  Usage: all <deg>")
        else:
            try:
                deg = float(parts[1])
                for n in MOTOR_NAMES:
                    lo, hi = JOINT_LIMITS[n]
                    goal_pos[n] = max(lo, min(hi, deg))
                robot.send_action({f"{n}.pos": goal_pos[n] for n in MOTOR_NAMES})
                print(f"  All joints → {deg:.1f}° (clipped to limits)")
            except ValueError:
                print(f"  Bad angle: {parts[1]}")

    elif parts[0] in MOTOR_NAMES:
        name = parts[0]
        if len(parts) < 2:
            print(f"  Usage: {name} <deg>")
        else:
            try:
                deg = float(parts[1])
                lo, hi = JOINT_LIMITS[name]
                goal_pos[name] = max(lo, min(hi, deg))
                robot.send_action({f"{n}.pos": goal_pos[n] for n in MOTOR_NAMES})
                print(f"  {name} → {goal_pos[name]:.1f}°")
            except ValueError:
                print(f"  Bad angle: {parts[1]}")
    else:
        print(f"  Unknown: '{cmd}'. Type 'help' for commands.")

    return False

# ── Pygame panel ───────────────────────────────────────────────────────────────
#
# Layout (760 × 380):
#   Left  (x 0..300): controller visualization — sticks + shoulder buttons
#   Right (x 310..760): 6 motor rows — position, load bar, current bar, status

WIN_W, WIN_H = 760, 380
MAX_CURRENT_MA = 1500.0   # full-scale for the current bar


def draw_motor_row(surf, y: int, name: str,
                   pos_deg: float, goal_deg: float,
                   load_pct: float, cur_mA: float,
                   volt_v: float, temp_c: float,
                   status_flags: list[str]):
    font = pygame.font.SysFont("monospace", 12)
    x0 = 315
    row_h = 56

    overload = "OVERLOAD" in status_flags
    fault = len(status_flags) > 0 and status_flags != ["OK"]
    name_color = RED if overload else (YELLOW if fault else WHITE)

    # Motor name + pose
    surf.blit(font.render(f"{name:<14}", True, name_color), (x0, y + 2))
    surf.blit(font.render(f"{pos_deg:+6.1f}° → {goal_deg:+6.1f}°", True, GRAY), (x0 + 110, y + 2))

    # Load bar (centred on midpoint; negative = left of centre, positive = right)
    bar_x = x0
    bar_y = y + 20
    bar_w = 220
    bar_h = 10
    mid = bar_x + bar_w // 2
    pygame.draw.rect(surf, (40, 40, 40), (bar_x, bar_y, bar_w, bar_h))
    pygame.draw.line(surf, GRAY, (mid, bar_y), (mid, bar_y + bar_h), 1)
    fill = int(abs(load_pct) / 100.0 * (bar_w // 2))
    fill = min(fill, bar_w // 2)
    load_color = RED if abs(load_pct) > 80 else (YELLOW if abs(load_pct) > 50 else GREEN)
    if load_pct >= 0:
        pygame.draw.rect(surf, load_color, (mid, bar_y, fill, bar_h))
    else:
        pygame.draw.rect(surf, load_color, (mid - fill, bar_y, fill, bar_h))
    surf.blit(font.render(f"load {load_pct:+5.0f}%", True, GRAY), (bar_x + bar_w + 6, bar_y - 1))

    # Current bar
    cur_y = bar_y + 16
    pygame.draw.rect(surf, (40, 40, 40), (bar_x, cur_y, bar_w, bar_h))
    fill_cur = int(min(cur_mA / MAX_CURRENT_MA, 1.0) * bar_w)
    cur_color = RED if cur_mA > 1000 else (YELLOW if cur_mA > 500 else CYAN)
    pygame.draw.rect(surf, cur_color, (bar_x, cur_y, fill_cur, bar_h))
    surf.blit(font.render(f"{cur_mA:5.0f}mA", True, GRAY), (bar_x + bar_w + 6, cur_y - 1))

    # Status + voltage/temp (right column)
    status_str = "|".join(status_flags)
    status_color = RED if fault else GREEN
    surf.blit(font.render(status_str, True, status_color), (x0 + 355, y + 2))
    surf.blit(font.render(f"{volt_v:.1f}V  {temp_c:.0f}°C", True, GRAY), (x0 + 355, y + 18))

    # Row separator
    pygame.draw.line(surf, (35, 35, 35), (x0, y + row_h - 2), (WIN_W - 5, y + row_h - 2), 1)


def draw_panel(surf, joystick, profile: dict, motor_data: dict, goal_pos: dict):
    surf.fill(BLACK)

    if joystick is None:
        surf.blit(pygame.font.SysFont("monospace", 16).render(
            "No gamepad", True, RED), (80, 180))
    else:
        draw_controller(surf, joystick, profile,
                        stick_left_xy=(80, 130), stick_right_xy=(210, 130),
                        shoulder_col_x=290, face_center=(265, 100))

    # Divider
    pygame.draw.line(surf, GRAY, (308, 0), (308, WIN_H), 1)

    # Column headers
    font = pygame.font.SysFont("monospace", 11)
    surf.blit(font.render("MOTOR           POSITION → GOAL", True, CYAN), (315, 2))
    surf.blit(font.render("LOAD / CURRENT / V / T / STATUS", True, CYAN), (315 + 355, 2))

    for j, name in enumerate(MOTOR_NAMES):
        d = motor_data.get(name, {})
        draw_motor_row(
            surf, y=18 + j * 58, name=name,
            pos_deg=d.get("pos_deg", 0.0),
            goal_deg=goal_pos.get(name, 0.0),
            load_pct=d.get("load_pct", 0.0),
            cur_mA=d.get("cur_mA", 0.0),
            volt_v=d.get("volt_v", 0.0),
            temp_c=d.get("temp_c", 0.0),
            status_flags=d.get("status_flags", ["?"]),
        )

# ── CSV fields ────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "time_s", "motor",
    "position_deg", "goal_deg",
    "load_pct", "current_mA", "voltage_v", "temperature_c",
    "moving", "status_flags",
]

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate",     type=float, default=10,
                        help="Polling rate in Hz (default: 10)")
    parser.add_argument("--no-rerun", action="store_true",
                        help="Skip Rerun viewer")
    args = parser.parse_args()

    # ── Logging ───────────────────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir  = LOG_DIR / ts
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "command_log.csv"
    print(f"Logging to: {log_dir}")

    # ── Connect robot ─────────────────────────────────────────────────────────
    robot = SOFollower(SOFollowerRobotConfig(port=PORT, id=ROBOT_ID))
    robot.connect()
    bus = robot.bus

    obs      = robot.get_observation()
    goal_pos = {n: obs.get(f"{n}.pos", 0.0) for n in MOTOR_NAMES}

    # ── Rerun ─────────────────────────────────────────────────────────────────
    if not args.no_rerun:
        rr.init("so101_command_log")
        rr.connect_grpc()

    # ── Pygame + joystick ─────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("SO-101 Command Log")

    joystick = None
    profile  = None
    smoother = DeltaSmoother(alpha=0.45)
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        profile = detect_profile(joystick)
        print(f"  Joystick ready: {joystick.get_name()}")
    else:
        print("  No joystick detected — text commands only.")

    # ── Background text input ─────────────────────────────────────────────────
    cmd_queue: queue.Queue = queue.Queue()
    threading.Thread(target=_input_thread, args=(cmd_queue,), daemon=True).start()

    print(f"\nStreaming at {args.rate} Hz → pygame panel + {csv_path.name}")
    print("Type 'help' for text commands. Ctrl-C to stop.\n")

    dt          = 1.0 / args.rate
    t0          = time.perf_counter()
    motor_data   = {}                        # name → latest decoded readings
    error_counts = {n: 0 for n in MOTOR_NAMES}  # consecutive read-error counter
    should_exit  = False

    with open(csv_path, "w", newline="") as csv_f:
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        try:
            while not should_exit:
                loop_start = time.perf_counter()
                t_rel = loop_start - t0

                # ── pygame events ──────────────────────────────────────────────
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        should_exit = True

                if should_exit:
                    break

                # ── text commands ──────────────────────────────────────────────
                while not cmd_queue.empty():
                    cmd = cmd_queue.get_nowait()
                    if process_command(cmd, goal_pos, robot, bus):
                        should_exit = True
                        break

                if should_exit:
                    break

                # ── joystick → joint targets ───────────────────────────────────
                if joystick is not None:
                    deltas = smoother(get_joint_deltas(joystick, profile))
                    if not is_neutral(deltas):
                        goal_pos = apply_deltas(goal_pos, deltas)
                        robot.send_action({f"{n}.pos": goal_pos[n] for n in MOTOR_NAMES})

                # ── read sensors ───────────────────────────────────────────────
                if not args.no_rerun:
                    rr.set_time("time", timestamp=t_rel)

                obs = robot.get_observation()

                for name in MOTOR_NAMES:
                    pos_deg = obs.get(f"{name}.pos", 0.0)
                    try:
                        raw_load   = bus.read("Present_Load",        name, normalize=False)
                        raw_temp   = bus.read("Present_Temperature",  name, normalize=False)
                        raw_volt   = bus.read("Present_Voltage",      name, normalize=False)
                        raw_cur    = bus.read("Present_Current",      name, normalize=False)
                        raw_status = bus.read("Status",               name, normalize=False)
                        raw_moving = bus.read("Moving",               name, normalize=False)
                    except Exception as e:
                        error_counts[name] += 1
                        if error_counts[name] == 1:
                            print(f"  Read error {name}: {e}")
                        elif error_counts[name] % 30 == 0:
                            print(f"  {name} still in error ({error_counts[name]} polls) — sending hold to clear overload")
                            try:
                                robot.send_action({f"{n}.pos": goal_pos[n] for n in MOTOR_NAMES})
                            except Exception:
                                pass
                        continue

                    error_counts[name] = 0  # reset on successful read

                    load_pct    = decode_load(raw_load)
                    temp_c      = float(raw_temp)
                    volt_v      = decode_voltage(raw_volt)
                    cur_mA      = decode_current_mA(raw_cur)
                    status_flags = decode_status(raw_status)
                    status_str  = "|".join(status_flags)

                    motor_data[name] = dict(
                        pos_deg=pos_deg, load_pct=load_pct,
                        cur_mA=cur_mA, volt_v=volt_v, temp_c=temp_c,
                        status_flags=status_flags,
                    )

                    if not args.no_rerun:
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

                    writer.writerow({
                        "time_s":        f"{t_rel:.3f}",
                        "motor":         name,
                        "position_deg":  f"{pos_deg:.2f}",
                        "goal_deg":      f"{goal_pos[name]:.2f}",
                        "load_pct":      f"{load_pct:.1f}",
                        "current_mA":    f"{cur_mA:.1f}",
                        "voltage_v":     f"{volt_v:.2f}",
                        "temperature_c": f"{temp_c:.0f}",
                        "moving":        raw_moving,
                        "status_flags":  status_str,
                    })
                    csv_f.flush()

                # ── draw panel ─────────────────────────────────────────────────
                draw_panel(screen, joystick, profile, motor_data, goal_pos)
                pygame.display.flip()

                elapsed = time.perf_counter() - loop_start
                time.sleep(max(dt - elapsed, 0.0))

        except KeyboardInterrupt:
            pass
        finally:
            graceful_shutdown(robot)
            try:
                robot.disconnect()
            except Exception as e:
                # Overloaded motor flags error on the Torque_Enable write,
                # but the write itself succeeds — arm is safe to release.
                print(f"  Disconnect warning (torque disable likely succeeded): {e}")
            pygame.quit()
            print(f"Logs saved to {log_dir}")


if __name__ == "__main__":
    main()
