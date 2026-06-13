#!/usr/bin/env python
"""
Joystick / gamepad debugger for the SO-101 control mapping.

This is intentionally independent of the robot, MuJoCo, Rerun, and LeRobot
datasets. It verifies whether pygame/SDL can see the controller, then shows raw
axes/buttons/hats plus the joint velocity commands produced by gamepad_utils.py.

Usage:
    python gamepad_debug.py
    python gamepad_debug.py --headless
"""

import argparse
import os
import time

os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

import pygame

from gamepad_utils import (
    BLACK,
    BLUE,
    CYAN,
    GRAY,
    GREEN,
    JOINT_LIMITS,
    MOTOR_NAMES,
    ORANGE,
    RED,
    WHITE,
    YELLOW,
    axis_deadzone,
    button_index,
    detect_profile,
    draw_controller,
    filtered_axis_value,
    get_joint_deltas,
    save_axis_calibration,
    save_layout_override,
)


WIN_W, WIN_H = 920, 700

NINTENDO_BUTTON_LABELS = {
    0: "B",
    1: "A",
    2: "Y",
    3: "X",
    4: "L",
    5: "R",
    6: "ZL",
    7: "ZR",
    8: "-",
    9: "+",
    10: "L3",
    11: "R3",
    12: "Home",
    13: "Cap",
}

GENERIC_BUTTON_LABELS = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "LB",
    5: "RB",
    6: "LT",
    7: "RT",
    8: "Back",
    9: "Start",
    10: "L3",
    11: "R3",
    12: "Home",
}

CALIBRATION_STEPS = [
    ("B", "press B"),
    ("A", "press A"),
    ("Y", "press Y"),
    ("X", "press X"),
    ("L", "press L / left bumper"),
    ("R", "press R / right bumper"),
    ("ZL", "press ZL / left trigger"),
    ("ZR", "press ZR / right trigger"),
    ("-", "press minus / select"),
    ("+", "press plus / start"),
    ("L3", "press left stick"),
    ("R3", "press right stick"),
    ("Home", "press Home"),
    ("Cap", "press Capture / screenshot"),
]


def open_joystick(device_index: int):
    joystick = pygame.joystick.Joystick(device_index)
    joystick.init()
    profile = detect_profile(joystick)
    print(
        f"Connected: name='{joystick.get_name()}', "
        f"instance={joystick.get_instance_id()}, "
        f"guid={getattr(joystick, 'get_guid', lambda: 'unknown')()}"
    )
    print(
        f"  axes={joystick.get_numaxes()} "
        f"buttons={joystick.get_numbuttons()} "
        f"hats={joystick.get_numhats()}"
    )
    return joystick, profile


def axis_values(joystick):
    return [joystick.get_axis(i) for i in range(joystick.get_numaxes())]


def button_values(joystick):
    return [joystick.get_button(i) for i in range(joystick.get_numbuttons())]


def hat_values(joystick):
    return [joystick.get_hat(i) for i in range(joystick.get_numhats())]


def button_label(joystick, index: int, profile: dict | None = None) -> str:
    if profile is not None:
        for label, mapped_index in profile.get("buttons", {}).items():
            if int(mapped_index) == index:
                return label
        for mapped_index, label in profile.get("face", {}).items():
            if int(mapped_index) == index:
                return label
        for label, mapped_index in profile.get("shoulder", {}).items():
            if int(mapped_index) == index:
                return label

    name = joystick.get_name().lower() if joystick is not None else ""
    labels = NINTENDO_BUTTON_LABELS
    if not any(token in name for token in ("pro controller", "nintendo", "switch")):
        labels = GENERIC_BUTTON_LABELS
    return labels.get(index, str(index))


def draw_bar(screen, font, x, y, width, label, value, color):
    pygame.draw.rect(screen, GRAY, (x, y, width, 12), 1)
    mid = x + width // 2
    pygame.draw.line(screen, GRAY, (mid, y), (mid, y + 12), 1)
    if value >= 0:
        bar_x = mid
        bar_w = int((width / 2 - 2) * min(value, 1.0))
    else:
        bar_w = int((width / 2 - 2) * min(-value, 1.0))
        bar_x = mid - bar_w
    pygame.draw.rect(screen, color, (bar_x, y + 2, bar_w, 8))
    screen.blit(font.render(f"{label:>6s} {value:+.3f}", True, WHITE), (x + width + 10, y - 3))


def draw_axis_row(screen, font, joystick, profile, x, y, width, axis_index, raw_value, axis_peaks):
    filtered = filtered_axis_value(joystick, profile, axis_index)
    deadzone = axis_deadzone(profile, axis_index)
    color = GREEN if filtered else BLUE
    draw_bar(screen, font, x, y, width, f"axis {axis_index}", raw_value, color)
    screen.blit(
        font.render(
            f"filt {filtered:+.3f}  dz {deadzone:.2f}  peak {axis_peaks.get(axis_index, 0.0):.2f}",
            True,
            YELLOW if filtered else GRAY,
        ),
        (x + width + 118, y - 3),
    )


def draw_button_row(screen, font, joystick, profile, x, y, buttons, cols=7):
    for i, pressed in enumerate(buttons):
        bx = x + (i % cols) * 66
        by = y + (i // cols) * 34
        pygame.draw.rect(screen, GREEN if pressed else GRAY, (bx, by, 54, 24), border_radius=3)
        text_color = BLACK if pressed else WHITE
        label = f"{i}:{button_label(joystick, i, profile)}"
        screen.blit(font.render(label[:7], True, text_color), (bx + 4, by + 5))


def draw_joint_velocities(screen, font, x, y, velocities):
    screen.blit(font.render("JOINT COMMANDS (deg/s, gripper units/s)", True, CYAN), (x, y))
    for row, name in enumerate(MOTOR_NAMES):
        value = velocities.get(name, 0.0)
        lo, hi = JOINT_LIMITS[name]
        full_scale = max(abs(lo), abs(hi), 1)
        bar_w = int(90 * min(abs(value) / full_scale, 1.0))
        bx = x + 210
        by = y + 24 + row * 22
        pygame.draw.rect(screen, GRAY, (bx, by, 180, 12), 1)
        mid = bx + 90
        pygame.draw.line(screen, GRAY, (mid, by), (mid, by + 12), 1)
        if value >= 0:
            pygame.draw.rect(screen, ORANGE, (mid, by + 2, bar_w, 8))
        else:
            pygame.draw.rect(screen, ORANGE, (mid - bar_w, by + 2, bar_w, 8))
        screen.blit(font.render(f"{name:14s} {value:+7.2f}", True, WHITE), (x, by - 3))


def draw_event_log(screen, font, x, y, event_log):
    screen.blit(font.render("EVENT LOG", True, CYAN), (x, y))
    if not event_log:
        screen.blit(font.render("waiting for input...", True, GRAY), (x, y + 24))
        return
    for row, line in enumerate(event_log[-10:]):
        screen.blit(font.render(line, True, YELLOW), (x, y + 24 + row * 18))


def format_event(event, joystick, profile) -> str | None:
    if event.type == pygame.JOYAXISMOTION:
        return f"axis {event.axis} = {event.value:+.3f}"
    if event.type == pygame.JOYBUTTONDOWN:
        return f"button {event.button}:{button_label(joystick, event.button, profile)} down"
    if event.type == pygame.JOYBUTTONUP:
        return f"button {event.button}:{button_label(joystick, event.button, profile)} up"
    if event.type == pygame.JOYHATMOTION:
        return f"hat {event.hat} = {event.value}"
    if event.type == pygame.JOYDEVICEADDED:
        return f"device added index={event.device_index}"
    if event.type == pygame.JOYDEVICEREMOVED:
        return f"device removed instance={event.instance_id}"
    return None


def draw_window(screen, joystick, profile, event_log, axis_peaks):
    screen.fill(BLACK)
    font = pygame.font.SysFont("monospace", 13)
    title = pygame.font.SysFont("monospace", 16, bold=True)

    if joystick is None:
        screen.blit(title.render("No gamepad detected", True, RED), (28, 28))
        screen.blit(font.render("Plug in / pair the controller. Press q or Esc to quit.", True, WHITE), (28, 58))
        pygame.display.flip()
        return

    axes = axis_values(joystick)
    buttons = button_values(joystick)
    hats = hat_values(joystick)
    velocities = get_joint_deltas(joystick, profile, dt=1.0)

    screen.blit(title.render(joystick.get_name(), True, CYAN), (24, 18))
    screen.blit(
        font.render(
            f"instance={joystick.get_instance_id()}  axes={len(axes)}  "
            f"buttons={len(buttons)}  hats={len(hats)}",
            True,
            WHITE,
        ),
        (24, 44),
    )

    draw_controller(
        screen,
        joystick,
        profile,
        stick_left_xy=(105, 160),
        stick_right_xy=(300, 160),
        shoulder_col_x=515,
        face_center=(625, 142),
    )

    screen.blit(font.render("RAW AXES / TRIGGERS", True, CYAN), (24, 270))
    for i, value in enumerate(axes):
        draw_axis_row(screen, font, joystick, profile, 24, 298 + i * 24, 210, i, value, axis_peaks)

    screen.blit(font.render("ALL RAW BUTTONS", True, CYAN), (410, 270))
    draw_button_row(screen, font, joystick, profile, 410, 304, buttons)

    if hats:
        screen.blit(font.render(f"HATS / DPAD {hats}", True, YELLOW), (410, 414))

    draw_event_log(screen, font, 430, 444, event_log)
    draw_joint_velocities(screen, font, 24, 510, velocities)
    screen.blit(font.render("Nintendo expected: 0=B 1=A 2=Y 3=X 4=L 5=R 6=ZL 7=ZR 8=- 9=+", True, WHITE), (24, 652))
    screen.blit(font.render("q/Esc quits", True, YELLOW), (800, 652))
    pygame.display.flip()


def draw_calibration_prompt(screen, joystick, profile, step_index, label, prompt, assigned, waiting_release=False):
    draw_window(screen, joystick, profile, [], {})
    overlay = pygame.Surface((WIN_W, 112))
    overlay.set_alpha(235)
    overlay.fill((20, 20, 20))
    screen.blit(overlay, (0, 0))
    font = pygame.font.SysFont("monospace", 15)
    title = pygame.font.SysFont("monospace", 18, bold=True)
    screen.blit(title.render("Calibration", True, CYAN), (24, 16))
    if waiting_release:
        screen.blit(font.render("Release all buttons before the next prompt.", True, WHITE), (24, 44))
    else:
        screen.blit(font.render(f"{step_index + 1}/{len(CALIBRATION_STEPS)}: {prompt}", True, WHITE), (24, 44))
    screen.blit(font.render("Press the requested control. Press s to skip, q/Esc to stop.", True, YELLOW), (24, 70))
    if assigned:
        tail = "  ".join(f"{k}={v}" for k, v in assigned.items())
        screen.blit(font.render(tail[-100:], True, GREEN), (24, 92))
    pygame.display.flip()


def pressed_buttons(joystick):
    return [i for i in range(joystick.get_numbuttons()) if joystick.get_button(i)]


def wait_for_all_released(screen, joystick, profile, step_index, label, prompt, assigned):
    while pressed_buttons(joystick):
        if screen is not None:
            draw_calibration_prompt(screen, joystick, profile, step_index, label, prompt, assigned, waiting_release=True)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                raise KeyboardInterrupt
        time.sleep(0.03)
    pygame.event.clear()


def wait_for_button(screen, joystick, profile, step_index, label, prompt, assigned):
    print(f"{step_index + 1:02d}/{len(CALIBRATION_STEPS)} {prompt}")
    wait_for_all_released(screen, joystick, profile, step_index, label, prompt, assigned)
    while True:
        if screen is not None:
            draw_calibration_prompt(screen, joystick, profile, step_index, label, prompt, assigned)
        event = pygame.event.wait()
        if event.type == pygame.QUIT:
            raise KeyboardInterrupt
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                raise KeyboardInterrupt
            if event.key == pygame.K_s:
                print(f"  skipped {label}")
                return None
        if event.type == pygame.JOYBUTTONDOWN:
            duplicate = [k for k, v in assigned.items() if v == event.button]
            if duplicate:
                print(f"  ignored duplicate button {event.button}; already assigned to {', '.join(duplicate)}")
                wait_for_all_released(screen, joystick, profile, step_index, label, prompt, assigned)
                continue
            print(f"  {label} -> button {event.button}")
            idx = int(event.button)
            wait_for_all_released(screen, joystick, profile, step_index, label, prompt, assigned)
            return idx


def run_calibration(screen, joystick, profile):
    if joystick is None:
        print("No gamepad detected; cannot calibrate.")
        return profile

    print("\nCalibration records the physical button layout reported by SDL.")
    print("Use the pygame window for skip/stop keys. Ctrl-C also stops.\n")
    assigned = {}
    try:
        for step_index, (label, prompt) in enumerate(CALIBRATION_STEPS):
            idx = wait_for_button(screen, joystick, profile, step_index, label, prompt, assigned)
            if idx is None:
                continue
            assigned[label] = idx
    except KeyboardInterrupt:
        print("\nCalibration stopped.")

    if not assigned:
        print("No buttons recorded.")
        return profile

    path = save_layout_override(joystick, assigned)
    print(f"\nSaved layout override: {path}")

    fresh_profile = detect_profile(joystick)
    print("\nDetected layout:")
    for label, _prompt in CALIBRATION_STEPS:
        idx = assigned.get(label)
        expected = button_index(profile, label)
        if idx is None:
            print(f"  {label:4s}: skipped")
        elif expected is None:
            print(f"  {label:4s}: button {idx}")
        elif expected != idx:
            print(f"  {label:4s}: button {idx}  (was {expected})")
        else:
            print(f"  {label:4s}: button {idx}")
    print("")
    return fresh_profile


def run_axis_calibration(joystick, duration_s=2.0):
    if joystick is None:
        print("No gamepad detected; cannot calibrate axes.")
        return None

    print("\nAxis calibration: release sticks/triggers and keep the controller still.")
    print(f"Sampling rest/noise for {duration_s:.1f}s...\n")
    samples = {i: [] for i in range(joystick.get_numaxes())}
    end = time.monotonic() + duration_s
    while time.monotonic() < end:
        pygame.event.pump()
        for i in range(joystick.get_numaxes()):
            samples[i].append(joystick.get_axis(i))
        time.sleep(0.01)

    calibration = {}
    for axis, values in samples.items():
        if not values:
            continue
        rest = sum(values) / len(values)
        jitter = max(abs(v - rest) for v in values)
        threshold = max(0.04, min(0.12, jitter * 4.0 + 0.02))
        calibration[axis] = {
            "rest": rest,
            "threshold": threshold,
            "jitter": jitter,
        }
        print(
            f"  axis {axis}: rest={rest:+.3f} "
            f"jitter={jitter:.3f} threshold={threshold:.3f}"
        )

    path = save_axis_calibration(joystick, calibration)
    print(f"\nSaved axis calibration: {path}\n")
    return detect_profile(joystick)


def print_snapshot(joystick, profile, axis_peaks):
    axes = " ".join(f"{v:+.2f}" for v in axis_values(joystick))
    filtered_axes = " ".join(
        f"{filtered_axis_value(joystick, profile, i):+.2f}"
        for i in range(joystick.get_numaxes())
    )
    peaks = " ".join(f"{axis_peaks.get(i, 0.0):.2f}" for i in range(joystick.get_numaxes()))
    buttons = "".join(str(v) for v in button_values(joystick))
    hats = " ".join(str(v) for v in hat_values(joystick))
    velocities = get_joint_deltas(joystick, profile, dt=1.0)
    cmd = " ".join(f"{name}={velocities[name]:+.1f}" for name in MOTOR_NAMES)
    print(f"axes=[{axes}] filt=[{filtered_axes}] peak=[{peaks}] buttons={buttons} hats=[{hats}] cmd=[{cmd}]")


def main():
    parser = argparse.ArgumentParser(description="Debug pygame joystick detection and SO-101 mapping.")
    parser.add_argument("--headless", action="store_true", help="Print to terminal without opening a window.")
    parser.add_argument("--calibrate", action="store_true", help="Record physical button layout and save an override.")
    parser.add_argument("--axis-calibrate", action="store_true", help="Record analog axis rest/noise thresholds.")
    parser.add_argument("--rate", type=float, default=10.0, help="Terminal/window update rate in Hz.")
    args = parser.parse_args()
    if args.calibrate and args.headless:
        parser.error("--calibrate needs the pygame window; omit --headless.")

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    pygame.init()
    pygame.joystick.init()
    screen = None
    if not args.headless:
        screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("SO-101 Gamepad Debug")

    joystick = None
    profile = None
    event_log = []
    axis_peaks = {}
    if pygame.joystick.get_count() > 0:
        joystick, profile = open_joystick(0)
    else:
        print("No gamepad detected by pygame/SDL.")
        print("  Linux checks: ls -l /dev/input/js* /dev/input/event*")
        print("  Bluetooth checks: bluetoothctl devices; bluetoothctl info <MAC>")

    next_print = 0.0
    dt = 1.0 / max(args.rate, 0.1)

    if args.calibrate:
        if screen is None:
            screen = pygame.display.set_mode((WIN_W, WIN_H))
            pygame.display.set_caption("SO-101 Gamepad Calibration")
        profile = run_calibration(screen, joystick, profile)
    if args.axis_calibrate:
        profile = run_axis_calibration(joystick) or profile

    try:
        while True:
            for event in pygame.event.get():
                event_text = format_event(event, joystick, profile)
                if event_text:
                    event_log.append(event_text)
                    event_log = event_log[-20:]
                    if args.headless:
                        print(event_text)
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return
                if event.type == pygame.JOYDEVICEADDED and joystick is None:
                    joystick, profile = open_joystick(event.device_index)
                if event.type == pygame.JOYDEVICEREMOVED and joystick is not None:
                    if event.instance_id == joystick.get_instance_id():
                        print("Disconnected.")
                        joystick = None
                        profile = None

            if joystick is None and pygame.joystick.get_count() > 0:
                joystick, profile = open_joystick(0)

            if joystick is not None:
                for i, value in enumerate(axis_values(joystick)):
                    axis_peaks[i] = max(axis_peaks.get(i, 0.0), abs(value))

            now = time.monotonic()
            if screen is not None:
                draw_window(screen, joystick, profile, event_log, axis_peaks)
            elif joystick is not None and now >= next_print:
                print_snapshot(joystick, profile, axis_peaks)
                next_print = now + dt

            time.sleep(dt if screen is None else min(dt, 1.0 / 30.0))
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
