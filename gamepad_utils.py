"""
Shared gamepad / joint-control utilities for SO-101 scripts.

Provides constants, controller profiles, joystick delta calculation, and pygame drawing
helpers used by station.py and sim_collect.py. Run this file directly for the gamepad
debugger / calibrator (it absorbed the old gamepad_debug.py):

    python gamepad_utils.py                 # live axes/buttons + joint-command preview
    python gamepad_utils.py --headless      # terminal only
    python gamepad_utils.py --calibrate     # record the physical button layout
    python gamepad_utils.py --axis-calibrate
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import pygame

# ── Robot constants ────────────────────────────────────────────────────────────

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

JOINT_SPEED = {           # max speed at full stick, deg/s (gripper: units/s)
    "shoulder_pan":  60.0,
    "shoulder_lift": 50.0,
    "elbow_flex":    50.0,
    "wrist_flex":    35.0,
    "wrist_roll":    90.0,
    "gripper":       35.0,
}

# Gripper is driven by face buttons (ZR was failing — firing without a press).
# A opens, B closes. Change these labels to remap.
GRIPPER_OPEN_BTN = "A"
GRIPPER_CLOSE_BTN = "B"

JOINT_LIMITS = {
    "shoulder_pan":  (-90,  90),
    "shoulder_lift": (-90,  90),
    "elbow_flex":    (-90,  90),
    "wrist_flex":    (-80,  80),
    "wrist_roll":   (-150, 150),
    "gripper":       (  0, 100),
}

DEADZONE = 0.08
DEFAULT_AXIS_DEADZONES = {
    0: DEADZONE,  # left stick X
    1: DEADZONE,  # left stick Y
    2: DEADZONE,  # right stick X / Z on some SDL mappings
    3: DEADZONE,  # right stick Y / RZ on some SDL mappings
}

# ── Controller profiles ────────────────────────────────────────────────────────

CONTROLLER_PROFILES = {
    "nintendo": {
        "patterns": ["pro controller", "nintendo switch", "switch pro"],
        "face":     {0: "B", 1: "A", 2: "Y", 3: "X"},
        "shoulder": {"L": 4, "R": 5, "ZL": 6, "ZR": 7},
    },
    "xbox": {
        "patterns": ["xbox", "x-box", "microsoft"],
        "face":     {0: "A", 1: "B", 2: "X", 3: "Y"},
        "shoulder": {"L": 4, "R": 5, "ZL": 6, "ZR": 7},
    },
    "playstation": {
        "patterns": ["dualsense", "dualshock", "sony", "playstation"],
        "face":     {0: "X", 1: "O", 2: "□", 3: "△"},
        "shoulder": {"L": 4, "R": 5, "ZL": 6, "ZR": 7},
    },
    "generic": {
        "patterns": [],
        "face":     {0: "0", 1: "1", 2: "2", 3: "3"},
        "shoulder": {"L": 4, "R": 5, "ZL": 6, "ZR": 7},
    },
}


LAYOUT_DIR = Path(__file__).resolve().parent / "config"


def _safe_layout_id(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text).strip("_") or "gamepad"


def joystick_guid(joystick) -> str:
    try:
        guid = joystick.get_guid()
    except Exception:
        guid = ""
    return guid or "unknown"


def layout_path_for(joystick) -> Path:
    guid = joystick_guid(joystick)
    if guid != "unknown":
        return LAYOUT_DIR / f"{_safe_layout_id(guid)}.json"
    return LAYOUT_DIR / f"{_safe_layout_id(joystick.get_name())}.json"


def _load_layout_override(joystick) -> dict | None:
    path = layout_path_for(joystick)
    if not path.exists():
        return None
    try:
        with path.open() as f:
            layout = json.load(f)
    except Exception as exc:
        print(f"  Gamepad layout override unreadable ({path}: {exc}); ignoring.")
        return None
    if not isinstance(layout.get("buttons"), dict):
        print(f"  Gamepad layout override missing 'buttons' ({path}); ignoring.")
        return None
    buttons = {str(k): int(v) for k, v in layout["buttons"].items()}
    reverse = {}
    duplicates = []
    for label, idx in buttons.items():
        if idx in reverse:
            duplicates.append(f"{label}/{reverse[idx]}={idx}")
        reverse[idx] = label
    if duplicates:
        print(f"  Gamepad layout override has duplicate buttons ({', '.join(duplicates)}).")
        print("  Re-run: python gamepad_utils.py --calibrate")
    return layout


def _apply_layout_override(profile: dict, layout: dict) -> dict:
    mapped = copy.deepcopy(profile)
    buttons = {str(k): int(v) for k, v in layout["buttons"].items()}
    mapped["buttons"] = buttons
    if isinstance(layout.get("axis_deadzone"), dict):
        mapped["axis_deadzone"] = {
            int(axis): float(deadzone)
            for axis, deadzone in layout["axis_deadzone"].items()
        }
    if isinstance(layout.get("axis_calibration"), dict):
        mapped["axis_calibration"] = {
            int(axis): {
                str(k): float(v)
                for k, v in calib.items()
            }
            for axis, calib in layout["axis_calibration"].items()
            if isinstance(calib, dict)
        }

    face = {}
    for label in ("B", "A", "Y", "X"):
        if label in buttons:
            face[buttons[label]] = label
    if face:
        mapped["face"] = face

    shoulder = {}
    for label in ("L", "R", "ZL", "ZR"):
        if label in buttons:
            shoulder[label] = buttons[label]
    if shoulder:
        mapped["shoulder"] = {**mapped["shoulder"], **shoulder}

    return mapped


def detect_profile(joystick) -> dict:
    name = joystick.get_name().lower()
    selected_key = "generic"
    selected_profile = CONTROLLER_PROFILES["generic"]
    for key, profile in CONTROLLER_PROFILES.items():
        if any(p in name for p in profile["patterns"]):
            selected_key = key
            selected_profile = profile
            break

    override = _load_layout_override(joystick)
    if override is not None:
        mapped = _apply_layout_override(selected_profile, override)
        print(f"  Controller: '{joystick.get_name()}' → {selected_key} profile + layout override")
        print(f"  Layout: {layout_path_for(joystick)}")
        return mapped

    print(f"  Controller: '{joystick.get_name()}' → {selected_key} profile")
    return selected_profile


def save_layout_override(joystick, buttons: dict) -> Path:
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    path = layout_path_for(joystick)
    previous = {}
    if path.exists():
        try:
            with path.open() as f:
                previous = json.load(f)
        except Exception:
            previous = {}
    payload = {
        "name": joystick.get_name(),
        "guid": joystick_guid(joystick),
        "buttons": {str(k): int(v) for k, v in buttons.items()},
    }
    if isinstance(previous.get("axis_deadzone"), dict):
        payload["axis_deadzone"] = previous["axis_deadzone"]
    if isinstance(previous.get("axis_calibration"), dict):
        payload["axis_calibration"] = previous["axis_calibration"]
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def save_axis_calibration(joystick, axis_calibration: dict) -> Path:
    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    path = layout_path_for(joystick)
    payload = {}
    if path.exists():
        try:
            with path.open() as f:
                payload = json.load(f)
        except Exception:
            payload = {}
    payload.setdefault("name", joystick.get_name())
    payload.setdefault("guid", joystick_guid(joystick))
    payload.setdefault("buttons", {})
    payload["axis_calibration"] = {
        str(axis): {str(k): float(v) for k, v in calib.items()}
        for axis, calib in axis_calibration.items()
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def face_index(profile: dict, label: str) -> int | None:
    """Joystick button index for a face label ('A'/'B'/'X'/'Y'), or None.
    Looks in the override 'buttons' map first, then the profile's 'face'
    (which is keyed index→label)."""
    buttons = profile.get("buttons")
    if isinstance(buttons, dict) and label in buttons:
        return int(buttons[label])
    face = profile.get("face")
    if isinstance(face, dict):
        for idx, lbl in face.items():
            if lbl == label:
                return int(idx)
    return None


def button_index(profile: dict, label: str) -> int | None:
    buttons = profile.get("buttons")
    if isinstance(buttons, dict) and label in buttons:
        return int(buttons[label])
    if label in profile.get("shoulder", {}):
        return int(profile["shoulder"][label])
    for idx, face_label in profile.get("face", {}).items():
        if face_label == label:
            return int(idx)
    return None


def axis_deadzone(profile: dict, axis_index: int) -> float:
    calibration = profile.get("axis_calibration", {})
    if axis_index in calibration and "threshold" in calibration[axis_index]:
        return float(calibration[axis_index]["threshold"])
    if str(axis_index) in calibration and "threshold" in calibration[str(axis_index)]:
        return float(calibration[str(axis_index)]["threshold"])
    deadzones = profile.get("axis_deadzone", {})
    if axis_index in deadzones:
        return float(deadzones[axis_index])
    if str(axis_index) in deadzones:
        return float(deadzones[str(axis_index)])
    return DEFAULT_AXIS_DEADZONES.get(axis_index, DEADZONE)


def filtered_axis_value(joystick, profile: dict, axis_index: int) -> float:
    value = joystick.get_axis(axis_index)
    calibration = profile.get("axis_calibration", {})
    axis_cal = calibration.get(axis_index, calibration.get(str(axis_index), {}))
    rest = float(axis_cal.get("rest", 0.0)) if isinstance(axis_cal, dict) else 0.0
    centered = value - rest
    if abs(centered) <= axis_deadzone(profile, axis_index):
        return 0.0
    return max(-1.0, min(1.0, centered))


# ── Joint velocity helpers ─────────────────────────────────────────────────────

class ButtonDebouncer:
    """Require a button to read the same value for `stable` consecutive polls
    before reporting the change — kills single-frame contact chatter (e.g. a worn
    ZR firing spuriously). 3 polls ≈ 60 ms at the 50 Hz command tick, below
    perception but well above bounce."""

    def __init__(self, stable: int = 3):
        self.stable = stable
        self._state = {}   # index -> currently reported bool
        self._count = {}   # index -> consecutive polls the candidate has held

    def __call__(self, index: int, raw: bool) -> bool:
        reported = self._state.get(index, False)
        if raw == reported:
            self._count[index] = 0
            return reported
        self._count[index] = self._count.get(index, 0) + 1
        if self._count[index] >= self.stable:
            self._state[index] = raw
            self._count[index] = 0
            return raw
        return reported


def get_joint_deltas(joystick, profile: dict, dt: float, debounce: "ButtonDebouncer | None" = None) -> dict:
    """Per-tick goal increments: expo stick × JOINT_SPEED (deg/s) × dt (tick length).
    dt is required — every caller states its own tick so speeds stay in deg/s.
    Pass a ButtonDebouncer to reject single-frame button chatter."""
    def axis(i):
        v = filtered_axis_value(joystick, profile, i)
        return v * abs(v)       # expo: fine control near centre, full speed at the edge

    def btn(i):
        try:    v = bool(joystick.get_button(i))
        except: return 0
        if debounce is not None:
            v = debounce(i, v)
        return int(v)

    def face_btn(label):
        idx = face_index(profile, label)
        return btn(idx) if idx is not None else 0

    lx = axis(0); ly = axis(1)
    rx = axis(2); ry = axis(3)
    sh = profile["shoulder"]
    gripper_open = face_btn(GRIPPER_OPEN_BTN)
    gripper_close = face_btn(GRIPPER_CLOSE_BTN)
    return {
        "shoulder_pan":  -lx * JOINT_SPEED["shoulder_pan"] * dt,
        "shoulder_lift":  ly * JOINT_SPEED["shoulder_lift"] * dt,
        "elbow_flex":    -ry * JOINT_SPEED["elbow_flex"] * dt,
        "wrist_roll":     rx * JOINT_SPEED["wrist_roll"] * dt,
        "wrist_flex":    (btn(sh["L"]) - btn(sh["R"])) * JOINT_SPEED["wrist_flex"] * dt,
        "gripper":       (gripper_open - gripper_close) * JOINT_SPEED["gripper"] * dt,
    }


def apply_deltas(joint_pos: dict, deltas: dict) -> dict:
    result = {}
    for name in MOTOR_NAMES:
        lo, hi = JOINT_LIMITS[name]
        result[name] = max(lo, min(hi, joint_pos[name] + deltas.get(name, 0.0)))
    return result


def is_neutral(deltas: dict) -> bool:
    return all(v == 0.0 for v in deltas.values())


class DeltaSmoother:
    """
    Low-pass filter on joystick velocity commands.

    Smooths both ramp-up (stick pushed) and ramp-down (stick released),
    giving a soft trapezoidal velocity profile without a separate motion
    planner. Lower alpha = more smoothing but more lag; higher = snappier.

    Suggested values:
      alpha=0.25  — very smooth, slight lag (good for shoulder/elbow)
      alpha=0.45  — moderate (good default)
      alpha=0.70  — near-instant, almost no smoothing
    """

    def __init__(self, alpha: float = 0.45):
        self.alpha   = alpha
        self._smooth = {n: 0.0 for n in MOTOR_NAMES}

    def __call__(self, raw: dict) -> dict:
        a = self.alpha
        for n in MOTOR_NAMES:
            self._smooth[n] = a * raw.get(n, 0.0) + (1.0 - a) * self._smooth[n]
        # snap tiny residuals to zero so is_neutral() works cleanly
        return {n: v if abs(v) > 0.01 else 0.0 for n, v in self._smooth.items()}

    def reset(self) -> None:
        self._smooth = {n: 0.0 for n in MOTOR_NAMES}


# ── Pygame palette ─────────────────────────────────────────────────────────────

BLACK  = ( 15,  15,  15)
WHITE  = (220, 220, 220)
GRAY   = ( 80,  80,  80)
GREEN  = ( 50, 200,  80)
RED    = (220,  60,  60)
YELLOW = (220, 200,  50)
CYAN   = ( 50, 200, 220)
ORANGE = (220, 140,  50)
BLUE   = ( 60, 120, 220)


# ── Pygame widget helpers ──────────────────────────────────────────────────────

def draw_stick(surf, cx: int, cy: int, r: int, ax: float, ay: float, label: str):
    pygame.draw.circle(surf, GRAY, (cx, cy), r, 2)
    pygame.draw.line(surf, GRAY, (cx - r, cy), (cx + r, cy), 1)
    pygame.draw.line(surf, GRAY, (cx, cy - r), (cx, cy + r), 1)
    dx = int(ax * (r - 6)); dy = int(ay * (r - 6))
    color = GREEN if (abs(ax) > DEADZONE or abs(ay) > DEADZONE) else GRAY
    pygame.draw.circle(surf, color, (cx + dx, cy + dy), 8)
    font = pygame.font.SysFont("monospace", 13)
    surf.blit(font.render(label, True, WHITE), (cx - 20, cy + r + 6))
    surf.blit(font.render(f"{ax:+.2f} {ay:+.2f}", True, YELLOW), (cx - 30, cy + r + 22))


def draw_button(surf, x: int, y: int, pressed: bool, label: str):
    color = GREEN if pressed else GRAY
    pygame.draw.circle(surf, color, (x, y), 14)
    font = pygame.font.SysFont("monospace", 11)
    surf.blit(font.render(label, True, BLACK if pressed else WHITE), (x - 7, y - 7))


# ── Graceful shutdown ──────────────────────────────────────────────────────────

# Two-stage landing (2026-06-12, rest pose re-sited 2026-06-14). REST_POSE is the
# full-torque approach target, ~12 mm (FK-checked) above SETTLE_POSE — margin against
# collisions. From there graceful_shutdown clamps Torque_Limit to SOFT_TORQUE and
# floats down to SETTLE_POSE (the resting equilibrium): the limited servo stalls gently
# on the table instead of pressing, and the final torque-off drops millimeters.
# SETTLE_POSE is the pose Javad hand-placed (2026-06-14) so the folded arm does NOT
# occlude the desk ArUco tag — the old (lift 81, elbow -5) pose blocked it. REST_POSE
# is the same pose raised ~12 mm (shoulder_lift -12, wrist_flex compensates to stay level).
REST_POSE = {
    "shoulder_pan":  4.7,
    "shoulder_lift": 47.7,
    "elbow_flex":    36.6,
    "wrist_flex":    -56.4,
    "wrist_roll":    0.0,
    "gripper":       4.0,
}

SETTLE_POSE = {
    "shoulder_pan":  4.7,
    "shoulder_lift": 59.7,
    "elbow_flex":    36.6,
    "wrist_flex":    -74.4,
    "wrist_roll":    0.0,
    "gripper":       4.0,
}

SOFT_TORQUE = 150      # Torque_Limit (RAM, 0-1000) during the compliant descent


def graceful_shutdown(robot, duration_s: float = 5.0, hz: float = 30) -> None:
    """Interpolate all joints to REST_POSE before torque is disabled."""
    import time
    print("\nShutting down — moving to rest pose...")
    try:
        obs = robot.get_observation()
        start = {n: obs.get(f"{n}.pos", 0.0) for n in MOTOR_NAMES}
    except Exception:
        return

    steps = max(1, round(duration_s * hz))
    dt    = duration_s / steps
    for i in range(1, steps + 1):
        alpha = i / steps
        # ease-out: fast at first, slows near rest so torque-off is gentle
        alpha_eased = 1.0 - (1.0 - alpha) ** 2
        goal = {n: start[n] + alpha_eased * (REST_POSE[n] - start[n]) for n in MOTOR_NAMES}
        try:
            robot.send_action({f"{n}.pos": goal[n] for n in MOTOR_NAMES})
        except Exception:
            break
        time.sleep(dt)

    # compliant final descent: clamp servo output, float onto the table, stall
    # gently on contact; then zero the position error before restoring torque so
    # nothing presses, and the upcoming torque-off is a non-event.
    try:
        robot.bus.sync_write("Torque_Limit",
                             {n: SOFT_TORQUE for n in MOTOR_NAMES}, normalize=False)
        steps = round(2.0 * hz)
        for i in range(1, steps + 1):
            alpha = i / steps
            goal = {n: REST_POSE[n] + alpha * (SETTLE_POSE[n] - REST_POSE[n])
                    for n in MOTOR_NAMES}
            robot.send_action({f"{n}.pos": goal[n] for n in MOTOR_NAMES})
            time.sleep(1.0 / hz)
        time.sleep(0.3)
        obs = robot.get_observation()
        robot.send_action({f"{n}.pos": obs.get(f"{n}.pos", SETTLE_POSE[n])
                           for n in MOTOR_NAMES})
        robot.bus.sync_write("Torque_Limit",
                             {n: 1000 for n in MOTOR_NAMES}, normalize=False)
    except Exception:
        pass
    print("  Rest pose reached (soft landing).")


# ── Pygame widget helpers ──────────────────────────────────────────────────────

def draw_controller(surf, joystick, profile: dict,
                    stick_left_xy: tuple, stick_right_xy: tuple,
                    shoulder_col_x: int, face_center: tuple):
    """Draw sticks, shoulder buttons, and face buttons in standard layout."""
    try:
        lx = filtered_axis_value(joystick, profile, 0)
        ly = filtered_axis_value(joystick, profile, 1)
        rx = filtered_axis_value(joystick, profile, 2)
        ry = filtered_axis_value(joystick, profile, 3)
    except Exception:
        lx = ly = rx = ry = 0.0

    draw_stick(surf, *stick_left_xy,  65, lx, ly, "L: pan / lift")
    draw_stick(surf, *stick_right_xy, 65, rx, ry, "R: wrist / elbow")

    sh = profile["shoulder"]
    for row, (key, idx) in enumerate(sh.items()):
        try:    pressed = joystick.get_button(idx)
        except: pressed = False
        draw_button(surf, shoulder_col_x, 50 + row * 36, pressed, key)

    fcx, fcy = face_center
    face_pos = {0: (fcx,      fcy + 30), 1: (fcx + 30, fcy),
                2: (fcx,      fcy - 30), 3: (fcx - 30, fcy)}
    for idx, lbl in profile["face"].items():
        try:    pressed = joystick.get_button(idx)
        except: pressed = False
        x, y = face_pos.get(idx, (fcx, fcy))
        draw_button(surf, x, y, pressed, lbl)


# ── Gamepad debugger / calibrator (run this file directly) ───────────────────────────
# (Absorbed from the former gamepad_debug.py — no separate entry point.)

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

    os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
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
