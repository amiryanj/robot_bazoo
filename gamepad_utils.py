"""
Shared gamepad / joint-control utilities for SO-101 scripts.

Provides constants, controller profiles, joystick delta calculation,
and pygame drawing helpers used by teleop_gamepad, sim_collect, command_log.
"""

import copy
import json
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
        print("  Re-run: python gamepad_debug.py --calibrate")
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

# Two-stage landing (2026-06-12). REST_POSE is the full-torque approach target,
# ~14 mm (FK-checked) above the arm's measured min-energy pose — margin against
# collisions. From there graceful_shutdown clamps Torque_Limit to SOFT_TORQUE and
# floats down to SETTLE_POSE (the torque-off equilibrium measured from station.csv
# cold starts, commanded ~1.5 deg past contact): the limited servo stalls gently on
# the table instead of pressing, and the final torque-off drops millimeters. The
# old single pose (lift 40, wrist 0) dropped the gripper 87 mm with a clunk.
REST_POSE = {
    "shoulder_pan":  0.0,
    "shoulder_lift": 73.0,
    "elbow_flex":    -4.0,
    "wrist_flex":    -60.0,
    "wrist_roll":    0.0,
    "gripper":       4.0,
}

SETTLE_POSE = {
    "shoulder_pan":  0.0,
    "shoulder_lift": 81.0,
    "elbow_flex":    -5.0,
    "wrist_flex":    -72.0,
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
