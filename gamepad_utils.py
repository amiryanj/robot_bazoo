"""
Shared gamepad / joint-control utilities for SO-101 scripts.

Provides constants, controller profiles, joystick delta calculation,
and pygame drawing helpers used by teleop_gamepad, sim_collect, command_log.
"""

import pygame

# ── Robot constants ────────────────────────────────────────────────────────────

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

JOINT_SPEED = {
    "shoulder_pan":  2.5,
    "shoulder_lift": 2.0,
    "elbow_flex":    2.0,
    "wrist_flex":    1.5,
    "wrist_roll":    3.0,
    "gripper":       1.5,
}

JOINT_LIMITS = {
    "shoulder_pan":  (-90,  90),
    "shoulder_lift": (-90,  90),
    "elbow_flex":    (-90,  90),
    "wrist_flex":    (-80,  80),
    "wrist_roll":   (-150, 150),
    "gripper":       (  0, 100),
}

DEADZONE = 0.08

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


def detect_profile(joystick) -> dict:
    name = joystick.get_name().lower()
    for key, profile in CONTROLLER_PROFILES.items():
        if any(p in name for p in profile["patterns"]):
            print(f"  Controller: '{joystick.get_name()}' → {key} profile")
            return profile
    print(f"  Controller: '{joystick.get_name()}' → generic profile")
    return CONTROLLER_PROFILES["generic"]


# ── Joint velocity helpers ─────────────────────────────────────────────────────

def get_joint_deltas(joystick, profile: dict) -> dict:
    def axis(i):
        v = joystick.get_axis(i)
        return v if abs(v) > DEADZONE else 0.0

    def btn(i):
        try:    return joystick.get_button(i)
        except: return False

    lx = axis(0); ly = axis(1)
    rx = axis(2); ry = axis(3)
    sh = profile["shoulder"]
    return {
        "shoulder_pan":  -lx * JOINT_SPEED["shoulder_pan"],
        "shoulder_lift":  ly * JOINT_SPEED["shoulder_lift"],
        "elbow_flex":    -ry * JOINT_SPEED["elbow_flex"],
        "wrist_roll":     rx * JOINT_SPEED["wrist_roll"],
        "wrist_flex":    (btn(sh["L"]) - btn(sh["R"])) * JOINT_SPEED["wrist_flex"],
        "gripper":       (btn(sh["ZL"]) - btn(sh["ZR"])) * JOINT_SPEED["gripper"],
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

# Safe resting pose used during graceful shutdown.
# shoulder_lift should be low enough that gravity doesn't yank the arm
# when torque cuts off. Tune this to match your arm's physical rest position.
REST_POSE = {
    "shoulder_pan":  0.0,
    "shoulder_lift": 40.0,    # lower arm toward table — tune if needed
    "elbow_flex":    0.0,
    "wrist_flex":    0.0,
    "wrist_roll":    0.0,
    "gripper":       0.0,
}


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
    print("  Rest pose reached.")


# ── Pygame widget helpers ──────────────────────────────────────────────────────

def draw_controller(surf, joystick, profile: dict,
                    stick_left_xy: tuple, stick_right_xy: tuple,
                    shoulder_col_x: int, face_center: tuple):
    """Draw sticks, shoulder buttons, and face buttons in standard layout."""
    try:
        lx = joystick.get_axis(0); ly = joystick.get_axis(1)
        rx = joystick.get_axis(2); ry = joystick.get_axis(3)
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
