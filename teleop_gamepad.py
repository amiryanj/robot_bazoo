#!/usr/bin/env python
"""
SO-101 integrated teleoperation — gamepad + MuJoCo twin + Rerun.

What runs simultaneously:
  1. Pygame panel     — live joystick axes, buttons, joint position bars
  2. MuJoCo viewer   — 3-D digital twin of the arm (real-time joint angles)
  3. Rerun viewer    — camera feeds (Realsense + wrist) + joint time-series

Controls (Nintendo Switch Pro Controller):
  Left  stick L/R   → shoulder_pan
  Left  stick U/D   → shoulder_lift
  Right stick U/D   → elbow_flex
  Right stick L/R   → wrist_roll
  L / R buttons     → wrist_flex tilt
  ZL / ZR           → gripper close / open
  Ctrl-C            → exit
"""

import argparse
import math
import os
import sys
import time

os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

import mujoco
import mujoco.viewer
import numpy as np
import pygame
import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.teleoperators.gamepad import GamepadTeleop
from lerobot.teleoperators.gamepad.configuration_gamepad import GamepadTeleopConfig
from lerobot.utils.robot_utils import precise_sleep

from gamepad_utils import (
    MOTOR_NAMES, JOINT_LIMITS,
    detect_profile, get_joint_deltas, apply_deltas, is_neutral,
    graceful_shutdown, DeltaSmoother,
    BLACK, WHITE, GRAY, GREEN, RED, YELLOW, CYAN,
    draw_stick, draw_button, draw_controller,
)

# ── Config ────────────────────────────────────────────────────────────────────

SCENE_XML = "/home/javad/workspace/lerobot_all/SO-ARM100/Simulation/SO101/scene.xml"
PORT      = "/dev/ttyACM0"
ROBOT_ID  = "so101"
FPS       = 30

# ── Robot config with both cameras ───────────────────────────────────────────

def make_robot_config(use_realsense: bool = True, use_wrist: bool = True):
    cameras = {}
    if use_realsense:
        cameras["realsense"] = RealSenseCameraConfig(
            serial_number_or_name="117222251972",
            fps=15, width=640, height=480,
        )
    if use_wrist:
        cameras["wrist"] = OpenCVCameraConfig(
            index_or_path=15,
            fps=25, width=640, height=480,
        )
    return SOFollowerRobotConfig(port=PORT, id=ROBOT_ID, cameras=cameras)

# ── MuJoCo twin ───────────────────────────────────────────────────────────────

def update_twin(mj_data, obs: dict):
    for name in MOTOR_NAMES:
        key = f"{name}.pos"
        if key in obs:
            mj_data.joint(name).qpos[0] = math.radians(obs[key])

# ── Rerun logging ─────────────────────────────────────────────────────────────

def log_to_rerun(obs: dict, t: float):
    rr.set_time("time", timestamp=t)
    for cam_name in ("realsense", "wrist"):
        frame = obs.get(cam_name)
        if frame is not None and isinstance(frame, np.ndarray):
            rr.log(f"cameras/{cam_name}", rr.Image(frame))
    for name in MOTOR_NAMES:
        val = obs.get(f"{name}.pos")
        if val is not None:
            rr.log(f"joints/{name}", rr.Scalars(float(val)))

# ── Pygame debug panel ────────────────────────────────────────────────────────

WIN_W, WIN_H = 520, 340


def draw_debug(surf, joystick, obs: dict, profile: dict):
    surf.fill(BLACK)
    if joystick is None:
        surf.blit(pygame.font.SysFont("monospace", 18).render(
            "No gamepad detected", True, RED), (140, 140))
        return

    draw_controller(surf, joystick, profile,
                    stick_left_xy=(100, 120), stick_right_xy=(290, 120),
                    shoulder_col_x=460, face_center=(430, 90))

    font = pygame.font.SysFont("monospace", 12)
    surf.blit(font.render("JOINT POSITIONS (deg)", True, CYAN), (10, 218))
    for j, name in enumerate(MOTOR_NAMES):
        val = obs.get(f"{name}.pos", 0.0)
        lo, hi = JOINT_LIMITS[name]
        bar_w = int(max(0, min(1, (val - lo) / max(hi - lo, 1))) * 100)
        pygame.draw.rect(surf, GRAY,  (160, 235 + j * 16, 100, 10))
        pygame.draw.rect(surf, GREEN, (160, 235 + j * 16, bar_w, 10))
        surf.blit(font.render(f"{name:<14} {val:+6.1f}°", True, WHITE),
                  (10, 233 + j * 16))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-realsense", action="store_true")
    parser.add_argument("--no-wrist",     action="store_true")
    parser.add_argument("--twin",  action="store_true",
                        help="Launch MuJoCo 3-D digital twin viewer.")
    parser.add_argument("--torque", type=int, default=None,
                        help="Set Max_Torque_Limit for all motors (0-1000).")
    args = parser.parse_args()

    if args.twin:
        os.chdir(os.path.dirname(SCENE_XML))
        mj_model = mujoco.MjModel.from_xml_path(SCENE_XML)
        mj_data  = mujoco.MjData(mj_model)
    else:
        mj_model = mj_data = None

    rr.init("so101_teleop")
    rr.connect_grpc()

    robot  = SOFollower(make_robot_config(
        use_realsense=not args.no_realsense,
        use_wrist=not args.no_wrist,
    ))
    teleop = GamepadTeleop(GamepadTeleopConfig(use_gripper=True))

    print("Connecting robot (cameras may take a few seconds)...")
    robot.connect()
    teleop.connect()

    if args.torque is not None:
        for name in MOTOR_NAMES:
            robot.bus.write("Max_Torque_Limit", name, args.torque, normalize=False)
        print(f"  Max_Torque_Limit set to {args.torque}/1000 for all motors.")

    if teleop.gamepad is None or teleop.gamepad.joystick is None:
        print("\nERROR: No gamepad detected.\n", file=sys.stderr)
        robot.disconnect()
        return

    joystick = teleop.gamepad.joystick
    profile  = detect_profile(joystick)

    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("SO-101 Gamepad Debug")
    twin = mujoco.viewer.launch_passive(mj_model, mj_data) if args.twin else None

    mode_str = "Rerun + MuJoCo + Pygame" if args.twin else "Rerun + Pygame"
    print(f"\nRunning at {FPS} Hz — {mode_str}.")
    print("Ctrl-C to exit.\n")

    obs      = robot.get_observation()
    goal_pos = {n: obs.get(f"{n}.pos", 0.0) for n in MOTOR_NAMES}  # held across frames
    smoother = DeltaSmoother(alpha=0.45)
    t0       = time.perf_counter()

    try:
        while twin.is_running() if twin else True:
            t_loop = time.perf_counter()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            obs = robot.get_observation()

            teleop.gamepad.update()
            deltas = smoother(get_joint_deltas(joystick, profile))

            if not is_neutral(deltas):
                goal_pos = apply_deltas(goal_pos, deltas)
                robot.send_action({f"{n}.pos": goal_pos[n] for n in MOTOR_NAMES})

            if twin:
                update_twin(mj_data, obs)
                mujoco.mj_forward(mj_model, mj_data)
                twin.sync()

            log_to_rerun(obs, t_loop - t0)

            draw_debug(screen, joystick, obs, profile)
            pygame.display.flip()

            precise_sleep(max(1.0 / FPS - (time.perf_counter() - t_loop), 0.0))

    except KeyboardInterrupt:
        pass
    finally:
        graceful_shutdown(robot)
        if twin:
            twin.close()
        teleop.disconnect()
        try:
            robot.disconnect()
        except Exception as e:
            print(f"  Disconnect warning (torque disable likely succeeded): {e}")


if __name__ == "__main__":
    main()
