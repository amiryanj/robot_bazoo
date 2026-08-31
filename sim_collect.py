#!/usr/bin/env python
"""
SO-101 simulation-based data collection.

Gamepad drives the MuJoCo SO-101 model with real physics (ball rolls/bounces).
Both cameras are rendered each frame and streamed to Rerun.
Press + to start/stop recording episodes; episodes save in LeRobot dataset format.

Controls (Nintendo Switch Pro Controller):
  Sticks / L / R / ZL / ZR  →  joint control (same mapping as teleop_gamepad.py)
  + (button 9)               →  start / stop recording current episode
  - (button 8)               →  reset ball to starting position (keeps arm pose)
  Ctrl-C                     →  exit (auto-saves any open episode)

Usage:
    conda activate lerobot
    python sim_collect.py
    python sim_collect.py --push           # push dataset to HF Hub when done
    python sim_collect.py --repo-id <id>   # override HuggingFace repo id
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

import mujoco
import mujoco.viewer
import numpy as np
import pygame
import rerun as rr

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from gamepad_utils import (
    MOTOR_NAMES, JOINT_LIMITS,
    detect_profile, get_joint_deltas, apply_deltas, is_neutral, ButtonDebouncer,
    BLACK, WHITE, GRAY, GREEN, RED, YELLOW, CYAN, ORANGE,
    draw_stick, draw_button, draw_controller,
)

# ── Config ────────────────────────────────────────────────────────────────────

from config import SCENE_SIM_XML as SCENE_XML, DATASETS
FPS         = 30
HF_REPO_ID  = "mgh-ja-1395/so101-sim-ball"
DATASET_ROOT = DATASETS / "so101-sim-ball"
TASK        = "push the orange ball with the robot arm"

BALL_INIT   = [0.22, 0.0, 0.02, 1.0, 0.0, 0.0, 0.0]  # xyz + quaternion (wxyz)

# ── MuJoCo helpers ────────────────────────────────────────────────────────────

def get_joint_pos_deg(mj_data) -> dict:
    return {name: math.degrees(mj_data.joint(name).qpos[0]) for name in MOTOR_NAMES}

def set_joint_targets(mj_data, mj_model, joint_pos_deg: dict):
    for name in MOTOR_NAMES:
        mj_data.ctrl[mj_model.actuator(name).id] = math.radians(joint_pos_deg[name])

def step_physics(mj_model, mj_data):
    n_substeps = max(1, round(1.0 / FPS / mj_model.opt.timestep))
    for _ in range(n_substeps):
        mujoco.mj_step(mj_model, mj_data)

def reset_ball(mj_model, mj_data):
    """Reset only the ball to its starting pose; arm stays."""
    jnt_id      = mj_model.joint("ball_free").id
    qposadr     = mj_model.jnt_qposadr[jnt_id]
    dofadr      = mj_model.jnt_dofadr[jnt_id]
    mj_data.qpos[qposadr:qposadr + 7] = BALL_INIT
    mj_data.qvel[dofadr:dofadr + 6]   = 0.0
    mujoco.mj_forward(mj_model, mj_data)

# ── Rendering ─────────────────────────────────────────────────────────────────

def render_cameras(wrist_rend, ext_rend, mj_model, mj_data):
    wrist_rend.update_scene(mj_data, camera="wrist_cam")
    ext_rend.update_scene(mj_data,   camera="external_cam")
    return wrist_rend.render().copy(), ext_rend.render().copy()

# ── LeRobot dataset ───────────────────────────────────────────────────────────

FEATURES = {
    "observation.state": {
        "dtype": "float32", "shape": (6,), "names": MOTOR_NAMES,
    },
    "observation.images.wrist": {
        "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"],
    },
    "observation.images.external": {
        "dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"],
    },
    "action": {
        "dtype": "float32", "shape": (6,), "names": MOTOR_NAMES,
    },
}

def make_frame(obs_deg: dict, action_deg: dict,
               wrist_img: np.ndarray, ext_img: np.ndarray) -> dict:
    return {
        "task":                        TASK,
        "observation.state":           np.array([obs_deg[n]    for n in MOTOR_NAMES], dtype=np.float32),
        "action":                      np.array([action_deg[n] for n in MOTOR_NAMES], dtype=np.float32),
        "observation.images.wrist":    wrist_img,
        "observation.images.external": ext_img,
    }

# ── Rerun ─────────────────────────────────────────────────────────────────────

def log_to_rerun(obs_deg: dict, wrist_img, ext_img, t: float, recording: bool):
    rr.set_time("time", timestamp=t)
    if wrist_img is not None:
        rr.log("cameras/wrist",    rr.Image(wrist_img))
    if ext_img is not None:
        rr.log("cameras/external", rr.Image(ext_img))
    for name in MOTOR_NAMES:
        rr.log(f"joints/{name}", rr.Scalars(obs_deg[name]))
    rr.log("recording", rr.Scalars(float(recording)))

# ── Pygame debug panel ────────────────────────────────────────────────────────

WIN_W, WIN_H = 540, 400


def draw_debug(surf, joystick, obs_deg: dict, profile: dict,
               recording: bool, n_saved: int, frame_count: int):
    surf.fill(BLACK)
    font_sm = pygame.font.SysFont("monospace", 12)
    font_md = pygame.font.SysFont("monospace", 14)

    if recording:
        pygame.draw.circle(surf, RED, (520, 16), 8)
        surf.blit(font_md.render(f"REC  ep={n_saved + 1}  frame={frame_count}", True, RED), (10, 8))
    else:
        surf.blit(font_md.render(f"IDLE  {n_saved} episodes saved", True, CYAN), (10, 8))

    if joystick is None:
        surf.blit(font_md.render("No gamepad", True, RED), (150, 150))
    else:
        draw_controller(surf, joystick, profile,
                        stick_left_xy=(100, 130), stick_right_xy=(290, 130),
                        shoulder_col_x=470, face_center=(445, 100))

    surf.blit(font_sm.render("JOINT POSITIONS (deg)", True, CYAN), (10, 238))
    for j, name in enumerate(MOTOR_NAMES):
        val = obs_deg.get(name, 0.0)
        lo, hi = JOINT_LIMITS[name]
        bar_w = int(max(0, min(1, (val - lo) / max(hi - lo, 1))) * 100)
        pygame.draw.rect(surf, GRAY,  (160, 255 + j * 16, 100, 10))
        pygame.draw.rect(surf, GREEN, (160, 255 + j * 16, bar_w, 10))
        surf.blit(font_sm.render(f"{name:<14} {val:+6.1f}°", True, WHITE), (10, 253 + j * 16))

    surf.blit(font_sm.render("+ = start/stop record    - = reset ball", True, GRAY), (10, 380))

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=HF_REPO_ID,
                        help="HuggingFace dataset repo id")
    parser.add_argument("--root",    default=str(DATASET_ROOT),
                        help="Local dataset root directory")
    parser.add_argument("--push",      action="store_true",
                        help="Push dataset to HF Hub on exit (requires huggingface-cli login)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete existing dataset directory and start fresh")
    args = parser.parse_args()

    # MuJoCo
    os.chdir(os.path.dirname(SCENE_XML))
    mj_model = mujoco.MjModel.from_xml_path(SCENE_XML)
    mj_data  = mujoco.MjData(mj_model)

    # Zero all actuators (safe starting pose)
    for name in MOTOR_NAMES:
        mj_data.ctrl[mj_model.actuator(name).id] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    # Renderers
    wrist_rend = mujoco.Renderer(mj_model, height=480, width=640)
    ext_rend   = mujoco.Renderer(mj_model, height=480, width=640)

    # Rerun
    rr.init("so101_sim_collect")
    rr.connect_grpc()

    # Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("SO-101 Sim Collect")

    # MuJoCo passive viewer
    twin = mujoco.viewer.launch_passive(mj_model, mj_data)

    # Gamepad
    if pygame.joystick.get_count() == 0:
        print("ERROR: No gamepad detected.", file=sys.stderr)
        twin.close()
        return
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    profile = detect_profile(joystick)
    debouncer = ButtonDebouncer()          # reject single-frame button chatter (worn ZR)

    # Dataset
    import json, shutil
    dataset_root = Path(args.root)
    if dataset_root.exists():
        meta_info = dataset_root / "meta" / "info.json"
        n_existing = 0
        if meta_info.exists():
            try:
                n_existing = json.loads(meta_info.read_text()).get("total_episodes", 0)
            except Exception:
                pass
        if n_existing > 0 and not args.overwrite:
            print(f"\nDataset at {dataset_root} already has {n_existing} episodes.")
            print("  Use --overwrite to delete it and start fresh.\n")
            twin.close(); pygame.quit(); sys.exit(1)
        shutil.rmtree(dataset_root)
        if n_existing > 0:
            print(f"  Deleted {n_existing}-episode dataset (--overwrite).")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=FPS,
        features=FEATURES,
        root=dataset_root,
        robot_type="so101_follower",
    )

    recording    = False
    frame_count  = 0
    n_saved      = 0
    joint_pos    = {n: 0.0 for n in MOTOR_NAMES}  # commanded targets (degrees)
    t0           = time.perf_counter()

    print(f"\nSim collect ready — dataset: {args.root}")
    print("  + button  = start / stop recording episode")
    print("  - button  = reset ball to starting position")
    print("  Ctrl-C    = exit\n")

    try:
        while twin.is_running():
            t_loop = time.perf_counter()

            # ── Events ────────────────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.JOYBUTTONDOWN:
                    if event.button == 9:       # + = toggle recording
                        if not recording:
                            recording   = True
                            frame_count = 0
                            print(f"  Recording episode {n_saved + 1} ...")
                        else:
                            recording = False
                            if frame_count > 0:
                                dataset.save_episode()
                                n_saved += 1
                                print(f"  Saved episode {n_saved} ({frame_count} frames)")
                            else:
                                print("  Discarded empty episode")
                    elif event.button == 8:     # - = reset ball
                        reset_ball(mj_model, mj_data)
                        print("  Ball reset.")

            # ── Joint control ─────────────────────────────────────────────────
            deltas = get_joint_deltas(joystick, profile, dt=1.0 / FPS, debounce=debouncer)
            if not is_neutral(deltas):
                joint_pos = apply_deltas(joint_pos, deltas)
            set_joint_targets(mj_data, mj_model, joint_pos)
            step_physics(mj_model, mj_data)

            # ── Render cameras ────────────────────────────────────────────────
            wrist_img, ext_img = render_cameras(wrist_rend, ext_rend, mj_model, mj_data)

            # ── Observations ──────────────────────────────────────────────────
            obs_deg = get_joint_pos_deg(mj_data)  # current sim joint positions

            # ── MuJoCo viewer ─────────────────────────────────────────────────
            twin.sync()

            # ── Rerun ─────────────────────────────────────────────────────────
            log_to_rerun(obs_deg, wrist_img, ext_img, t_loop - t0, recording)

            # ── Record frame ──────────────────────────────────────────────────
            if recording:
                dataset.add_frame(make_frame(obs_deg, joint_pos, wrist_img, ext_img))
                frame_count += 1

            # ── Pygame panel ──────────────────────────────────────────────────
            draw_debug(screen, joystick, obs_deg, profile, recording, n_saved, frame_count)
            pygame.display.flip()

            # ── Timing ────────────────────────────────────────────────────────
            time.sleep(max(1.0 / FPS - (time.perf_counter() - t_loop), 0.0))

    except KeyboardInterrupt:
        if recording and frame_count > 0:
            dataset.save_episode()
            n_saved += 1
            print(f"\n  Auto-saved last episode ({frame_count} frames)")
        print(f"\nDone. {n_saved} episodes saved to {args.root}")
    finally:
        if args.push and n_saved > 0:
            print("Pushing to HuggingFace Hub...")
            dataset.push_to_hub()
            print("Done.")
        del wrist_rend
        del ext_rend
        twin.close()
        pygame.quit()


if __name__ == "__main__":
    main()
