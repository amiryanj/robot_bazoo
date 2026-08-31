#!/usr/bin/env python
"""Autonomous data-collection session: drive the arm through safe in-air poses while
the top-down Realsense captures frames — gets the finger hearts in varied orientations
for detector training (a static rest pose never shows them).

Safety: joint-space poses are FK-filtered into a clear air box ABOVE the workspace
(no contact possible), moves are slow interpolations, and the arm is eased to the
rest pose on every exit. One bounded session (~5 min).

Saves per pose: frame_<i>.png, depth_<i>.npy, joints_<i>.json (joints enable
FK-projected label QC later: a heart label must sit near the projected gripper).

    python vision/collect_marker_data.py [--poses 18] [--port /dev/ttyACM1]
"""
import argparse
import itertools
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

# clear-air box for the TCP (base frame, metres): central, well above plate + ball
# (ball-on-cylinder sits at ~(0.30, -0.14), top ~z+0.02 — box stays clear of it)
BOX_X = (0.15, 0.27)
BOX_Y = (-0.10, 0.13)
BOX_Z = (0.07, 0.16)
ROLLS = (-90, -50, 0, 50, 90)
MOVE_S = 2.2
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def candidate_poses(kin, n_target):
    """Joint-space poses whose TCP lands in the safe box, spread over pan/wrist/roll."""
    cands = []
    for pan, lift, elbow, wrist in itertools.product(
            (-25, 0, 25), (-30, -10, 10, 30), (10, 40, 70), (-40, 0, 40, 80)):
        ang = {"shoulder_pan": pan, "shoulder_lift": lift, "elbow_flex": elbow,
               "wrist_flex": wrist, "wrist_roll": 0.0, "gripper": 30.0}
        _, p = kin.fk(ang)
        if (BOX_X[0] <= p[0] <= BOX_X[1] and BOX_Y[0] <= p[1] <= BOX_Y[1]
                and BOX_Z[0] <= p[2] <= BOX_Z[1]):
            cands.append((ang, p))
    # spread: sort by (pan, wrist) and stride-sample, then assign rolls round-robin
    cands.sort(key=lambda c: (c[0]["shoulder_pan"], c[0]["wrist_flex"]))
    step = max(len(cands) // max(n_target, 1), 1)
    out = []
    for i, (ang, p) in enumerate(cands[::step][:n_target]):
        a = dict(ang)
        a["wrist_roll"] = float(ROLLS[i % len(ROLLS)])
        out.append(a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, default=18)
    ap.add_argument("--port", default="/dev/ttyACM1")
    args = ap.parse_args()

    import cv2
    from pick_ball import Kin, move_to, read_angles
    from realsense import Realsense
    from gamepad_utils import graceful_shutdown
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    kin = Kin()
    poses = candidate_poses(kin, args.poses)
    print(f"{len(poses)} FK-checked in-air poses")
    if not poses:
        sys.exit("no safe poses found")

    out = ROOT / "outputs/vision" / f"markers_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out.mkdir(parents=True, exist_ok=True)

    cam = Realsense()
    robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
    robot.connect()
    try:
        cur = read_angles(robot)
        for i, pose in enumerate(poses):
            cur = move_to(robot, cur, pose, seconds=MOVE_S)
            time.sleep(0.6)                                # settle, kill motion blur
            color, depth, K = cam.grab()
            ang = read_angles(robot)                       # measured, not commanded
            cv2.imwrite(str(out / f"frame_{i:02d}.png"), color)
            np.save(out / f"depth_{i:02d}.npy", depth)
            json.dump(dict(joints=ang, K=K), open(out / f"joints_{i:02d}.json", "w"))
            print(f"  pose {i + 1}/{len(poses)} captured "
                  f"(roll={pose['wrist_roll']:.0f}, wrist={pose['wrist_flex']:.0f})")
    except Exception as e:
        print(f"\nUnexpected error: {e!r}\nLanding the arm safely...")
    finally:
        cam.stop()
        graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception:
            pass
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
