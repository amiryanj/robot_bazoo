#!/usr/bin/env python
"""Record real-arm pick demos into a LeRobot dataset, driven by the scripted policy.

This is critical-path step 4 (CLAUDE.md): the scripted pick (`pick_ball.run_grasp`)
runs the arm while every commanded step is logged as a LeRobot frame
(measured joint state + commanded action + top-down Realsense + wrist camera).
There is no leader arm and joystick pick-and-place is too clumsy, so the
demonstrator IS the scripted policy.

Per episode it asks you to place the ball, runs one full pick cycle, then asks
keep / discard (the scripted grasp succeeds only sometimes — keep the good ones).
Episodes append to the same dataset across sessions.

Usage:
    conda activate lerobot
    python record_pick.py                 # record into datasets/so101-pick-ball, both cams
    python record_pick.py --no-wrist      # top-down Realsense only (dodges the USB stall)
    python record_pick.py --episodes 1    # stop after 1 kept episode
    python record_pick.py --overwrite     # delete the dataset and start fresh
    python record_pick.py --push          # push to the HF Hub on exit

The arm ALWAYS lands via graceful_shutdown on exit, whatever happens.

Note on rate: frames are logged one-per-commanded-step at ~FPS Hz (the scripted
loop plus two camera grabs). The declared dataset fps is nominal — good enough for
a first ACT smoke test; lock it down later if replay speed matters.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from pick_ball import BallDetector, Kin, MOTOR_NAMES, run_grasp
from gamepad_utils import graceful_shutdown
from config import ARM_PORT

FPS = 10
REPO_ID = "mgh-ja-1395/so101-pick-ball"
ROOT_DIR = ROOT / "datasets/so101-pick-ball"
TASK = "pick up the basketball"
WRIST_INDEX = 15


def features(use_wrist):
    f = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": MOTOR_NAMES},
        "observation.images.realsense": {
            "dtype": "video", "shape": (480, 640, 3),
            "names": ["height", "width", "channels"]},
        "action": {"dtype": "float32", "shape": (6,), "names": MOTOR_NAMES},
    }
    if use_wrist:
        f["observation.images.wrist"] = {
            "dtype": "video", "shape": (480, 640, 3),
            "names": ["height", "width", "channels"]}
    return f


class RecordingRobot:
    """Wrap an SOFollower so every send_action() also logs one dataset frame:
    measured state (read now) + commanded action + a fresh frame from each camera.
    Everything else (connect, get_observation, ...) delegates to the real robot, so
    pick_ball.run_grasp drives this transparently."""

    def __init__(self, robot, dataset, cam, wrist, use_wrist):
        self._robot = robot
        self.ds = dataset
        self.cam = cam
        self.wrist = wrist
        self.use_wrist = use_wrist
        self.n_frames = 0
        self._last_wrist = np.zeros((480, 640, 3), np.uint8)

    def __getattr__(self, name):
        return getattr(self._robot, name)

    def get_observation(self):
        return self._robot.get_observation()

    def _wrist_rgb(self):
        ok, frame = self.wrist.read()
        if ok and frame is not None:
            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480))
            self._last_wrist = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._last_wrist

    def send_action(self, action):
        obs = self._robot.get_observation()
        state = np.array([obs.get(f"{n}.pos", 0.0) for n in MOTOR_NAMES], np.float32)
        act = np.array([action.get(f"{n}.pos", 0.0) for n in MOTOR_NAMES], np.float32)
        color, _, _ = self.cam.grab()
        frame = {
            "task": TASK,
            "observation.state": state,
            "action": act,
            "observation.images.realsense": cv2.cvtColor(color, cv2.COLOR_BGR2RGB),
        }
        if self.use_wrist:
            frame["observation.images.wrist"] = self._wrist_rgb()
        self.ds.add_frame(frame)
        self.n_frames += 1
        return self._robot.send_action(action)


def open_wrist():
    cap = cv2.VideoCapture(WRIST_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 25)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def main():
    ap = argparse.ArgumentParser(description="Record scripted pick demos to a LeRobot dataset.")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--root", default=str(ROOT_DIR))
    ap.add_argument("--episodes", type=int, default=0,
                    help="Stop after N kept episodes (0 = until you quit).")
    ap.add_argument("--no-wrist", action="store_true", help="Top-down Realsense only.")
    ap.add_argument("--overwrite", action="store_true", help="Delete the dataset and start fresh.")
    ap.add_argument("--push", action="store_true", help="Push to the HF Hub on exit.")
    ap.add_argument("--port", default=ARM_PORT)
    args = ap.parse_args()

    use_wrist = not args.no_wrist

    # ── Cameras (we own them; the robot is connected with cameras={}) ──────────────
    from realsense import Realsense
    cam = Realsense()
    wrist = None
    if use_wrist:
        wrist = open_wrist()
        if wrist is None:
            cam.stop()
            sys.exit(f"Wrist camera (index {WRIST_INDEX}) would not open — it may be the "
                     "two-camera USB stall. Re-run with --no-wrist for top-down only.")
        print("Both cameras open. If the Realsense stalls, it's the known USB-bandwidth "
              "issue — re-run with --no-wrist.")

    # ── Dataset (append if it exists, else create) ────────────────────────────────
    root = Path(args.root)
    if root.exists() and args.overwrite:
        shutil.rmtree(root)
    if root.exists():
        dataset = LeRobotDataset(args.repo_id, root=root)
        if ("observation.images.wrist" in dataset.meta.features) != use_wrist:
            sys.exit("Existing dataset's cameras don't match this run's "
                     f"(it {'has' if 'observation.images.wrist' in dataset.meta.features else 'lacks'} "
                     "a wrist stream). Match it, or --overwrite to start fresh.")
        print(f"Appending to {root} ({dataset.meta.total_episodes} episodes so far).")
    else:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id, fps=FPS, features=features(use_wrist),
            root=root, robot_type="so101_follower")
        print(f"Created new dataset at {root}.")

    # ── Robot + scripted-pick brains ──────────────────────────────────────────────
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    print("Loading detector + IK model...")
    detector = BallDetector()
    kin = Kin()
    robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
    robot.connect()
    rec = RecordingRobot(robot, dataset, cam, wrist, use_wrist)

    saved = 0
    try:
        while not (args.episodes and saved >= args.episodes):
            cmd = input(f"\nEpisode {saved + 1}: place the ball, then ENTER to record "
                        "(q = quit): ").strip().lower()
            if cmd == "q":
                break
            rec.n_frames = 0
            try:
                status = run_grasp(rec, cam, detector, kin)
            except KeyboardInterrupt:
                print("\n  interrupted — discarding this episode.")
                dataset.clear_episode_buffer()
                continue
            print(f"  grasp result: {status}  ({rec.n_frames} frames)")
            if rec.n_frames == 0:
                dataset.clear_episode_buffer()
                print("  nothing moved — skipping.")
                continue
            if input("  keep this episode? [Y/n] ").strip().lower() == "n":
                dataset.clear_episode_buffer()
                print("  discarded.")
            else:
                dataset.save_episode()
                saved += 1
                print(f"  saved. kept this session: {saved}  "
                      f"(dataset total: {dataset.meta.total_episodes})")
    finally:
        print("\nLanding the arm...")
        graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception:
            pass
        cam.stop()
        if wrist is not None:
            wrist.release()
        if args.push and saved > 0:
            print("Pushing to the HF Hub...")
            dataset.push_to_hub()
        print(f"Done. {saved} episode(s) recorded this session "
              f"-> {root} (total {dataset.meta.total_episodes}).")


if __name__ == "__main__":
    main()
