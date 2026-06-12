#!/usr/bin/env python
"""Identify the wrist_roll mapping offset (real zero vs MuJoCo model zero).

Holds an in-air pose, sweeps wrist_roll, tracks the pink heart with the camera, and
finds the constant roll offset DELTA (and a re-solved marker offset) that makes
FK(roll + DELTA) match the measurements. A constant-roll hand-eye calibration cannot
see this error — the solved marker offset absorbs it at that one roll.

    python vision/roll_id.py [--port /dev/ttyACM1]
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

# pose with the gripper roughly horizontal (heart can face the camera) and LOW —
# the D455 has ~0.4 m minimum depth; a high pose puts the heart in the blind zone.
# Visibility arc measured 2026-06-12: wrist_flex <= -20, roll in about [-70, +30].
BASE_POSE = {"shoulder_pan": 0.0, "shoulder_lift": 22.0, "elbow_flex": 45.0,
             "wrist_flex": -40.0, "wrist_roll": 0.0, "gripper": 2.0}   # closed jaws:
# spinning open jaws near the plate could brush it; heart is on the outer face anyway
ROLL_SWEEP = np.arange(-80, 41, 6.0)
MOTOR_NAMES = list(BASE_POSE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    args = ap.parse_args()

    import json
    import torch
    from handeye_calib import HeartDetector, Realsense, backproject, make_fk
    from pick_ball import move_to, read_angles
    from gamepad_utils import graceful_shutdown
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    he = json.load(open(ROOT / "outputs/calib/handeye.json"))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])

    det = HeartDetector("cuda" if torch.cuda.is_available() else "cpu")
    fk = make_fk()
    cam = Realsense()
    robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
    robot.connect()

    samples = []                                   # (angles_meas, p_base_measured)
    try:
        cur = read_angles(robot)
        cur = move_to(robot, cur, BASE_POSE, seconds=2.5)
        for roll in ROLL_SWEEP:
            tgt = dict(BASE_POSE, wrist_roll=float(roll))
            cur = move_to(robot, cur, tgt, seconds=0.8)
            time.sleep(0.5)
            color, depth, K = cam.grab()
            uv, _ = det.marker_uv(color)
            if uv is None:
                print(f"  roll={roll:+.0f}: heart not visible")
                continue
            pc = backproject(*uv, depth, K)
            if pc is None:
                print(f"  roll={roll:+.0f}: heart seen but NO DEPTH (D455 min-Z?)")
                continue
            ang = read_angles(robot)
            samples.append((ang, R_cb @ pc + t_cb))
            print(f"  roll={roll:+.0f}: heart seen at "
                  f"{np.round((R_cb @ pc + t_cb) * 1000).astype(int)}mm")
    except Exception as e:
        print(f"\nUnexpected error: {e!r}\nLanding the arm safely...")
    finally:
        cam.stop()
        graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception:
            pass

    if len(samples) < 6:
        sys.exit(f"only {len(samples)} sightings — not enough; adjust BASE_POSE so the "
                 f"heart faces the camera over more of the sweep.")

    # fit: for candidate DELTA, re-solve the marker offset (linear LSQ), report residual
    def fit(delta):
        A, b = [], []
        for ang, p_meas in samples:
            a2 = dict(ang); a2["wrist_roll"] = ang["wrist_roll"] + delta
            R_w, t_w = fk(a2)
            A.append(R_w)
            b.append(p_meas - t_w)
        A = np.concatenate(A); b = np.concatenate(b)
        o, *_ = np.linalg.lstsq(A, b, rcond=None)
        res = (A @ o - b).reshape(-1, 3)
        return float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))), o

    print(f"\n{len(samples)} sightings. Residual RMS vs assumed roll offset:")
    coarse = {d: fit(d)[0] for d in (-180, -90, 0, 90, 180)}
    for d, r in coarse.items():
        print(f"  DELTA={d:+4d}deg -> RMS={r * 1000:6.1f}mm")
    d0 = min(coarse, key=coarse.get)
    fine = {d: fit(d)[0] for d in np.arange(d0 - 20, d0 + 21, 2.0)}
    d_best = min(fine, key=fine.get)
    rms, o = fit(d_best)
    print(f"\nBEST: wrist_roll offset = {d_best:+.0f} deg  (RMS {rms * 1000:.1f}mm)")
    print(f"      re-solved marker offset = {np.round(o * 1000, 1)} mm")
    print("If DELTA is ~0 the mapping is fine and the visual was the mesh; "
          "if ~±90, apply it in make_fk + recalibrate.")


if __name__ == "__main__":
    main()
