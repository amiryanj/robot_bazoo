#!/usr/bin/env python
"""Contact-free kinematic rehearsal: how far is FK from reality at grasp poses?

No tags needed — the depth camera measures the GRIPPER itself. The arm hovers at safe
offsets around the (sphere-fit certified) ball position; at each pose we segment the
gripper's points from the cloud (points above the ball, near the predicted TCP) and
compare:
  - absolute: cloud-measured gripper centroid / lowest point vs FK-predicted TCP,
  - differential: commanded 20 mm steps vs measured displacement per axis
    (differential errors expose mapping/scale problems independent of any constant
    site-vs-fingertip offset).

All poses stay >= 35 mm clear of the ball top. Graceful landing on every exit.

    python vision/kin_rehearsal.py [--port /dev/ttyACM1]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

CLEAR = 0.040                       # hover height above ball TOP (m)
STEP = 0.020                        # commanded delta per axis (m)
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def gripper_cloud_stats(depth, K, R_cb, t_cb, p_tcp, z_floor):
    """Segment gripper points: in base frame, above z_floor, within 80 mm (xy) of the
    predicted TCP. Returns (centroid, min_z, n) or None."""
    from cloud import crop_z, deproject
    pts = crop_z(deproject(depth, K), (0.20, 1.2))
    base = (R_cb @ pts.T).T + t_cb
    sel = base[(base[:, 2] > z_floor)
               & (np.linalg.norm(base[:, :2] - p_tcp[:2], axis=1) < 0.080)]
    if len(sel) < 80:
        return None
    return sel.mean(0), float(np.percentile(sel[:, 2], 2)), len(sel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    args = ap.parse_args()

    from handeye_calib import Realsense
    from pick_ball import Kin, ik_best, localize_base, move_to, read_angles, GRIP_OPEN
    from gamepad_utils import graceful_shutdown
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    he = json.load(open(ROOT / "outputs/calib/handeye.json"))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])

    print("Localizing ball (sphere fit)...")
    p_ball, ball = localize_base()
    if p_ball is None or not ball["fit_ok"]:
        sys.exit("No certified ball localization — aborting.")
    z_top = p_ball[2] + ball["radius_m"]
    print(f"  ball at {np.round(p_ball * 1000).astype(int)} mm, top at {z_top * 1000:.0f} mm")

    home = np.array([p_ball[0], p_ball[1], z_top + CLEAR])
    offsets = [np.zeros(3),
               [STEP, 0, 0], [-STEP, 0, 0], [0, 0, 0],
               [0, STEP, 0], [0, -STEP, 0], [0, 0, 0],
               [0, 0, STEP], [0, 0, 2 * STEP], [0, 0, 0]]

    kin = Kin()
    cam = Realsense()
    robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
    robot.connect()
    rows = []
    try:
        cur = read_angles(robot)
        for i, off in enumerate(offsets):
            target = home + np.asarray(off, float)
            sol, e_pos, tilt = ik_best(kin, target)
            if e_pos > 0.010:
                print(f"  pose {i}: IK err {e_pos * 1000:.0f}mm — skipped")
                continue
            sol["gripper"] = GRIP_OPEN
            cur = move_to(robot, cur, sol, seconds=1.6)
            time.sleep(0.6)
            color, depth, K = cam.grab()
            ang = read_angles(robot)
            _, p_fk = kin.fk(ang)                       # TCP from MEASURED joints
            st = gripper_cloud_stats(depth, K, R_cb, t_cb, p_fk, z_top + 0.005)
            if st is None:
                print(f"  pose {i}: gripper not segmented in cloud")
                continue
            cen, zmin, n = st
            rows.append(dict(off=off, target=target, p_fk=p_fk, cen=cen, zmin=zmin))
            print(f"  pose {i}: cmd_off={np.round(np.asarray(off) * 1000).astype(int)} "
                  f"fk_tcp={np.round(p_fk * 1000).astype(int)} "
                  f"cloud_cen={np.round(cen * 1000).astype(int)} "
                  f"cloud_zmin={zmin * 1000:.0f}  n={n}")
    except Exception as e:
        print(f"\nUnexpected error: {e!r}\nLanding the arm safely...")
    finally:
        cam.stop()
        graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception:
            pass

    if len(rows) < 4:
        sys.exit("not enough poses measured")

    # absolute: cloud centroid minus FK TCP (constant offset = site-vs-body geometry,
    # NOT necessarily an error; its SPREAD across poses is the error signal)
    diff = np.array([r["cen"] - r["p_fk"] for r in rows]) * 1000
    print(f"\ncloud-centroid minus FK-TCP [mm]: mean={np.round(diff.mean(0), 1)} "
          f"std={np.round(diff.std(0), 1)}")
    # differential: commanded vs measured displacement between consecutive poses
    print("differential (commanded step -> measured cloud step) [mm]:")
    for a, b in zip(rows, rows[1:]):
        cmd = (np.asarray(b["off"]) - np.asarray(a["off"])) * 1000
        meas = (b["cen"] - a["cen"]) * 1000
        if np.linalg.norm(cmd) < 1:
            continue
        print(f"  cmd={np.round(cmd).astype(int)}  meas={np.round(meas, 1)}")


if __name__ == "__main__":
    main()
