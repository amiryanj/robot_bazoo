#!/usr/bin/env python
"""Ball-as-marker calibration sweep.

Grip the mini-basketball lightly (friction hold), move it through the workspace, and at
each pose record the 2 finger tags + the detected ball (pixel + 3D base) + the FK gripper
pose. The ball is rigid to the gripper, so detected-ball vs FK and vs the finger-tag
midpoint maps the residual detection/calibration error across the workspace -- a dataset
to refine the calibration later.

    python vision/ball_cal_collect.py            # grasp, sweep, record, release, land

Safe: light grip (stall + 2deg), FK-filtered poses, opens the gripper + lands on any exit.
"""
import itertools
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vision"))

HANDEYE = ROOT / "outputs/calib/handeye.json"
GRASP_SQUEEZE = 2.0          # deg past contact (user: 1-3deg, friction holds the ball)
# record grid: pan/lift/elbow deltas around the grasp pose (orientation kept ~constant so
# the finger tags stay camera-facing and the ball stays gripped); FK-filtered for safety.
D_PAN = (-20.0, 0.0, 20.0)
D_LIFT = (-10.0, 0.0, 12.0)
D_ELBOW = (-12.0, 12.0)
WS_X = (0.15, 0.40); WS_Y = (-0.20, 0.20); WS_Z = (0.02, 0.18)


def light_close(robot, pose, read_angles, MOTOR_NAMES):
    """Close until the jaws stall on the ball, then squeeze only GRASP_SQUEEZE deg."""
    cmd = dict(pose)
    for g in np.arange(pose["gripper"], 2.0, -3.0):
        cmd["gripper"] = float(g)
        robot.send_action({f"{n}.pos": cmd[n] for n in MOTOR_NAMES})
        time.sleep(0.15)
        meas = read_angles(robot)["gripper"]
        if meas - g > 6.0:                                # jaws stalled on the ball
            cmd["gripper"] = float(meas - GRASP_SQUEEZE)
            robot.send_action({f"{n}.pos": cmd[n] for n in MOTOR_NAMES})
            print(f"  grip: stalled at {meas:.0f}, holding {cmd['gripper']:.0f}")
            return cmd
    print("  WARNING: no clear contact -- ball may not be gripped")
    return cmd


def detect_finger_tags(color, depth, K, R_cb, t_cb):
    import cv2
    from handeye_calib import backproject
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), par)
    c, ids, _ = det.detectMarkers(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
    out = {}
    if ids is not None:
        for c4, tid in zip(c, ids.ravel()):
            if int(tid) in (1, 2):
                uv = c4[0].mean(0)
                pc = backproject(int(round(uv[0])), int(round(uv[1])), depth, K)
                if pc is not None:
                    out[int(tid)] = dict(px=uv.tolist(), base3d=(R_cb @ pc + t_cb).tolist())
    return out


def main():
    import cv2  # noqa
    from handeye_calib import Realsense
    from ball_yolo import ball_from_box
    from pick_ball import (Kin, BallDetector, localize_base, plan_waypoints, ik_best,
                           read_angles, move_to, GRIP_OPEN, GRASP_DEPTH, LIFT_CLEAR,
                           MOTOR_NAMES)
    from gamepad_utils import graceful_shutdown
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    he = json.load(open(HANDEYE)); R_cb, t_cb = np.array(he["R"]), np.array(he["t"])
    kin = Kin(); detector = BallDetector()

    # 1) detect the ball + plan the grasp
    p_ball, ball = localize_base(detector)
    if p_ball is None:
        sys.exit("ball not detected -- place it on the holder in view.")
    print(f"ball at {np.round(p_ball * 1000).astype(int)} mm")
    target = p_ball - np.array([0, 0, GRASP_DEPTH])
    above, grasp, e_g, tilt, e_a = plan_waypoints(kin, target)
    print(f"grasp IK err {e_g*1000:.1f}mm, tilt {tilt:.0f}deg")
    if e_g > 0.012:
        sys.exit("grasp IK too far -- aborting.")

    # 2) record poses: pan/lift/elbow grid around the grasp, FK-filtered
    rec = []
    for dp, dl, de in itertools.product(D_PAN, D_LIFT, D_ELBOW):
        ang = dict(grasp)
        ang["shoulder_pan"] += dp; ang["shoulder_lift"] += dl; ang["elbow_flex"] += de
        _, p = kin.fk(ang)
        if WS_X[0] < p[0] < WS_X[1] and WS_Y[0] < p[1] < WS_Y[1] and WS_Z[0] < p[2] < WS_Z[1]:
            rec.append(ang)
    print(f"{len(rec)} FK-checked record poses")

    robot = SOFollower(SOFollowerRobotConfig(port="/dev/ttyACM1", id="so101", cameras={}))
    robot.connect()
    cam = Realsense()
    records = []
    cur = read_angles(robot)
    try:
        # 3) grasp: approach open, descend, light-close, lift
        cur = move_to(robot, cur, dict(above, gripper=GRIP_OPEN), seconds=2.5)
        cur = move_to(robot, cur, dict(grasp, gripper=GRIP_OPEN), seconds=2.0)
        cur = light_close(robot, dict(grasp, gripper=GRIP_OPEN), read_angles, MOTOR_NAMES)
        grip = cur["gripper"]
        lift_ang, _, _ = ik_best(kin, target + np.array([0, 0, LIFT_CLEAR]))
        cur = move_to(robot, cur, dict(lift_ang, gripper=grip), seconds=1.5)

        # 4) sweep + record
        for i, ang in enumerate(rec):
            cur = move_to(robot, cur, dict(ang, gripper=grip), seconds=1.6)
            time.sleep(0.5)
            color, depth, K = cam.grab()
            meas = read_angles(robot)
            tags = detect_finger_tags(color, depth, K, R_cb, t_cb)
            box, score = detector.detect(color)
            ballp = None
            if box:
                b = ball_from_box(box, score, depth, K)
                if b is not None:
                    ballp = dict(px=[float(b["uv"][0]), float(b["uv"][1])],
                                 base3d=(R_cb @ b["center3d"] + t_cb).tolist(),
                                 radius=float(b["radius_m"]))
            _, t_w = kin.fk(meas)
            r = dict(angles=meas, tcp_fk=t_w.tolist(), tags=tags, ball=ballp)
            if ballp and len(tags) == 2:
                mid = (np.array(tags[1]["base3d"]) + np.array(tags[2]["base3d"])) / 2
                r["ball_minus_tagmid_mm"] = ((np.array(ballp["base3d"]) - mid) * 1000).tolist()
            records.append(r)
            e = r.get("ball_minus_tagmid_mm")
            print(f"  pose {i+1}/{len(rec)}: {len(tags)} tag(s), ball={'y' if ballp else 'n'}"
                  + (f", ball-tagmid={np.round(e).astype(int)}mm" if e else ""))

        # 5) release over the holder
        cur = move_to(robot, cur, dict(grasp, gripper=grip), seconds=2.0)
        robot.send_action({f"{n}.pos": dict(grasp, gripper=GRIP_OPEN)[n] for n in MOTOR_NAMES})
        time.sleep(0.6)
    except Exception as ex:
        print(f"\nError: {ex!r}\nopening gripper + landing...")
        try:
            robot.send_action({f"{n}.pos": dict(cur, gripper=GRIP_OPEN)[n] for n in MOTOR_NAMES})
            time.sleep(0.5)
        except Exception:
            pass
    finally:
        cam.stop(); graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception:
            pass

    out = ROOT / f"outputs/calib/ball_cal_{datetime.now():%Y%m%d_%H%M%S}.json"
    json.dump(dict(records=records, n=len(records), created=datetime.now().isoformat()),
              open(out, "w"), indent=2)
    print(f"\n{len(records)} records -> {out}")
    errs = np.array([r["ball_minus_tagmid_mm"] for r in records if "ball_minus_tagmid_mm" in r])
    if len(errs):
        print(f"ball-vs-tagmid over {len(errs)} poses: mean {np.round(errs.mean(0)).astype(int)}mm, "
              f"std {np.round(errs.std(0)).astype(int)}mm")


if __name__ == "__main__":
    main()
