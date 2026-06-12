#!/usr/bin/env python
"""Measure the wrist_roll mapping offset with an ArUco finger tag (full 6-DoF pose —
unlike a point marker, rotation about the roll axis is directly observable).

Sweeps wrist_roll (plus two wrist_flex levels for conditioning) while reading the
tag pose via PnP at 720p. Then jointly solves, exactly like the hand-eye calibration:

    unknowns: T_gripperbody->tag (6)  +  DELTA, the constant roll offset (1)
    per detection (6 eqs):  T_base->tag_meas  =  FK(q, roll+DELTA) ∘ T_gripperbody->tag

Seeded at DELTA ∈ {-90, 0, +90, 180}deg with the tag transform initialised exactly
from one sample per seed. Tag side length is MEASURED from depth corners (the print
scale is not assumed). Gripper opening stays fixed throughout (tag rigid w.r.t. the
gripper body).

    python vision/tag_sweep.py [--port /dev/ttyACM1]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

BASE_POSE = {"shoulder_pan": 0.0, "shoulder_lift": 22.0, "elbow_flex": 45.0,
             "wrist_flex": -40.0, "wrist_roll": 0.0, "gripper": 2.0}
SWEEPS = [(-40.0, np.arange(-100, 101, 8.0)),
          (-60.0, np.arange(-60, 41, 20.0)),
          (-20.0, np.arange(-60, 41, 20.0))]
MOTOR_NAMES = list(BASE_POSE)
W_POS = 10.0                       # 10 cm position error ~ 1 rad orientation error


def detect_tag(det_aruco, color, depth, K, backproject):
    import cv2
    corners, ids, _ = det_aruco.detectMarkers(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return None
    c = corners[0][0]
    # physical side from depth corners when available (don't trust the print scale)
    P = [backproject(int(u), int(v), depth, K, win=3) for u, v in c]
    P = [p for p in P if p is not None]
    side = None
    if len(P) == 4:
        P = np.array(P)
        side = float(np.mean([np.linalg.norm(P[i] - P[(i + 1) % 4]) for i in range(4)]))
    return int(ids.ravel()[0]), c, side


def pnp_pose(c, side, K):
    import cv2
    S = side
    obj = np.array([[-S / 2, S / 2, 0], [S / 2, S / 2, 0],
                    [S / 2, -S / 2, 0], [-S / 2, -S / 2, 0]], np.float32)
    Km = np.array([[K["fx"], 0, K["ppx"]], [0, K["fy"], K["ppy"]], [0, 0, 1]], np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, c.astype(np.float32), Km, np.zeros(5),
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    args = ap.parse_args()

    import cv2
    import json
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from handeye_calib import Realsense, backproject, make_fk
    from pick_ball import move_to, read_angles
    from gamepad_utils import graceful_shutdown
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    he = json.load(open(ROOT / "outputs/calib/handeye.json"))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])

    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(dic, par)

    fk = make_fk()
    cam = Realsense(color_res=(1280, 720))
    robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
    robot.connect()

    samples = []                   # (ang, R_bt, t_bt)
    sides = []
    try:
        cur = read_angles(robot)
        for wf, rolls in SWEEPS:
            for roll in rolls:
                tgt = dict(BASE_POSE, wrist_flex=wf, wrist_roll=float(roll))
                cur = move_to(robot, cur, tgt, seconds=0.8)
                time.sleep(0.45)
                color, depth, K = cam.grab()
                got = detect_tag(det, color, depth, K, backproject)
                if got is None:
                    print(f"  wf={wf:+.0f} roll={roll:+.0f}: no tag")
                    continue
                tid, corners, side = got
                if side:
                    sides.append(side)
                S = float(np.median(sides)) if sides else 0.0276
                pose = pnp_pose(corners, S, K)
                if pose is None:
                    continue
                R_ct, t_ct = pose
                R_bt, t_bt = R_cb @ R_ct, R_cb @ t_ct + t_cb
                ang = read_angles(robot)
                samples.append((ang, R_bt, t_bt))
                print(f"  wf={wf:+.0f} roll={roll:+.0f}: tag{tid} side={1000 * (side or 0):.1f}mm "
                      f"t={np.round(t_bt * 1000).astype(int)}")
    except Exception as e:
        print(f"\nUnexpected error: {e!r}\nLanding the arm safely...")
    finally:
        cam.stop()
        graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception:
            pass

    if len(samples) < 8:
        sys.exit(f"only {len(samples)} detections — not enough.")
    print(f"\n{len(samples)} tag poses, measured side={1000 * float(np.median(sides)):.1f}mm")

    def fk_pose(ang, delta_deg):
        a = dict(ang)
        a["wrist_roll"] = ang["wrist_roll"] + delta_deg
        return fk(a)

    def resid(x):
        R_gt = Rotation.from_rotvec(x[:3]).as_matrix()
        t_gt, delta = x[3:6], np.degrees(x[6])
        out = []
        for ang, R_bt, t_bt in samples:
            R_w, t_w = fk_pose(ang, delta)
            R_p, t_p = R_w @ R_gt, R_w @ t_gt + t_w
            out.append(Rotation.from_matrix(R_p.T @ R_bt).as_rotvec())
            out.append(W_POS * (t_bt - t_p))
        return np.concatenate(out)

    best = None
    for d0 in (-90.0, 0.0, 90.0, 180.0):
        ang0, R_bt0, t_bt0 = samples[0]
        R_w0, t_w0 = fk_pose(ang0, d0)
        R_gt0 = R_w0.T @ R_bt0                      # exact init from one sample
        t_gt0 = R_w0.T @ (t_bt0 - t_w0)
        x0 = np.concatenate([Rotation.from_matrix(R_gt0).as_rotvec(), t_gt0,
                             [np.radians(d0)]])
        sol = least_squares(resid, x0, method="lm", max_nfev=6000)
        rms = float(np.sqrt(np.mean(sol.fun ** 2)))
        print(f"  seed {d0:+4.0f}deg -> delta={np.degrees(sol.x[6]):+7.1f}deg  rms={rms:.4f}")
        if best is None or sol.cost < best[0].cost:
            best = (sol,)
    sol = best[0]
    delta = np.degrees(sol.x[6])
    err = sol.fun.reshape(-1, 2, 3)
    rot_rms = np.degrees(np.sqrt(np.mean(err[:, 0] ** 2)))
    pos_rms = np.sqrt(np.mean((err[:, 1] / W_POS) ** 2)) * 1000
    print(f"\nWRIST_ROLL OFFSET = {delta:+.2f} deg")
    print(f"residuals: rotation {rot_rms:.2f} deg rms, position {pos_rms:.1f} mm rms")
    print("(delta ~0 -> mapping fine; otherwise apply delta in one shared mapping layer "
          "and re-run the hand-eye calibration.)")


if __name__ == "__main__":
    main()
