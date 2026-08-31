#!/usr/bin/env python
"""Overlay the base frame onto a camera frame to eyeball the handeye, and refine the
camera-yaw against the static desk tag (id224) + plate.

The plate/tag are mounted square to the robot, so a grid that doesn't line up with the
plate borders = a yaw error in T_cb (the DOF shoulder_pan was absorbing). This tool
estimates that yaw from id224's x-parallel edges, rotates the drawn frame to match, and
writes the yaw-corrected T_cb to handeye_yawfix.json (never overwrites handeye.json).

    python vision/calib_overlay.py --park              # auto-yaw from the tag
    python vision/calib_overlay.py --park --yaw 6.5    # force a manual yaw nudge (deg)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vision"))
from realsense import Realsense, backproject

ZP = -0.030                     # plate height in base frame (m), ~ -29mm
# park swung to the +y side (pan=-60): TCP ~(0.21, 0.31, 0.14), ~350mm off the desk tag
# and clear of the camera->tag sightline, so the hand never occludes id224.
PARK = dict(shoulder_pan=-60, shoulder_lift=15, elbow_flex=25, wrist_flex=-40,
            wrist_roll=0, gripper=30)


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def proj(P, R, t, K, RZ):
    Pc = R.T @ (RZ @ np.asarray(P, float) - t)
    if Pc[2] <= 1e-6:
        return None
    return (int(round(K["fx"] * Pc[0] / Pc[2] + K["ppx"])),
            int(round(K["fy"] * Pc[1] / Pc[2] + K["ppy"])))


def _params():
    par = cv2.aruco.DetectorParameters()
    par.minMarkerPerimeterRate = 0.01            # the static tag is ~32px (small)
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    par.adaptiveThreshWinSizeMin = 3
    par.adaptiveThreshWinSizeMax = 23
    return par


def detect(color, dname, want=None):
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dname)), _params())
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    out = []
    c, ids, _ = det.detectMarkers(gray)
    if ids is not None:
        for c4, tid in zip(c, ids.ravel()):
            if want is None or int(tid) in want:
                out.append((int(tid), c4[0]))
    if out or not want:
        return out
    # fallback: 2x upscale (helps the small 36H12 tag)
    big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    c, ids, _ = det.detectMarkers(big)
    if ids is not None:
        for c4, tid in zip(c, ids.ravel()):
            if int(tid) in want:
                out.append((int(tid), c4[0] / 2.0))
    return out


DESK_ID = 8                     # the desk tag, DICT_ARUCO_MIP_36H12 (stable, std 0.2px)


def static_corners(color, want_id=DESK_ID):
    """Corners of the desk tag (DICT_ARUCO_MIP_36H12 id 8). Filtering by id avoids the
    occasional 36H12 false positives; 2x-upscale fallback for the small (~32px) tag."""
    det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_MIP_36H12), _params())
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    big = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for im, sc in [(gray, 1.0), (big, 2.0)]:
        c, ids, _ = det.detectMarkers(im)
        if ids is not None:
            for cc, t in zip(c, ids.ravel()):
                if int(t) == want_id:
                    return cc[0] / sc
    return None


def ray_plate(u, v, R, t, K, zp=ZP):
    """Cast pixel (u,v) onto the plate plane z=zp in base frame (no depth needed)."""
    d = R @ np.array([(u - K["ppx"]) / K["fx"], (v - K["ppy"]) / K["fy"], 1.0])
    if abs(d[2]) < 1e-9:
        return None
    s = (zp - t[2]) / d[2]
    return t + s * d if s > 0 else None


def tag_yaw(color, K, R, t):
    """Yaw (rad) of the desk tag's x-parallel edges vs base +x, in the current frame."""
    crn = static_corners(color)
    if crn is None:
        return None
    base = [ray_plate(u, v, R, t, K) for u, v in crn]
    if any(b is None for b in base):
        return None
    angs = []
    for i in range(4):
        e = base[(i + 1) % 4] - base[i]
        if abs(e[0]) > abs(e[1]):                       # an ~x-parallel edge
            a = math.atan2(e[1], e[0])
            if a > math.pi / 2: a -= math.pi
            elif a < -math.pi / 2: a += math.pi
            angs.append(a)
    return float(np.mean(angs)) if angs else None


def _run(robot, yaw_arg):
    cam = Realsense(color_res=(1280, 720))
    color, depth, K = cam.grab()
    for _ in range(8):                                  # grab until the desk tag is seen
        if static_corners(color) is not None:
            break
        color, depth, K = cam.grab()
    cam.stop()
    he = json.load(open(ROOT / "outputs/calib/handeye.json"))
    R, t = np.array(he["R"]), np.array(he["t"])

    if yaw_arg is not None:
        yaw = math.radians(yaw_arg)
        src = "manual"
    else:
        y = tag_yaw(color, K, R, t)
        if y is None:
            print("  WARNING: id224 not usable this frame — yaw=0")
        yaw = y or 0.0
        src = "auto from desk tag"
    RZ = Rz(yaw)
    print(f"applied yaw = {math.degrees(yaw):+.2f} deg ({src})")

    img = color.copy()

    def pj(P):
        return proj(P, R, t, K, RZ)

    for x in np.arange(0, 0.46, 0.05):                  # plate grid (5 cm)
        a, b = pj((x, -0.10, ZP)), pj((x, 0.25, ZP))
        if a and b: cv2.line(img, a, b, (70, 70, 70), 1)
    for y in np.arange(-0.10, 0.26, 0.05):
        a, b = pj((0, y, ZP)), pj((0.45, y, ZP))
        if a and b: cv2.line(img, a, b, (70, 70, 70), 1)

    a, b = pj((0, 0, ZP)), pj((0.45, 0, ZP))            # pan=0 / y=0 line
    if a and b:
        cv2.line(img, a, b, (0, 255, 255), 2)
        cv2.putText(img, "pan=0 / y=0", (b[0] - 150, b[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    for x in np.arange(0.1, 0.46, 0.1):
        p = pj((x, 0, ZP))
        if p: cv2.putText(img, f"{int(x*100)}", (p[0] - 8, p[1] + 16),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)

    O = pj((0, 0, 0))                                   # base origin + axes
    for ax, col, lbl in [((0.15, 0, 0), (0, 0, 255), "+x"),
                         ((0, 0.12, 0), (0, 255, 0), "+y"),
                         ((0, 0, 0.12), (255, 0, 0), "+z")]:
        pe = pj(ax)
        if O and pe:
            cv2.arrowedLine(img, O, pe, col, 2, tipLength=0.18)
            cv2.putText(img, lbl, pe, cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    if O:
        cv2.circle(img, O, 5, (0, 255, 255), -1)
        cv2.putText(img, "base O", (O[0] + 7, O[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    for tid, poly in detect(color, "DICT_4X4_50", None):           # finger tags
        p = poly.astype(int)
        cv2.polylines(img, [p], True, (255, 140, 0), 2)
        cv2.putText(img, f"id{tid}", (p.mean(0).astype(int)[0] - 12, p.mean(0).astype(int)[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)
    crn = static_corners(color)                                    # desk tag
    if crn is not None:
        p = crn.astype(int)
        cv2.polylines(img, [p], True, (255, 0, 255), 2)
        cv2.putText(img, "desk tag", (p.mean(0).astype(int)[0] - 20, p.mean(0).astype(int)[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    out = ROOT / "outputs/calib/calib_overlay.png"
    cv2.imwrite(str(out), img)
    print(f"saved {out}  (grid=5cm; numbers=cm along +x; yellow=pan=0/y=0)")

    # write the yaw-corrected T_cb candidate (does NOT overwrite handeye.json)
    R_new, t_new = Rz(-yaw) @ R, Rz(-yaw) @ t
    fix = ROOT / "outputs/calib/handeye_yawfix.json"
    json.dump(dict(R=R_new.tolist(), t=t_new.tolist(), yaw_applied_deg=math.degrees(yaw),
                   source=src, note="handeye.json with camera-yaw corrected vs id224/plate"),
              open(fix, "w"), indent=2)
    print(f"yaw-corrected T_cb -> {fix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--park", action="store_true", help="curl the arm up out of frame first")
    ap.add_argument("--yaw", type=float, default=None, help="manual yaw in deg (else auto)")
    ap.add_argument("--port", default="/dev/ttyACM1")
    args = ap.parse_args()

    robot = None
    if args.park:
        from pick_ball import move_to, read_angles
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
        robot.connect()
        move_to(robot, read_angles(robot), PARK, seconds=2.0)
        time.sleep(0.6)
    try:
        _run(robot, args.yaw)
    finally:
        if robot is not None:
            from gamepad_utils import graceful_shutdown
            graceful_shutdown(robot)
            try:
                robot.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    main()
