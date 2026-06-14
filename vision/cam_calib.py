#!/usr/bin/env python
"""The ONE camera-pose calibration path: solve / correct / verify T_cam->base.

Coordinate system: the robot BASE is the world origin. The camera pose T_cam->base is
the only thing that changes when you touch the camera. Two FIXED references on the table
let us recover it without an arm dance:
  - the white plate PLANE  -> level (pitch/roll) + height (z); a big flat surface, so
    this is rock-solid and is exactly what a single tag is bad at;
  - the desk ArUco TAG     -> in-plane position (x,y) + yaw; what the plane can't give.

Subcommands:
  level   - correct the CURRENT handeye.json in place: rotate it so the measured table
            plane becomes truly level (kills a tilted hand-eye). No arm motion. Backs up.
  check   - self-check the current hand-eye: table tilt, table height, and whether the
            detected ball's bottom sits ON the plate. Exit non-zero if it fails.

Run `check` after anything that touches the camera; run `level` to fix a tilt.
"""
import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

HANDEYE = ROOT / "outputs/calib/handeye.json"
ANCHOR = ROOT / "outputs/calib/desk_anchor.json"   # the fixed desk-tag reference, base frame
DESK_TAG_DICT = "DICT_4X4_50"                       # fixed desk tag is 4x4 id 13 (1,2 = fingers)
DESK_TAG_ID = 13
DESK_TAG_SIDE_M = 0.0274
# Tolerances for a trustworthy hand-eye (the self-check thresholds).
MAX_TILT_DEG = 2.0          # the flat level desk must map within this of horizontal
MAX_BALL_GAP_MM = 8.0       # ball bottom must sit within this of the support plane


def load_he():
    he = json.load(open(HANDEYE))
    return np.array(he["R"]), np.array(he["t"]), he


def save_he(R, t, base, note):
    backup = HANDEYE.with_name(f"handeye_backup_{datetime.now():%Y%m%d_%H%M%S}.json")
    backup.write_text(json.dumps(base, indent=2))
    out = dict(base)
    out["R"] = R.tolist()
    out["t"] = t.tolist()
    out["note"] = note
    HANDEYE.write_text(json.dumps(out, indent=2))
    print(f"  handeye.json updated (backup: {backup.name})")


def warm_grab(cam, n=8):
    """Discard the first frames (RealSense depth needs to stabilise) and return a good one."""
    for _ in range(n):
        color, depth, K = cam.grab()
    return color, depth, K


def table_plane(base_pts):
    """Dominant near-horizontal plane in the base frame -> (up-normal, centroid). None if
    no flat plane is found."""
    from cloud import extract_planes
    best = None
    for pl in extract_planes(base_pts[::4], max_planes=3):
        n = np.array(pl["n"], float)
        if n[2] < 0:
            n = -n
        if n[2] > 0.9 and (best is None or pl["centroid"][2] is not None):
            best = (n, np.array(pl["centroid"], float))
            break
    return best


def level_correct(R, t, base_pts):
    """Rotate (R,t) about the table centroid so the table normal becomes vertical.
    Pivoting about the centroid keeps the table at its measured height while removing the
    tilt. Returns (R', t', tilt_before_deg)."""
    pl = table_plane(base_pts)
    if pl is None:
        raise RuntimeError("no flat table plane found to level against")
    n, c = pl
    tilt = math.degrees(math.acos(min(n[2], 1.0)))
    up = np.array([0.0, 0.0, 1.0])
    axis = np.cross(n, up)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return R.copy(), t.copy(), tilt
    axis /= s
    ang = math.atan2(s, float(np.dot(n, up)))
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_fix = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)  # Rodrigues
    R_new = R_fix @ R
    t_new = R_fix @ (t - c) + c                       # pivot about the table centroid
    return R_new, t_new, tilt


def plate_in_cam(depth, K):
    """Dominant level plane in the CAMERA frame -> (normal toward camera, point-on-plane).
    Uses RANSAC on the cam-frame cloud; the normal is oriented to point back at the camera
    (so it is the 'up' direction in the world)."""
    from ball import WORKSPACE_Z
    from cloud import crop_z, extract_planes
    from cloud import deproject as dp
    pc = crop_z(dp(depth, K), WORKSPACE_Z)
    for pl in extract_planes(pc[::4], max_planes=3):
        n = np.array(pl["n"], float)
        if abs(n[2]) > 0.9:
            if n[2] > 0:                      # camera looks along +z; 'up' points back (-z)
                n = -n
            return n, np.array(pl["centroid"], float)
    return None


def detect_desk_tag(gray):
    """Return (center_px, corners_px[4,2]) for desk tag id 8, or None."""
    import cv2
    dic = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DESK_TAG_DICT))
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    corners, ids, _ = cv2.aruco.ArucoDetector(dic, par).detectMarkers(gray)
    if ids is None:
        return None
    for c, tid in zip(corners, ids.ravel()):
        if int(tid) == DESK_TAG_ID:
            cp = c[0].astype(float)
            return cp.mean(0), cp
    return None


def _ray(u, v, K):
    return np.array([(u - K["ppx"]) / K["fx"], (v - K["ppy"]) / K["fy"], 1.0])


def ray_to_plane(u, v, K, n, p0):
    """Pixel ray (from cam origin) intersected with plane (n, p0) -> 3D cam point.
    Robust z for the tag: uses the fitted plate plane, not noisy per-pixel depth."""
    d = _ray(u, v, K)
    return d * (float(n @ p0) / float(n @ d))


def tag_frame_cam(depth, K, tag):
    """Tag centre + in-plane x-axis in the CAMERA frame, both placed on the plate plane.
    Returns (origin_cam, xaxis_cam, up_cam) or None."""
    pl = plate_in_cam(depth, K)
    if pl is None:
        return None
    n, p0 = pl
    ctr_px, corn = tag
    o = ray_to_plane(ctr_px[0], ctr_px[1], K, n, p0)
    pc = np.array([ray_to_plane(u, v, K, n, p0) for u, v in corn])
    ex = ((pc[1] - pc[0]) + (pc[2] - pc[3])) / 2          # tag local +x (avg of both edges)
    ex = ex - (ex @ n) * n                                 # keep it in-plane
    return o, ex / np.linalg.norm(ex), n


def sense_tag(cam, n=12):
    """Average the desk-tag corners over n frames (kills per-frame corner jitter, which is
    what makes single-frame yaw noisy) and return (origin_cam, xaxis_cam, up_cam, depth, K)
    from the averaged corners + the plate fit."""
    import cv2
    acc, last = [], None
    for _ in range(n + 4):
        color, depth, K = cam.grab()
        tag = detect_desk_tag(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
        if tag is not None:
            acc.append(tag[1]); last = (depth, K)
    if not acc:
        raise RuntimeError(f"desk tag (id {DESK_TAG_ID}) not detected over {n} frames")
    corn = np.mean(acc, axis=0)
    depth, K = last
    tf = tag_frame_cam(depth, K, (corn.mean(0), corn))
    if tf is None:
        raise RuntimeError("no plate plane under the tag")
    return (*tf, depth, K, len(acc))


def _basis(zc, xc):
    """Right-handed orthonormal basis with given z and approximate x."""
    zc = zc / np.linalg.norm(zc)
    xc = xc - (xc @ zc) * zc
    xc = xc / np.linalg.norm(xc)
    return np.column_stack([xc, np.cross(zc, xc), zc])


def report(R, t):
    """Grab a warmed frame, return (metrics dict). Used by both level (after) and check."""
    from ball import WORKSPACE_Z
    from ball_yolo import ball_from_box
    from cloud import crop_z
    from cloud import deproject as dp
    from handeye_calib import Realsense
    from pick_ball import BallDetector

    cam = Realsense()
    try:
        color, depth, K = warm_grab(cam)
    finally:
        cam.stop()
    cam_pts = crop_z(dp(depth, K), WORKSPACE_Z)        # crop in cam-z, then to base
    base = (R @ cam_pts.T).T + t
    pl = table_plane(base)
    m = {}
    if pl is not None:
        n, c = pl
        m["tilt_deg"] = math.degrees(math.acos(min(n[2], 1.0)))
        m["table_z_mm"] = c[2] * 1000
    det = BallDetector()
    box, score = det.detect(color)
    if box and pl is not None:
        b = ball_from_box(box, score, depth, K)
        pb = R @ b["center3d"] + t
        m["ball_base_mm"] = np.round(pb * 1000).astype(int).tolist()
        m["ball_bottom_mm"] = (pb[2] - b["radius_m"]) * 1000
        m["ball_gap_mm"] = m["ball_bottom_mm"] - m["table_z_mm"]   # +above / -below plate
    return m


def pin_anchor(R, t):
    """Pin the desk tag's pose in the BASE frame from the current hand-eye. The tag is a
    FIXED reference; once pinned, a single frame recovers T_cam->base via `recal`."""
    from handeye_calib import Realsense
    cam = Realsense(color_res=(1280, 720))            # high-res for sharp tag corners
    try:
        warm_grab(cam)
        o_cam, ex_cam, _, _, _, ngood = sense_tag(cam)
    finally:
        cam.stop()
    print(f"  averaged tag over {ngood} frames")
    o_base = R @ o_cam + t
    x_base = R @ ex_cam
    x_base[2] = 0.0                                    # tag lies flat -> x-axis is horizontal
    x_base /= np.linalg.norm(x_base)
    ANCHOR.write_text(json.dumps({
        "origin_base": o_base.tolist(), "xaxis_base": x_base.tolist(),
        "side_m": DESK_TAG_SIDE_M, "pinned": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))
    print(f"  anchor pinned: tag at base {np.round(o_base * 1000).astype(int)}mm  "
          f"yaw-x {np.round(x_base, 3)} -> {ANCHOR.name}")


def recal_from_refs():
    """Single-frame T_cam->base from the fixed references: plate plane (up + z) and the
    desk tag (x/y + yaw). No arm motion. Requires a pinned anchor."""
    from handeye_calib import Realsense
    if not ANCHOR.exists():
        raise RuntimeError(f"no anchor yet — run `cam_calib.py anchor` once first")
    a = json.load(open(ANCHOR))
    o_base, x_base = np.array(a["origin_base"]), np.array(a["xaxis_base"])
    cam = Realsense(color_res=(1280, 720))
    try:
        warm_grab(cam)
        o_cam, ex_cam, up_cam, _, _, _ = sense_tag(cam)
    finally:
        cam.stop()
    Bc = _basis(up_cam, ex_cam)                        # cam-frame reference basis
    Bb = _basis(np.array([0.0, 0.0, 1.0]), x_base)     # base-frame reference basis
    R = Bb @ Bc.T                                      # maps cam vectors -> base
    t = o_base - R @ o_cam
    return R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["level", "check", "anchor", "recal"])
    args = ap.parse_args()

    from ball import WORKSPACE_Z
    from cloud import crop_z
    from cloud import deproject as dp
    from handeye_calib import Realsense

    if args.cmd == "level":
        R, t, base_doc = load_he()
        cam = Realsense()
        try:
            color, depth, K = warm_grab(cam)
        finally:
            cam.stop()
        base_pts = (R @ dp(depth, K).T).T + t
        from ball import WORKSPACE_Z as WZ
        # keep only points near the table height band so the plane fit is the desk
        keep = (base_pts[:, 2] > -0.10) & (base_pts[:, 2] < 0.05)
        R2, t2, tilt = level_correct(R, t, base_pts[keep])
        print(f"  table tilt before leveling: {tilt:.1f} deg")
        save_he(R2, t2, base_doc, f"leveled against table plane {datetime.now():%Y-%m-%d %H:%M}; "
                                  f"was {tilt:.1f}deg tilted")
        m = report(R2, t2)
        print(f"  after: tilt={m.get('tilt_deg', float('nan')):.1f}deg  "
              f"table_z={m.get('table_z_mm', float('nan')):.0f}mm  "
              f"ball_gap={m.get('ball_gap_mm', float('nan')):.0f}mm "
              f"(ball bottom vs plate; ~0 = resting)")
        return 0

    if args.cmd == "anchor":
        R, t, _ = load_he()
        pin_anchor(R, t)
        return 0

    if args.cmd == "recal":
        _, _, base_doc = load_he()
        R, t = recal_from_refs()
        save_he(R, t, base_doc, f"recal from desk-tag + plate refs {datetime.now():%Y-%m-%d %H:%M}")
        m = report(R, t)
        print(f"  after recal: tilt={m.get('tilt_deg', float('nan')):.1f}deg  "
              f"table_z={m.get('table_z_mm', float('nan')):.0f}mm  "
              f"ball_gap={m.get('ball_gap_mm', float('nan')):+.0f}mm")
        return 0

    if args.cmd == "check":
        R, t, _ = load_he()
        m = report(R, t)
        if "tilt_deg" not in m:
            print("FAIL: no table plane detected"); return 1
        ok = m["tilt_deg"] < MAX_TILT_DEG
        line = f"table tilt {m['tilt_deg']:.1f}deg (<{MAX_TILT_DEG})  z {m['table_z_mm']:.0f}mm"
        if "ball_gap_mm" in m:
            bg = abs(m["ball_gap_mm"]) < MAX_BALL_GAP_MM
            ok &= bg
            line += f"  ball {m['ball_base_mm']}mm gap {m['ball_gap_mm']:+.0f}mm (<{MAX_BALL_GAP_MM})"
        print(("OK   " if ok else "FAIL ") + line)
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
