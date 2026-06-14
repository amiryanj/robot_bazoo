#!/usr/bin/env python
"""Headless one-shot geometry audit — no arm motion, just measure and draw.

Grabs ONE aligned Realsense frame + an IMU burst and reports, all in the BASE frame:
  - the ball (depth -> base z) and its sphere-fit inliers,
  - the RANSAC support planes (white plate / desk) z heights,
  - the static desk tag (id 8) measured z (PnP + depth) vs the pinned zp in desk_tag.json,
  - the two finger tags (id 1,2) MEASURED z (PnP, depth-free) vs FK-PREDICTED z
    (the cyan boxes in scene_debug) -> separates a depth bias from an FK/offset bias,
  - the IMU gravity vector + wrist tilt.

Writes annotated.png (camera view) + side_xz.png / side_yz.png (base-frame z stacking)
+ report.txt to outputs/calib/twin_measure_<ts>/. Reads the arm only to get joint
angles, with torque DISABLED on connect (limp, like scene_debug) -- nothing moves.

    python vision/twin_measure.py [--no-arm] [--port /dev/ttyACM1]
"""
import argparse
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
sys.path.insert(0, str(ROOT / "ESP32"))

HANDEYE = ROOT / "outputs/calib/handeye.json"
TAG_CALIB = ROOT / "outputs/calib/tag_calib.json"
DESK_TAG = ROOT / "outputs/calib/desk_tag.json"
XML = str(ROOT / "SO-ARM100/Simulation/SO101/scene.xml")
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def tag_corners_obj(side):
    s = side / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], float)


def read_imu_gravity(n=300):
    """Average an IMU burst -> gravity vector (m/s^2) in the IMU frame + tilt angles."""
    from imu_serial import stream_samples, SCALE
    xs, ys, zs = [], [], []
    gen = stream_samples()
    for _, x, y, z in gen:
        xs.append(x); ys.append(y); zs.append(z)
        if len(xs) >= n:
            break
    g = np.array([np.mean(xs), np.mean(ys), np.mean(zs)]) * SCALE
    mag = float(np.linalg.norm(g))
    gh = g / mag
    # tilt of the IMU z-axis from true vertical, plus pitch/roll about x/y
    tilt = math.degrees(math.acos(min(1.0, abs(gh[2]))))
    pitch = math.degrees(math.atan2(gh[0], gh[2]))
    roll = math.degrees(math.atan2(gh[1], gh[2]))
    return g, mag, tilt, pitch, roll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--no-arm", action="store_true")
    args = ap.parse_args()

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mujoco
    import torch
    from ball import WORKSPACE_Z
    from ball_yolo import BALL_RADIUS_M, ball_from_box
    from cloud import crop_z, fit_sphere_known_r, extract_planes
    from cloud import deproject as cloud_deproject
    from handeye_calib import Realsense
    from pick_ball import BallDetector, JOINT_OFFSETS

    he = json.load(open(HANDEYE))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])

    def to_base(p):
        return (R_cb @ np.asarray(p).T).T + t_cb

    out = ROOT / f"outputs/calib/twin_measure_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    report = []

    def log(s):
        print(s)
        report.append(s)

    # ── IMU first (separate port, never moves) ────────────────────────────────
    try:
        g, gmag, tilt, pitch, roll = read_imu_gravity()
        log("IMU gravity (m/s^2, IMU frame): "
            f"[{g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f}]  |g|={gmag:.2f}")
        log(f"IMU tilt from vertical: {tilt:.1f} deg   (pitch {pitch:+.1f}, roll {roll:+.1f})")
    except Exception as e:
        log(f"IMU read failed: {e!r}")

    # ── arm angles (limp) ─────────────────────────────────────────────────────
    ang = None
    if not args.no_arm:
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
        robot.connect()
        robot.bus.disable_torque()
        log("Arm torque DISABLED — reading angles only, nothing moves.")
        obs = robot.get_observation()
        ang = {n: float(obs.get(f"{n}.pos", 0.0)) for n in MOTOR_NAMES}
        log("joint angles (deg): "
            + "  ".join(f"{n[:5]}={ang[n]:+.1f}" for n in MOTOR_NAMES))
        try:
            robot.disconnect()
        except Exception:
            pass

    # ── camera frame ──────────────────────────────────────────────────────────
    cam = Realsense(color_res=(1280, 720))
    color, depth, K = cam.grab()
    cam.stop()
    log(f"frame: color {color.shape[1]}x{color.shape[0]}  "
        f"K(fx={K['fx']:.0f} fy={K['fy']:.0f} ppx={K['ppx']:.0f} ppy={K['ppy']:.0f})")
    Kmat = np.array([[K["fx"], 0, K["ppx"]], [0, K["fy"], K["ppy"]], [0, 0, 1]])
    vis = color.copy()

    # ── ball ──────────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    box, conf = BallDetector(device).detect(color)
    p_ball = None
    if box:
        b = ball_from_box(box, conf, depth, K)
        if b:
            p_ball = to_base(b["center3d"])
            log(f"BALL  base z = {p_ball[2] * 1000:+6.1f} mm   "
                f"(x {p_ball[0] * 1000:+.0f}, y {p_ball[1] * 1000:+.0f})  conf {conf:.2f}  "
                f"r {b['radius_m'] * 1000:.1f}mm")
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), (60, 220, 60), 2)
        cv2.putText(vis, f"ball z={p_ball[2]*1000:+.0f}mm" if p_ball is not None else "ball",
                    (box[0], box[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 220, 60), 2)
    else:
        log("BALL  not detected")

    # ── support planes ──────────────────────────────────────────────────────────
    pc = crop_z(cloud_deproject(depth, K), WORKSPACE_Z)
    base_pc = to_base(pc)
    planes = [pl for pl in extract_planes(base_pc[::4], max_planes=3)
              if abs(pl["n"][2]) > 0.95]
    plane_z = []
    for i, pl in enumerate(planes):
        z = float(pl["centroid"][2])
        plane_z.append(z)
        log(f"PLANE {i}: base z = {z * 1000:+6.1f} mm   "
            f"(n_z {pl['n'][2]:+.3f}, {len(pl['inliers']) if 'inliers' in pl else '?'} pts)")

    # ── tags ──────────────────────────────────────────────────────────────────
    def tag_center_pnp(corners_px, side):
        ok, rvec, tvec = cv2.solvePnP(tag_corners_obj(side), corners_px.astype(float),
                                      Kmat, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        return tvec.ravel() if ok else None  # tag center in camera frame

    def depth_at(uv):
        u, v = int(round(uv[0])), int(round(uv[1]))
        win = depth[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3]
        win = win[win > 0]
        if win.size == 0:
            return None
        z = float(np.median(win))
        ray = np.array([(uv[0] - K["ppx"]) / K["fx"], (uv[1] - K["ppy"]) / K["fy"], 1.0])
        return to_base(z * ray)

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    # finger tags: DICT_4X4_50 ids 1,2
    dic4 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det4 = cv2.aruco.ArucoDetector(dic4, par)
    corners, ids, _ = det4.detectMarkers(gray)
    tc = json.load(open(TAG_CALIB))
    tag_sides = {int(k): float(v["side_m"]) for k, v in tc["tags"].items()}
    meas_finger = {}
    if ids is not None:
        for c4, tid in zip(corners, ids.ravel()):
            tid = int(tid)
            poly = c4[0].astype(int)
            cv2.polylines(vis, [poly], True, (230, 230, 40), 2)
            if tid in tag_sides:
                tvec = tag_center_pnp(c4[0], tag_sides[tid])
                dep = depth_at(c4[0].mean(0))                  # depth at tag centre (reliable z)
                if tvec is not None:
                    pb = to_base(tvec)
                    meas_finger[tid] = pb
                    dz = f"{dep[2]*1000:+.1f}" if dep is not None else "n/a"
                    log(f"FINGER tag {tid}: PnP base z = {pb[2] * 1000:+6.1f} mm   "
                        f"DEPTH base z = {dz} mm   (x {pb[0]*1000:+.0f}, y {pb[1]*1000:+.0f})")
                    if dep is not None:
                        meas_finger[tid] = dep                 # prefer depth for the side-view
                    cv2.putText(vis, f"t{tid} z={pb[2]*1000:+.0f}", tuple(poly[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 40), 2)

    # desk tag: DICT_ARUCO_MIP_36H12 id 8
    desk = json.load(open(DESK_TAG))
    desk_side = float(desk["side_m"])
    dicd = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, "DICT_ARUCO_MIP_36h12", cv2.aruco.DICT_4X4_50))
    detd = cv2.aruco.ArucoDetector(dicd, par)
    dcorn, dids, _ = detd.detectMarkers(gray)
    desk_meas = None
    if dids is not None:
        for c4, tid in zip(dcorn, dids.ravel()):
            if int(tid) != 8:
                continue
            poly = c4[0].astype(int)
            cv2.polylines(vis, [poly], True, (230, 40, 230), 2)
            tvec = tag_center_pnp(c4[0], desk_side)
            ctr_px = c4[0].mean(0)
            pnp = to_base(tvec) if tvec is not None else None
            dep = depth_at(ctr_px)
            desk_meas = pnp
            log(f"DESK tag 8  PnP base z = {pnp[2]*1000:+.1f} mm   "
                f"depth base z = {dep[2]*1000:+.1f} mm   "
                f"(pinned zp = {desk['zp']*1000:+.0f} mm)"
                if pnp is not None and dep is not None else "DESK tag 8 partial")
            cv2.putText(vis, f"desk8 z={pnp[2]*1000:+.0f}" if pnp is not None else "desk8",
                        tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 40, 230), 2)
    if desk_meas is None:
        log(f"DESK tag 8 not detected (pinned zp = {desk['zp']*1000:+.0f} mm)")

    # ── FK-predicted finger tags (the cyan boxes) ─────────────────────────────
    fk_finger = {}
    if ang is not None:
        m = mujoco.MjModel.from_xml_path(XML)
        d = mujoco.MjData(m)
        adr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
               for j in MOTOR_NAMES}
        for j, a in adr.items():
            d.qpos[a] = math.radians(ang[j] + JOINT_OFFSETS.get(j, 0.0))
        mujoco.mj_forward(m, d)
        for tid, t in tc["tags"].items():
            tid = int(tid)
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, t["body"])
            bR, bt = d.xmat[bid].reshape(3, 3), d.xpos[bid]
            pb = bR @ np.array(t["t"]) + bt
            fk_finger[tid] = pb
            line = f"FINGER tag {tid} FK-PRED base z = {pb[2] * 1000:+6.1f} mm"
            if tid in meas_finger:
                gap = (meas_finger[tid] - pb) * 1000
                line += (f"   gap(meas-FK) = [{gap[0]:+.0f} {gap[1]:+.0f} {gap[2]:+.0f}] mm"
                         f"  |{np.linalg.norm(gap):.0f}|")
            log(line)

    # ── summary deltas ────────────────────────────────────────────────────────
    log("")
    if p_ball is not None and plane_z:
        log(f">> ball-to-nearest-plane z gap = "
            f"{(p_ball[2] - min(plane_z, key=lambda z: abs(z - p_ball[2]))) * 1000:+.1f} mm "
            f"(should be ~ +{BALL_RADIUS_M*1000:.0f} mm: ball center one radius above the plane)")
    if desk_meas is not None and plane_z:
        log(f">> desk-tag-to-nearest-plane z gap = "
            f"{(desk_meas[2] - min(plane_z, key=lambda z: abs(z - desk_meas[2]))) * 1000:+.1f} mm "
            "(tag lies ON a plane -> should be ~0)")

    # ── visuals ───────────────────────────────────────────────────────────────
    cv2.imwrite(str(out / "annotated.png"), vis)

    def side_view(ax, ai, title):
        ax.scatter(base_pc[::40, ai], base_pc[::40, 2] * 1000, s=1,
                   c="0.7", label="cloud")
        for z in plane_z:
            ax.axhline(z * 1000, color="goldenrod", lw=1.2, alpha=0.8)
        if p_ball is not None:
            ax.scatter(p_ball[ai], p_ball[2] * 1000, s=80, c="orange",
                       edgecolors="k", label="ball", zorder=5)
        for tid, pb in meas_finger.items():
            ax.scatter(pb[ai], pb[2] * 1000, s=60, c="cyan", marker="s",
                       edgecolors="k", zorder=5, label=f"tag{tid} meas")
        for tid, pb in fk_finger.items():
            ax.scatter(pb[ai], pb[2] * 1000, s=60, c="green", marker="x",
                       zorder=5, label=f"tag{tid} FK")
        if desk_meas is not None:
            ax.scatter(desk_meas[ai], desk_meas[2] * 1000, s=80, c="magenta",
                       marker="D", edgecolors="k", zorder=5, label="desk8")
        ax.axhline(desk["zp"] * 1000, color="magenta", ls=":", lw=1, alpha=0.6)
        ax.set_xlabel(f"base {'x' if ai == 0 else 'y'} (m)")
        ax.set_ylabel("base z (mm)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)

    for ai, name in ((0, "side_xz"), (1, "side_yz")):
        fig, axp = plt.subplots(figsize=(7, 5))
        side_view(axp, ai, f"base {'X' if ai == 0 else 'Y'}-Z (goldenrod=plane, "
                            "dotted magenta=pinned zp)")
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=110)
        plt.close(fig)

    (out / "report.txt").write_text("\n".join(report) + "\n")
    log(f"\nwrote: {out}/  (annotated.png, side_xz.png, side_yz.png, report.txt)")


if __name__ == "__main__":
    main()
