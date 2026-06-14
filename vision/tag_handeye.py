#!/usr/bin/env python
"""Tag-based hand-eye: re-solve T_cam->base alone, using the mounted finger tags.

Context (2026-06-12): tag_calib.json (vision/tag_sweep.py v2) pinned the wrist_roll
mapping offset DELTA and the per-tag mounting transforms via a jaw sweep at ONE arm
pose — well-conditioned for DELTA, weakly conditioned for the camera pose it also
estimated (which disagrees with the heart-based handeye.json by 4-6 cm; the grasp
keeps missing on a mixed calibration). This tool holds DELTA + tag mounts FIXED and
re-solves ONLY T_cam->base (6 unknowns) over VARIED arm poses — the conditioning the
jaw sweep lacked. Residual = pixel reprojection of predicted tag corners (no PnP, so
no planar-pose ambiguity). Fully autonomous: scripted in-air poses, no gamepad.

    python vision/tag_handeye.py            # collect + solve + report (arm + camera)
    python vision/tag_handeye.py --apply    # additionally back up and update handeye.json
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

HANDEYE = ROOT / "outputs/calib/handeye.json"
TAG_CALIB = ROOT / "outputs/calib/tag_calib.json"
GRIP_CMD = 40.0                    # fixed gripper opening during collection

# pose grid: tags face the camera around gripper-horizontal, vary the whole arm
PANS = (-25.0, 0.0, 25.0)
LIFTS = (5.0, 22.0)
ELBOWS = (35.0, 55.0)
WFS = (-55.0, -40.0)
ROLLS = (-25.0, 10.0)
BOX_X = (0.14, 0.34)
BOX_Y = (-0.20, 0.20)
BOX_Z = (0.03, 0.18)
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def tag_corners_obj(side):
    s = side / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--apply", action="store_true", help="update handeye.json (backup kept)")
    args = ap.parse_args()

    import cv2
    import mujoco
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from handeye_calib import Realsense
    from pick_ball import Kin, move_to, read_angles
    from gamepad_utils import graceful_shutdown
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    tc = json.load(open(TAG_CALIB))
    delta = float(tc["delta_deg"])
    jaw_a, jaw_b0 = float(tc["jaw_scale_a"]), float(tc["jaw_b0"])
    tags = {int(k): dict(body=v["body"], R=np.array(v["R"]), t=np.array(v["t"]),
                         side=float(v["side_m"])) for k, v in tc["tags"].items()}
    he = json.load(open(HANDEYE))
    T0 = (np.array(he["R"]), np.array(he["t"]))
    T1 = (np.array(tc["R_cam2base"]), np.array(tc["t_cam2base"]))

    kin = Kin()                                       # fk applies delta to wrist_roll
    # Kin now sources wrist_roll from offset_calib.json (refines tag_calib by <1deg);
    # tolerate that drift -- this tool only needs a consistent roll within ~1deg.
    assert abs(kin.roll_delta - delta) < 2.0, "pick_ball.Kin roll far from tag_calib"
    bid = {tid: mujoco.mj_name2id(kin.m, mujoco.mjtObj.mjOBJ_BODY, t["body"])
           for tid, t in tags.items()}
    jaw_adr = kin.adr["gripper"]

    def body_pose(ang, tid):
        """Pose of the tag's parent body, with the jaw map applied to the gripper."""
        for j, a in kin.adr.items():
            off = kin.roll_delta if j == "wrist_roll" else 0.0
            kin.d.qpos[a] = math.radians(ang.get(j, 0.0) + off)
        kin.d.qpos[jaw_adr] = math.radians(jaw_a * ang.get("gripper", 0.0) + jaw_b0)
        mujoco.mj_forward(kin.m, kin.d)
        b = bid[tid]
        return kin.d.xmat[b].reshape(3, 3).copy(), kin.d.xpos[b].copy()

    # FK-filtered pose list
    poses = []
    for pan, lift, elbow, wf, roll in itertools.product(PANS, LIFTS, ELBOWS, WFS, ROLLS):
        ang = dict(shoulder_pan=pan, shoulder_lift=lift, elbow_flex=elbow,
                   wrist_flex=wf, wrist_roll=roll, gripper=GRIP_CMD)
        _, p = kin.fk(ang)
        if (BOX_X[0] <= p[0] <= BOX_X[1] and BOX_Y[0] <= p[1] <= BOX_Y[1]
                and BOX_Z[0] <= p[2] <= BOX_Z[1]):
            poses.append(ang)
    step = max(len(poses) // 16, 1)
    poses = poses[::step][:16]
    print(f"{len(poses)} FK-checked poses")

    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(dic, par)

    cam = Realsense(color_res=(1280, 720))
    robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
    robot.connect()
    samples = []                                       # (tid, ang, corners_px, K)
    try:
        cur = read_angles(robot)
        for i, pose in enumerate(poses):
            cur = move_to(robot, cur, pose, seconds=1.4)
            time.sleep(0.5)
            color, depth, K = cam.grab()
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
            ang = read_angles(robot)
            n = 0
            if ids is not None:
                for c4, tid in zip(corners, ids.ravel()):
                    if int(tid) in tags:
                        samples.append((int(tid), ang, c4[0].astype(float), K))
                        n += 1
            print(f"  pose {i + 1}/{len(poses)}: {n} tag(s)")
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
        sys.exit(f"only {len(samples)} tag sightings — not enough.")
    print(f"\n{len(samples)} tag sightings")
    sp = ROOT / "outputs/calib/tag_handeye_samples.json"
    json.dump([dict(tid=tid, ang=ang, px=px.tolist(), K=K)
               for tid, ang, px, K in samples], open(sp, "w"))
    print(f"samples -> {sp}")

    tids = sorted(tags)

    def project(Rcb, tcb, Rt, tt, side, K):
        P = (Rt @ tag_corners_obj(side).T).T + tt                     # corners in base
        Pc = (Rcb.T @ (P - tcb).T).T                                  # -> camera frame
        return np.stack([K["fx"] * Pc[:, 0] / Pc[:, 2] + K["ppx"],
                         K["fy"] * Pc[:, 1] / Pc[:, 2] + K["ppy"]], axis=1)

    def resid(x, free_mounts):
        Rcb = Rotation.from_rotvec(x[:3]).as_matrix()
        tcb = x[3:6]
        mounts = {}
        for k, tid in enumerate(tids):
            if free_mounts:
                v = x[6 + 6 * k:12 + 6 * k]
                mounts[tid] = (Rotation.from_rotvec(v[:3]).as_matrix(), v[3:6])
            else:
                mounts[tid] = (tags[tid]["R"], tags[tid]["t"])
        out = []
        for tid, ang, px, K in samples:
            R_b, t_b = body_pose(ang, tid)
            Rm, tm = mounts[tid]
            uv = project(Rcb, tcb, R_b @ Rm, R_b @ tm + t_b, tags[tid]["side"], K)
            out.append((uv - px).ravel())
        return np.concatenate(out)

    def solve(free_mounts, label):
        best = None
        for name, (R0, t0) in (("handeye", T0), ("tag_calib", T1)):
            x0 = list(Rotation.from_matrix(R0).as_rotvec()) + list(t0)
            if free_mounts:
                for tid in tids:
                    x0 += list(Rotation.from_matrix(tags[tid]["R"]).as_rotvec())
                    x0 += list(tags[tid]["t"])
            sol = least_squares(resid, np.array(x0), args=(free_mounts,),
                                loss="soft_l1", f_scale=3.0, max_nfev=8000)
            rms = float(np.sqrt(np.mean(sol.fun ** 2)))
            print(f"  [{label}] seed {name:9s}: px_rms={rms:.2f}")
            if best is None or sol.cost < best.cost:
                best = sol
        return best

    sol_fixed = solve(False, "mounts fixed")
    sol_free = solve(True, "mounts free ")
    sol = sol_free
    R_new = Rotation.from_rotvec(sol.x[:3]).as_matrix()
    t_new = sol.x[3:6]
    rms = float(np.sqrt(np.mean(sol.fun ** 2)))

    def diff(T):
        dR = np.degrees(np.arccos(np.clip((np.trace(T[0].T @ R_new) - 1) / 2, -1, 1)))
        return dR, np.linalg.norm(T[1] - t_new) * 1000

    print(f"\nT_cam->base re-solved (mounts free): px_rms={rms:.2f}")
    print(f"  t = {np.round(t_new * 1000).astype(int)} mm")
    print(f"  vs heart handeye : rot {diff(T0)[0]:.1f} deg, trans {diff(T0)[1]:.0f} mm")
    print(f"  vs tag_calib BA  : rot {diff(T1)[0]:.1f} deg, trans {diff(T1)[1]:.0f} mm")

    out = ROOT / "outputs/calib/handeye_tag.json"
    json.dump(dict(R=R_new.tolist(), t=t_new.tolist(), px_rms=rms,
                   n_sightings=len(samples), created=datetime.now().isoformat(),
                   method="tag reprojection, delta+mounts fixed from tag_calib"),
              open(out, "w"), indent=2)
    print(f"  saved {out}")

    if args.apply:
        backup = HANDEYE.with_name(f"handeye_backup_{datetime.now():%Y%m%d_%H%M%S}.json")
        backup.write_text(HANDEYE.read_text())
        he.update(R=R_new.tolist(), t=t_new.tolist(),
                  note=f"updated by tag_handeye {datetime.now():%Y-%m-%d %H:%M}, "
                       f"px_rms={rms:.2f}; previous in {backup.name}")
        json.dump(he, open(HANDEYE, "w"), indent=2)
        print(f"  handeye.json UPDATED (backup: {backup.name})")


if __name__ == "__main__":
    main()
