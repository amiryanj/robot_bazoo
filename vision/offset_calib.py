#!/usr/bin/env python
"""Joint-zero offset calibration as a GTSAM factor graph.

LeRobot pins each joint's zero to an eyeballed homing pose, so the digital twin
carries a constant per-joint offset delta_i between the encoder angle and the true
kinematic angle. This identifies all six deltas so MuJoCo FK matches the real arm.

Factor graph (see outputs/calib/factor_graph.html for the picture):
  variables  delta (Vector6), T_cam->base (Pose3), tag mounts M_A,M_B (Pose3)
  factors    tag reprojection per sighting   {delta, T_cb, M_tid}
             (optional) IMU gravity per pose  {delta, R_imu}
  priors     T_cb  <- handeye.json (hand-eye)
             mounts <- tag_calib.json, tight on rotation  == "tag square to finger"
                       (assumption B: this is what breaks the wrist_roll/mount confound;
                        a tag-only graph leaves delta_5 in a 1-DOF gauge with the mount)
             delta  <- loose, seeded with the known wrist_roll mapping (~ -85 deg)

The marginal covariance on delta is the payoff: an unobservable joint (e.g. delta_6
with the gripper held fixed, or delta_5 with a loose mount prior) shows up as a huge
sigma instead of a wrong-but-confident number.

    python vision/offset_calib.py --selftest     # synthetic, no hardware
    python vision/offset_calib.py --collect       # drive the arm, record samples
    python vision/offset_calib.py                 # solve from recorded samples
    python vision/offset_calib.py --imu           # also use IMU factors (co-est. mount)
"""
import argparse
import itertools
import json
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))
sys.path.insert(0, str(ROOT / "ESP32"))

HANDEYE = ROOT / "outputs/calib/handeye.json"
TAG_CALIB = ROOT / "outputs/calib/tag_calib.json"
SAMPLES = ROOT / "outputs/calib/offset_calib_samples.json"
RESULT = ROOT / "outputs/calib/offset_calib.json"

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM5 = JOINTS[:5]                       # gripper handled via the jaw map
WRIST_BODY = "gripper"                  # the wrist_roll link (tag A + the IMU live here)

# autonomous collection pose grid (mirrors tag_handeye.py; tags face the camera)
PANS = (-25.0, 0.0, 25.0); LIFTS = (5.0, 22.0); ELBOWS = (35.0, 55.0)
WFS = (-55.0, -40.0); ROLLS = (-25.0, 10.0); GRIP_CMD = 40.0
BOX_X = (0.14, 0.34); BOX_Y = (-0.20, 0.20); BOX_Z = (0.03, 0.18)


def corners_obj(side):
    s = side / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], float)


class Model:
    """MuJoCo FK with a free per-joint offset vector applied on top of the encoder angle."""

    def __init__(self):
        import mujoco
        from pick_ball import Kin
        self.mj = mujoco
        self.kin = Kin()
        tc = json.load(open(TAG_CALIB))
        self.jaw_a, self.jaw_b0 = float(tc["jaw_scale_a"]), float(tc["jaw_b0"])
        self.tags = {int(k): dict(body=v["body"],
                                  R=np.array(v["R"]), t=np.array(v["t"]),
                                  side=float(v["side_m"])) for k, v in tc["tags"].items()}
        self.bid = {tid: mujoco.mj_name2id(self.kin.m, mujoco.mjtObj.mjOBJ_BODY, t["body"])
                    for tid, t in self.tags.items()}
        self.wrist_bid = mujoco.mj_name2id(self.kin.m, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY)
        self.delta_seed = float(tc["delta_deg"])      # known wrist_roll mapping (~ -85)

    def _set(self, ang, delta):
        k = self.kin
        for idx, j in enumerate(ARM5):
            k.d.qpos[k.adr[j]] = math.radians(ang[j] + math.degrees(delta[idx]))
        k.d.qpos[k.adr["gripper"]] = math.radians(self.jaw_a * ang["gripper"] + self.jaw_b0) + delta[5]
        self.mj.mj_forward(k.m, k.d)

    def body_pose(self, ang, delta, tid):
        self._set(ang, delta)
        b = self.bid[tid]
        return self.kin.d.xmat[b].reshape(3, 3).copy(), self.kin.d.xpos[b].copy()

    def wrist_rot(self, ang, delta):
        self._set(ang, delta)
        return self.kin.d.xmat[self.wrist_bid].reshape(3, 3).copy()


# ── projection ────────────────────────────────────────────────────────────────────
def project(T_cb, R_tag, t_tag, side, K):
    P = (R_tag @ corners_obj(side).T).T + t_tag                 # corners in base
    Rcb, tcb = T_cb[:3, :3], T_cb[:3, 3]
    Pc = (Rcb.T @ (P - tcb).T).T                                # -> camera frame
    return np.stack([K["fx"] * Pc[:, 0] / Pc[:, 2] + K["ppx"],
                     K["fy"] * Pc[:, 1] / Pc[:, 2] + K["ppy"]], axis=1)


# ── gtsam graph ─────────────────────────────────────────────────────────────────
def build_and_solve(model, poses, sightings, T_cb0, mounts0, use_imu=False,
                    mount_rot_sigma_deg=5.0, tcb_rot_sigma_deg=2.0, tcb_t_sigma_m=0.01,
                    desk=None, verbose=True):
    """poses: list of dict(ang=..., imu=unit3 or None).
       sightings: list of dict(tid, ang, px[4,2], K)."""
    import gtsam
    from gtsam import Pose3, Rot3, CustomFactor
    from functools import partial

    D = gtsam.symbol('d', 0)
    C = gtsam.symbol('c', 0)
    Mk = {tid: gtsam.symbol('m', tid) for tid in mounts0}
    I = gtsam.symbol('i', 0)

    def to_pose(Rt):
        R, t = Rt
        return Pose3(Rot3(np.asarray(R, float)), np.asarray(t, float))

    initial = gtsam.Values()
    delta0 = np.array([0, 0, 0, 0, math.radians(model.delta_seed), 0.0])
    initial.insert(D, delta0)
    initial.insert(C, to_pose(T_cb0))
    for tid, m in mounts0.items():
        initial.insert(Mk[tid], to_pose(m))
    if use_imu:
        initial.insert(I, Pose3())                              # IMU mount, co-estimated

    graph = gtsam.NonlinearFactorGraph()

    # priors
    graph.add(gtsam.PriorFactorPose3(C, to_pose(T_cb0),
              gtsam.noiseModel.Diagonal.Sigmas(
                  np.r_[np.full(3, math.radians(tcb_rot_sigma_deg)), np.full(3, tcb_t_sigma_m)])))
    mount_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.r_[np.full(3, math.radians(mount_rot_sigma_deg)), [.005, .005, .005]])
    for tid, m in mounts0.items():
        graph.add(gtsam.PriorFactorPose3(Mk[tid], to_pose(m), mount_noise))

    def delta_prior_err(d0, this, v, H):
        d = v.atVector(this.keys()[0])
        if H is not None:
            H[0] = np.eye(6)
        return d - d0
    graph.add(CustomFactor(
        gtsam.noiseModel.Diagonal.Sigmas(np.deg2rad([15, 15, 15, 15, 30, 60])),
        [D], partial(delta_prior_err, delta0)))

    if use_imu:
        graph.add(gtsam.PriorFactorPose3(I, Pose3(),
                  gtsam.noiseModel.Diagonal.Sigmas(np.r_[np.deg2rad([20, 20, 20]), [1, 1, 1]])))

    # numerical-Jacobian helpers (tangent space)
    def read(v, key, typ):
        return v.atPose3(key).matrix() if typ == 'p' else v.atVector(key)

    def bump(v, key, typ, e):
        return v.atPose3(key).retract(e).matrix() if typ == 'p' else v.atVector(key) + e

    def custom(keys, typs, res, dim_out, noise):
        def err(this, v, H):
            pl = [read(v, k, t) for k, t in zip(keys, typs)]
            r = res(pl)
            if H is not None:
                for i, (k, t) in enumerate(zip(keys, typs)):
                    d = 6 if t == 'p' else 6
                    J = np.zeros((dim_out, d))
                    for j in range(d):
                        e = np.zeros(d); e[j] = 1e-6
                        pl2 = list(pl); pl2[i] = bump(v, k, t, e)
                        J[:, j] = (res(pl2) - r) / 1e-6
                    H[i] = J
            return r
        return CustomFactor(noise, keys, err)

    # tag reprojection factors
    px_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Huber.Create(3.0),
        gtsam.noiseModel.Isotropic.Sigma(8, 1.0))
    for s in sightings:
        tid, ang, px, K = s["tid"], s["ang"], np.array(s["px"]), s["K"]
        side = model.tags[tid]["side"]

        def res(pl, ang=ang, tid=tid, px=px, K=K, side=side):
            delta, T_cb, M = pl
            R_b, t_b = model.body_pose(ang, delta, tid)
            R_tag = R_b @ M[:3, :3]
            t_tag = R_b @ M[:3, 3] + t_b
            return (project(T_cb, R_tag, t_tag, side, K) - px).ravel()
        graph.add(custom([D, C, Mk[tid]], ['v', 'p', 'p'], res, 8, px_noise))

    # desk-tag factor: a base-rigid fiducial (bottom edge on y=0, flat on the plate,
    # edge || x). Casts its corners onto z=zp and pins T_cb's YAW + lateral(y) -- the two
    # DOF shoulder_pan was confounded with. Joint-independent, so it frees pan.
    if desk is not None:
        dc = np.array(desk["corners"]); dK = desk["K"]; dzp = float(desk["zp"])
        half = float(desk["side_m"]) / 2; ext = float(desk.get("extends_y", 1))

        def desk_res(pl):
            T = pl[0]; Rcb, tcb = T[:3, :3], T[:3, 3]
            base = []
            for u, v in dc:
                d = Rcb @ np.array([(u - dK["ppx"]) / dK["fx"],
                                    (v - dK["ppy"]) / dK["fy"], 1.0])
                base.append(tcb + (dzp - tcb[2]) / d[2] * d)
            base = np.array(base)
            eyaw = []
            for i in range(4):
                e = base[(i + 1) % 4] - base[i]
                if abs(e[0]) > abs(e[1]):                # an ~x-parallel edge
                    if e[0] < 0: e = -e                  # point +x for consistent sign
                    eyaw.append(e[1] / np.linalg.norm(e[:2]))
            return np.array([np.mean(eyaw), base[:, 1].mean() - ext * half])
        desk_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([math.sin(math.radians(0.5)), 0.003]))
        graph.add(custom([C], ['p'], desk_res, 2, desk_noise))

    # IMU gravity factors (optional, mount co-estimated -> roll component NOT a clean breaker)
    if use_imu:
        g0 = np.array([0, 0, -1.0])
        imu_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.02)
        for p in poses:
            if p.get("imu") is None:
                continue
            ang, a_meas = p["ang"], np.array(p["imu"])

            def res(pl, ang=ang, a_meas=a_meas):
                delta, R_imu = pl
                R_w = model.wrist_rot(ang, delta)
                return R_imu[:3, :3].T @ (R_w.T @ g0) - a_meas
            graph.add(custom([D, I], ['v', 'p'], res, 3, imu_noise))

    params = gtsam.LevenbergMarquardtParams()
    if verbose:
        params.setVerbosityLM("SUMMARY")
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()

    delta = result.atVector(D)
    T_cb = result.atPose3(C).matrix()
    mounts = {tid: result.atPose3(Mk[tid]).matrix() for tid in mounts0}
    try:
        cov = gtsam.Marginals(graph, result).marginalCovariance(D)
        sig = np.degrees(np.sqrt(np.clip(np.diag(cov), 0, None)))
    except Exception as e:
        sig = None
        if verbose:
            print(f"  (marginals failed: {e})")
    rms = float(np.sqrt(np.mean(graph.error(result) * 2 / max(graph.size(), 1))))
    return dict(delta=delta, T_cb=T_cb, mounts=mounts, sigma_deg=sig, graph_rms=rms)


def report(res):
    print("\n  joint           offset(deg)   1-sigma(deg)")
    for i, j in enumerate(JOINTS):
        s = f"{res['sigma_deg'][i]:8.2f}" if res["sigma_deg"] is not None else "   n/a"
        flag = ""
        if res["sigma_deg"] is not None and res["sigma_deg"][i] > 5:
            flag = "  <- weakly observable"
        print(f"  {j:14s} {math.degrees(res['delta'][i]):+9.2f}   {s}{flag}")


# ── synthetic self-test ─────────────────────────────────────────────────────────
def selftest():
    print("=== synthetic self-test ===")
    model = Model()
    rng = np.random.default_rng(0)
    he = json.load(open(HANDEYE)); T_cb = (np.array(he["R"]), np.array(he["t"]))
    mounts = {tid: (m["R"], m["t"]) for tid, m in model.tags.items()}
    K = dict(fx=900.0, fy=900.0, ppx=640.0, ppy=360.0)

    from scipy.spatial.transform import Rotation
    R_imu_true = Rotation.from_euler('xyz', [10, -20, 35], degrees=True).as_matrix()
    delta_true = np.array([math.radians(d) for d in
                           (3.0, -2.5, 4.0, -1.5, model.delta_seed + 6.0, 0.0)])

    poses, sightings = [], []
    grid = itertools.product(PANS, LIFTS, ELBOWS, WFS, ROLLS)
    for pan, lift, elbow, wf, roll in grid:
        ang = dict(shoulder_pan=pan, shoulder_lift=lift, elbow_flex=elbow,
                   wrist_flex=wf, wrist_roll=roll, gripper=GRIP_CMD)
        _, p = model.kin.fk(ang)
        if not (BOX_X[0] <= p[0] <= BOX_X[1] and BOX_Y[0] <= p[1] <= BOX_Y[1]
                and BOX_Z[0] <= p[2] <= BOX_Z[1]):
            continue
        R_w = model.wrist_rot(ang, delta_true)
        imu = R_imu_true.T @ (R_w.T @ np.array([0, 0, -1.0]))
        poses.append(dict(ang=ang, imu=(imu + rng.normal(0, 0.01, 3)).tolist()))
        for tid in model.tags:
            R_b, t_b = model.body_pose(ang, delta_true, tid)
            Rm, tm = mounts[tid]
            uv = project(np.block([[T_cb[0], T_cb[1][:, None]], [0, 0, 0, 1]]),
                         R_b @ Rm, R_b @ tm + t_b, model.tags[tid]["side"], K)
            sightings.append(dict(tid=tid, ang=ang,
                                  px=(uv + rng.normal(0, 0.3, uv.shape)).tolist(), K=K))
    print(f"{len(poses)} poses, {len(sightings)} sightings; "
          f"true delta(deg) = {np.round(np.degrees(delta_true), 1)}")

    def err5(res):
        return np.degrees(np.abs(res["delta"] - delta_true))[:5]

    print("\n--- (a) tags only, LOOSE mount prior (expect roll unobservable) ---")
    ra = build_and_solve(model, poses, sightings, T_cb, mounts,
                         mount_rot_sigma_deg=45.0, verbose=False)
    report(ra)
    print("\n--- (b) tags + TIGHT mount prior  ==  assumption B (expect roll fixed) ---")
    rb = build_and_solve(model, poses, sightings, T_cb, mounts,
                         mount_rot_sigma_deg=5.0, verbose=False)
    report(rb)
    print("\n--- (c) tags + IMU, mount CO-ESTIMATED (circular: not a clean roll breaker) ---")
    rc = build_and_solve(model, poses, sightings, T_cb, mounts, use_imu=True,
                         mount_rot_sigma_deg=45.0, verbose=False)
    report(rc)

    print("\nmax |error| on delta_1..5 (deg):")
    print(f"  (a) loose mounts : {err5(ra).max():.2f}   roll err {err5(ra)[4]:.2f}")
    print(f"  (b) assumption B : {err5(rb).max():.2f}   roll err {err5(rb)[4]:.2f}")
    print(f"  (c) tags + IMU   : {err5(rc).max():.2f}   roll err {err5(rc)[4]:.2f}")
    ok = err5(rb).max() < 1.5
    print(f"\nselftest {'PASS' if ok else 'FAIL'} "
          f"(assumption-B recovers delta_1..5 within 1.5 deg)")
    return ok


# ── autonomous collection ───────────────────────────────────────────────────────
class ImuReader:
    """Background rolling-average gravity direction in the IMU frame. Best-effort."""
    def __init__(self):
        self.g = None; self._stop = False
        try:
            from imu_serial import stream_samples, SCALE
        except Exception as e:
            print(f"  IMU off ({e})"); return
        self._t = threading.Thread(target=self._run, args=(stream_samples, SCALE), daemon=True)
        self._t.start()

    def _run(self, stream_samples, SCALE):
        buf = deque(maxlen=200)
        try:
            for t_us, x, y, z in stream_samples():
                if self._stop:
                    break
                buf.append((x, y, z))
                if len(buf) >= 50:
                    a = np.array(buf).mean(0) * SCALE
                    self.g = (a / np.linalg.norm(a)).tolist()
        except Exception as e:
            print(f"  IMU stream ended ({e})")

    def stop(self):
        self._stop = True


def collect(port):
    import cv2
    from handeye_calib import Realsense
    from pick_ball import move_to, read_angles
    from gamepad_utils import graceful_shutdown
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    model = Model()
    pose_list = []
    for pan, lift, elbow, wf, roll in itertools.product(PANS, LIFTS, ELBOWS, WFS, ROLLS):
        ang = dict(shoulder_pan=pan, shoulder_lift=lift, elbow_flex=elbow,
                   wrist_flex=wf, wrist_roll=roll, gripper=GRIP_CMD)
        _, p = model.kin.fk(ang)
        if (BOX_X[0] <= p[0] <= BOX_X[1] and BOX_Y[0] <= p[1] <= BOX_Y[1]
                and BOX_Z[0] <= p[2] <= BOX_Z[1]):
            pose_list.append(ang)
    step = max(len(pose_list) // 18, 1)
    pose_list = pose_list[::step][:18]
    print(f"{len(pose_list)} FK-checked poses")

    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(dic, par)

    cam = Realsense(color_res=(1280, 720))
    imu = ImuReader()
    robot = SOFollower(SOFollowerRobotConfig(port=port, id="so101", cameras={}))
    robot.connect()
    poses, sightings = [], []
    try:
        cur = read_angles(robot)
        for i, pose in enumerate(pose_list):
            cur = move_to(robot, cur, pose, seconds=1.4)
            time.sleep(0.6)                                  # settle (static gravity)
            color, depth, K = cam.grab()
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
            ang = read_angles(robot)
            n = 0
            if ids is not None:
                for c4, tid in zip(corners, ids.ravel()):
                    if int(tid) in model.tags:
                        sightings.append(dict(tid=int(tid), ang=ang,
                                              px=c4[0].astype(float).tolist(), K=K))
                        n += 1
            poses.append(dict(ang=ang, imu=imu.g))
            print(f"  pose {i + 1}/{len(pose_list)}: {n} tag(s)"
                  f"{'  +imu' if imu.g is not None else ''}")
    except Exception as e:
        print(f"\nUnexpected error: {e!r}\nLanding the arm safely...")
    finally:
        imu.stop(); cam.stop(); graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception:
            pass

    json.dump(dict(poses=poses, sightings=sightings), open(SAMPLES, "w"))
    print(f"\n{len(sightings)} sightings, {len(poses)} poses -> {SAMPLES}")
    return len(sightings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--imu", action="store_true", help="use IMU factors in the solve")
    ap.add_argument("--port", default="/dev/ttyACM1")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if args.collect:
        if collect(args.port) < 8:
            sys.exit("too few sightings to solve.")

    if not SAMPLES.exists():
        sys.exit(f"no samples at {SAMPLES} — run with --collect first.")

    model = Model()
    data = json.load(open(SAMPLES))
    he = json.load(open(HANDEYE)); T_cb = (np.array(he["R"]), np.array(he["t"]))
    mounts = {tid: (m["R"], m["t"]) for tid, m in model.tags.items()}
    desk = None
    dp = ROOT / "outputs/calib/desk_tag.json"
    if dp.exists():
        desk = json.load(open(dp))
    print(f"solving: {len(data['sightings'])} sightings, {len(data['poses'])} poses, "
          f"imu={'on' if args.imu else 'off'}, desk_tag={'on' if desk else 'off'}")
    # with the desk fiducial pinning yaw+lateral, loosen the (yaw-biased) handeye prior
    tcb_rot = 15.0 if desk else 2.0
    tcb_t = 0.03 if desk else 0.01
    res = build_and_solve(model, data["poses"], data["sightings"], T_cb, mounts,
                          use_imu=args.imu, desk=desk,
                          tcb_rot_sigma_deg=tcb_rot, tcb_t_sigma_m=tcb_t)
    print(f"\ngraph reprojection RMS = {res['graph_rms']:.2f}px")
    report(res)

    out = dict(delta_deg={j: math.degrees(res["delta"][i]) for i, j in enumerate(JOINTS)},
               sigma_deg=(res["sigma_deg"].tolist() if res["sigma_deg"] is not None else None),
               graph_rms_px=res["graph_rms"], used_imu=args.imu, used_desk=desk is not None,
               T_cb_R=res["T_cb"][:3, :3].tolist(), T_cb_t=res["T_cb"][:3, 3].tolist(),
               n_sightings=len(data["sightings"]), created=datetime.now().isoformat())
    json.dump(out, open(RESULT, "w"), indent=2)
    print(f"\nsaved {RESULT}")


if __name__ == "__main__":
    main()
