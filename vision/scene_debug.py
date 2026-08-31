#!/usr/bin/env python
"""Live 3-D truth window: is our geometry chain right?

One MuJoCo viewer shows, in the BASE frame, live:
  - the arm, mirroring measured joint angles (torque is DISABLED — hand-move it),
  - the detected ball (orange sphere, depth-based, student-YOLO/GDINO detector),
  - the 3 ArUco tags (2 finger tags cyan, the static desk tag magenta),
  - the RANSAC table plane (grey slab, fitted once at startup).

Hand-move the arm around; if the ball sits at its real spot and the cyan tags overlay
the real finger tags, the geometry chain (detection, depth, hand-eye, FK + joint
offsets) is trustworthy. If the ball floats off its real spot, perception is lying.

    python vision/scene_debug.py [--port /dev/ttyACM1] [--no-arm]
Ctrl-C to quit (arm is limp throughout; nothing to land).
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

HANDEYE = ROOT / "outputs/calib/handeye.json"
XML = str(ROOT / "SO-ARM100/Simulation/SO101/scene.xml")
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--no-arm", action="store_true", help="camera only (no FK overlay)")
    ap.add_argument("--hold", action="store_true",
                    help="keep torque ON (rigid, honest FK) — can't hand-move; "
                         "avoids the backlash artifact of the limp arm")
    args = ap.parse_args()

    import cv2
    import math
    import mujoco
    import mujoco.viewer
    import rerun as rr
    import rerun.blueprint as rrb
    import torch
    from ball import WORKSPACE_Z
    from ball_yolo import BALL_RADIUS_M, ball_from_box
    from cloud import crop_z, fit_sphere_known_r
    from cloud import deproject as cloud_deproject
    from realsense import Realsense
    from pick_ball import BallDetector

    he = json.load(open(HANDEYE))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    balls = BallDetector(device)
    cam = Realsense()

    robot = None
    if not args.no_arm:
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
        robot.connect()
        if args.hold:
            print("Arm torque ON — holding its powered pose rigidly (honest FK, no backlash). "
                  "Can't hand-move.")
        else:
            robot.bus.disable_torque()
            print("Arm torque DISABLED — hand-move it freely. "
                  "(NOTE: backlash under hand-load can show ~1-2cm phantom gap; use --hold "
                  "for true geometry.)")

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    adr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
           for j in MOTOR_NAMES}
    viewer = mujoco.viewer.launch_passive(m, d)

    rr.init("scene_debug", spawn=True)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(rrb.Spatial2DView(origin="cam", name="camera"),
                       rrb.Spatial3DView(origin="world", name="cloud (base frame)")),
        collapse_panels=True))

    # support planes: fit once at startup (full-frame RANSAC is too slow per-loop)
    from cloud import extract_planes
    color, depth, K = cam.grab()
    pc0 = crop_z(cloud_deproject(depth, K), WORKSPACE_Z)
    base0 = (R_cb @ pc0.T).T + t_cb
    supports = [pl for pl in extract_planes(base0[::4], max_planes=3)
                if abs(pl["n"][2]) > 0.95]
    print("support planes at "
          f"{[round(float(pl['centroid'][2]) * 1000) for pl in supports]} mm (base frame)")
    z_table0 = float(supports[0]["centroid"][2]) if supports else -0.026
    for i, pl in enumerate(supports):
        lo, hi = np.asarray(pl["extent"][0]), np.asarray(pl["extent"][1])
        ctr = (lo + hi) / 2
        half = np.maximum((hi - lo) / 2, 0.01)
        rr.log(f"world/support_{i}", rr.Boxes3D(
            centers=[[ctr[0], ctr[1], float(pl['centroid'][2])]],
            half_sizes=[[half[0], half[1], 0.001]],
            colors=(200, 190, 160, 90)))

    # last scene scan (vision/scene_model.py): blob cards as wireframe boxes
    import scene_model as sm
    model = sm.load()
    if model:
        for i, b in enumerate(model.get("blobs", [])):
            lo, hi = np.asarray(b["extent_min"]), np.asarray(b["extent_max"])
            rr.log(f"world/blob_{i}", rr.Boxes3D(
                centers=[((lo + hi) / 2).tolist()],
                half_sizes=[((hi - lo) / 2).tolist()],
                colors=(90, 160, 255, 80),
                labels=[f"blob h={b['height'] * 1000:.0f}mm"]))

    def add_sphere(scn, i, pos, r, rgba):
        mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([r, 0, 0], float), np.asarray(pos, float),
                            np.eye(3).ravel(), np.array(rgba, np.float32))

    def to_base(pts):
        return (R_cb @ pts.T).T + t_cb

    def add_box(scn, i, ctr, R, half, rgba):
        mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_BOX,
                            np.array([half, half, 0.0006]), np.asarray(ctr, float),
                            np.asarray(R, float).ravel(), np.array(rgba, np.float32))

    # the 3 ArUco tags as thin planar geoms: 2 finger tags (cyan, on the gripper bodies)
    # + the static desk tag (magenta) -- to eyeball the tag calibration inside the twin.
    tc = json.load(open(ROOT / "outputs/calib/tag_calib.json"))
    finger_tags = [(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, t["body"]),
                    np.array(t["R"]), np.array(t["t"]), float(t["side_m"]) / 2)
                   for t in tc["tags"].values()]
    # the static desk tag (magenta) is detected LIVE each frame: DICT_4X4_50 id 13 (the
    # old MIP-36h12 "id 8" in desk_tag.json was stale). Drawn at its measured base position.
    desk_par = cv2.aruco.DetectorParameters()
    desk_par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    desk_det = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), desk_par)
    DESK_TAG_ID = 13

    frame_i = 0
    try:
        while viewer.is_running():
            color, depth, K = cam.grab()
            frame_i += 1

            # 3-D debug panel: workspace cloud (every 5th frame, subsampled)
            if frame_i % 5 == 1:
                pc = crop_z(cloud_deproject(depth, K), WORKSPACE_Z)
                rr.log("world/cloud", rr.Points3D(to_base(pc[::12]), radii=0.0012,
                                                  colors=(150, 150, 150)))

            box, conf = balls.detect(color)
            p_ball = None
            if box:
                b = ball_from_box(box, conf, depth, K)
                if b:
                    p_ball = R_cb @ b["center3d"] + t_cb
                    r_ball = b["radius_m"]
                # sphere-fit forensics: which points the fitter believed
                pad = max((box[2] - box[0]) // 6, 3)
                bp = crop_z(cloud_deproject(depth, K, box=(box[0] - pad, box[1] - pad,
                                                           box[2] + pad, box[3] + pad)),
                            WORKSPACE_Z)
                fit = fit_sphere_known_r(bp, BALL_RADIUS_M)
                if fit is not None:
                    err = np.abs(np.linalg.norm(bp - fit["center"], axis=1) - BALL_RADIUS_M)
                    inl = err < 0.004
                    rr.log("world/ball_inliers", rr.Points3D(to_base(bp[inl]),
                                                             radii=0.0015, colors=(40, 220, 60)))
                    rr.log("world/ball_outliers", rr.Points3D(to_base(bp[~inl]),
                                                              radii=0.0015, colors=(220, 60, 40)))
                if p_ball is not None:
                    rr.log("world/ball_center", rr.Points3D([p_ball], radii=r_ball,
                                                            colors=(245, 130, 40, 120)))

            if robot is not None:
                obs = robot.get_observation()
                ang = {n: obs.get(f"{n}.pos", 0.0) for n in MOTOR_NAMES}
                from pick_ball import JOINT_OFFSETS
                for j, a in adr.items():
                    off = JOINT_OFFSETS.get(j, 0.0)
                    d.qpos[a] = math.radians(ang[j] + off)
                mujoco.mj_forward(m, d)

            # camera panel: image + ball box
            rr.log("cam/image", rr.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)))
            if box:
                rr.log("cam/ball", rr.Boxes2D(
                    array=[[box[0], box[1], box[2] - box[0], box[3] - box[1]]],
                    array_format=rr.Box2DFormat.XYWH, labels=[f"ball {conf:.2f}"]))
            else:
                rr.log("cam/ball", rr.Clear(recursive=False))

            # live desk tag (id 13): position from depth at its centre, drawn flat
            desk_geom = None
            dcorn, dids, _ = desk_det.detectMarkers(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
            if dids is not None and DESK_TAG_ID in dids.ravel():
                c4 = dcorn[list(dids.ravel()).index(DESK_TAG_ID)][0]
                u0, v0 = c4.mean(0).astype(int)
                win = depth[max(v0 - 3, 0):v0 + 4, max(u0 - 3, 0):u0 + 4]
                win = win[win > 0]
                if win.size:
                    zc = float(np.median(win))
                    pc = np.array([(u0 - K["ppx"]) / K["fx"], (v0 - K["ppy"]) / K["fy"], 1.0]) * zc
                    desk_geom = (R_cb @ pc + t_cb, np.eye(3), 0.014)

            scn = viewer.user_scn
            i = 0
            # measured table plane (thin grey slab)
            mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_BOX,
                                np.array([0.45, 0.45, 0.001]),
                                np.array([0.25, 0.0, z_table0 - 0.001]),
                                np.eye(3).ravel(), np.array([0.8, 0.75, 0.65, 0.35], np.float32))
            i += 1
            if p_ball is not None:
                add_sphere(scn, i, p_ball, r_ball, [0.95, 0.5, 0.15, 1.0]); i += 1
            if robot is not None:                              # 2 finger tags (cyan)
                for bid, mR, mt, half in finger_tags:
                    bR = d.xmat[bid].reshape(3, 3); bt = d.xpos[bid]
                    add_box(scn, i, bR @ mt + bt, bR @ mR, half, [0.1, 0.9, 0.9, 0.9]); i += 1
            if desk_geom is not None:                          # static desk tag (magenta)
                add_box(scn, i, *desk_geom, [0.9, 0.2, 0.9, 0.9]); i += 1
            scn.ngeom = i
            viewer.sync()

            msg = []
            if p_ball is not None:
                msg.append(f"ball={np.round(p_ball * 1000).astype(int)}mm@{conf:.2f}")
            print("  " + ("   ".join(msg) if msg else "nothing detected"), end="\r")
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()
        if robot is not None:
            try:
                robot.disconnect()
            except Exception:
                pass
        viewer.close()


if __name__ == "__main__":
    main()
