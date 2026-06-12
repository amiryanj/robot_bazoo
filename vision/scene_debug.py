#!/usr/bin/env python
"""Live 3-D truth window: is our geometry chain right?

One MuJoCo viewer shows, in the BASE frame, live:
  - the arm, mirroring measured joint angles (torque is DISABLED — hand-move it),
  - the detected ball (orange sphere, depth-based, student-YOLO/GDINO detector),
  - the pink heart MEASURED by the camera (magenta dot: detection + depth + T_cam->base),
  - the same heart PREDICTED by kinematics (green dot: joints + FK + solved offset),
  - the RANSAC table plane (grey slab, fitted once at startup).

The magenta-vs-green gap is the end-to-end error of the whole chain (detection,
depth, hand-eye transform, FK) at this very moment — it is printed live in mm.
Hand-move the arm around; if the gap stays ~5-10 mm everywhere, the geometry is
trustworthy. If the ball floats off its real spot, perception is lying.

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
    args = ap.parse_args()

    import cv2
    import math
    import mujoco
    import mujoco.viewer
    import rerun as rr
    import rerun.blueprint as rrb
    import torch
    from ball import deproject, fit_table_plane, WORKSPACE_Z
    from ball_yolo import BALL_RADIUS_M, ball_from_box
    from cloud import crop_z, fit_sphere_known_r
    from cloud import deproject as cloud_deproject
    from handeye_calib import HeartDetector, Realsense, backproject, make_fk
    from pick_ball import BallDetector

    he = json.load(open(HANDEYE))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])
    offset = np.array(he["marker_offset"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    balls = BallDetector(device)
    hearts = HeartDetector(device)
    fk = make_fk()
    cam = Realsense()

    robot = None
    if not args.no_arm:
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
        robot.connect()
        robot.bus.disable_torque()
        print("Arm torque DISABLED — hand-move it freely.")

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

    def project(p_base, K):
        """Base-frame point -> image pixel via the hand-eye transform."""
        pc = R_cb.T @ (np.asarray(p_base) - t_cb)
        if pc[2] <= 0.05:
            return None
        return (K["fx"] * pc[0] / pc[2] + K["ppx"], K["fy"] * pc[1] / pc[2] + K["ppy"])

    # table plane: fit once (full-frame RANSAC is too slow per-loop)
    color, depth, K = cam.grab()
    pts = deproject(depth, K).reshape(-1, 3)
    valid = pts[(pts[:, 2] > WORKSPACE_Z[0]) & (pts[:, 2] < WORKSPACE_Z[1])]
    n_cam, d_cam, _ = fit_table_plane(valid)
    n_b = R_cb @ n_cam
    d_b = d_cam - n_b @ t_cb
    z_table0 = -(d_b + n_b[0] * 0.25) / n_b[2]            # plane height near x=0.25,y=0
    print(f"fitted plane ~z={z_table0 * 1000:.0f}mm in base frame (under x=0.25)")

    def add_sphere(scn, i, pos, r, rgba):
        mujoco.mjv_initGeom(scn.geoms[i], mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([r, 0, 0], float), np.asarray(pos, float),
                            np.eye(3).ravel(), np.array(rgba, np.float32))

    def to_base(pts):
        return (R_cb @ pts.T).T + t_cb

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
                b = ball_from_box(box, conf, depth, K, plane=(n_cam, d_cam))
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

            uv, heart_boxes = hearts.marker_uv(color)
            p_heart_meas = None
            if uv is not None:
                pc = backproject(*uv, depth, K)
                if pc is not None:
                    p_heart_meas = R_cb @ pc + t_cb

            for nm, p, col in (("heart_meas", p_heart_meas, (255, 30, 200)),):
                if p is not None:
                    rr.log(f"world/{nm}", rr.Points3D([p], radii=0.008, colors=col))

            p_heart_pred = None
            if robot is not None:
                obs = robot.get_observation()
                ang = {n: obs.get(f"{n}.pos", 0.0) for n in MOTOR_NAMES}
                for j, a in adr.items():
                    d.qpos[a] = math.radians(ang[j])
                mujoco.mj_forward(m, d)
                R_w, t_w = fk(ang)
                p_heart_pred = R_w @ offset + t_w
                rr.log("world/heart_pred", rr.Points3D([p_heart_pred], radii=0.008,
                                                       colors=(30, 255, 60)))

            # camera panel: image + ball box + every heart candidate + projected points
            rr.log("cam/image", rr.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)))
            if box:
                rr.log("cam/ball", rr.Boxes2D(
                    array=[[box[0], box[1], box[2] - box[0], box[3] - box[1]]],
                    array_format=rr.Box2DFormat.XYWH, labels=[f"ball {conf:.2f}"]))
            else:
                rr.log("cam/ball", rr.Clear(recursive=False))
            if heart_boxes:
                rr.log("cam/hearts", rr.Boxes2D(
                    array=[[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2), *_ in heart_boxes],
                    array_format=rr.Box2DFormat.XYWH,
                    labels=[f"{c:.2f} pink={f:.2f}" for _, c, _, f in heart_boxes]))
            else:
                rr.log("cam/hearts", rr.Clear(recursive=False))
            pts, cols = [], []
            if uv is not None:
                pts.append(list(uv)); cols.append((255, 30, 200))      # measured: magenta
            if p_heart_pred is not None:
                pp = project(p_heart_pred, K)
                if pp is not None:
                    pts.append(list(pp)); cols.append((30, 255, 60))   # predicted: green
            if pts:
                rr.log("cam/points", rr.Points2D(pts, radii=7, colors=cols))
            else:
                rr.log("cam/points", rr.Clear(recursive=False))

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
            if p_heart_meas is not None:
                add_sphere(scn, i, p_heart_meas, 0.008, [1.0, 0.1, 0.8, 1.0]); i += 1
            if p_heart_pred is not None:
                add_sphere(scn, i, p_heart_pred, 0.008, [0.1, 1.0, 0.2, 1.0]); i += 1
            scn.ngeom = i
            viewer.sync()

            msg = []
            if p_ball is not None:
                msg.append(f"ball={np.round(p_ball * 1000).astype(int)}mm@{conf:.2f}")
            if p_heart_meas is not None and p_heart_pred is not None:
                err = np.linalg.norm(p_heart_meas - p_heart_pred) * 1000
                msg.append(f"heart err={err:.1f}mm")
            elif p_heart_pred is not None:
                msg.append("heart: not seen by camera")
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
