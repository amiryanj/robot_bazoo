#!/usr/bin/env python
"""The alignment proof: render ONE MuJoCo scene, base frame, with everything overlaid —
the robot at its live joints, the camera's colored point cloud (the world as the camera
sees it), the detected ball, a z=0 reference plane + the measured table plane, the finger
tags (FK-predicted), and the camera itself as a pinhole frustum at its calibrated pose.

If the camera's cloud of the table lands flat on the plate and the ball sits on it right
where the arm reaches, T_cam->base is correct. Saves 3/4 + side PNGs to outputs/calib/.

    MUJOCO_GL=egl python vision/scene_align.py [--port /dev/ttyACM1] [--no-arm]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

HANDEYE = ROOT / "outputs/calib/handeye.json"
XML = str(ROOT / "SO-ARM100/Simulation/SO101/scene.xml")
OUT = ROOT / "outputs/calib"
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--no-arm", action="store_true")
    args = ap.parse_args()

    import cv2
    import mujoco as mj
    from ball import WORKSPACE_Z
    from ball_yolo import ball_from_box
    from cloud import crop_z
    from cloud import deproject as dp
    from realsense import Realsense
    from pick_ball import BallDetector, JOINT_OFFSETS

    he = json.load(open(HANDEYE))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])

    # live joints (read once, leave the arm limp)
    ang = {n: 0.0 for n in MOTOR_NAMES}
    if not args.no_arm:
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
        robot.connect(); robot.bus.disable_torque()
        obs = robot.get_observation()
        ang = {n: obs.get(f"{n}.pos", 0.0) for n in MOTOR_NAMES}
        robot.disconnect()

    # camera frame: cloud + ball
    cam = Realsense(color_res=(1280, 720))
    for _ in range(8):
        color, depth, K = cam.grab()
    cam.stop()
    cam_pts = crop_z(dp(depth, K), WORKSPACE_Z)
    ys, xs = np.where(depth > 0)
    z = depth[ys, xs]
    keep = (z > WORKSPACE_Z[0]) & (z < WORKSPACE_Z[1])
    ys, xs, z = ys[keep], xs[keep], z[keep]
    X = (xs - K["ppx"]) * z / K["fx"]; Y = (ys - K["ppy"]) * z / K["fy"]
    cloud_cam = np.stack([X, Y, z], 1)
    cloud_base = (R_cb @ cloud_cam.T).T + t_cb
    cloud_rgb = np.clip(color[ys, xs][:, ::-1] / 255.0 * 1.7, 0, 1)   # boost dark wood
    sub = np.random.choice(len(cloud_base), min(4500, len(cloud_base)), replace=False)

    det = BallDetector()
    box, score = det.detect(color)
    ball = None
    if box:
        b = ball_from_box(box, score, depth, K)
        ball = (R_cb @ b["center3d"] + t_cb, b["radius_m"])

    # MuJoCo model at live joints
    m = mj.MjModel.from_xml_path(XML)
    d = mj.MjData(m)
    adr = {j: m.jnt_qposadr[mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, j)] for j in MOTOR_NAMES}
    for j, a in adr.items():
        d.qpos[a] = math.radians(ang[j] + JOINT_OFFSETS.get(j, 0.0))
    mj.mj_forward(m, d)

    ren = mj.Renderer(m, 480, 640, max_geom=6000)
    cam_v = mj.MjvCamera()
    eye = np.eye(3).ravel()

    def G(scn, gtype, size, pos, rgba, mat=None):
        g = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        (mat if mat is not None else eye), np.asarray(rgba, np.float32))
        scn.ngeom += 1

    def line(scn, p0, p1, rgba, w=0.003):
        g = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(g, mj.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), eye,
                        np.asarray(rgba, np.float32))
        mj.mjv_connector(g, mj.mjtGeom.mjGEOM_CAPSULE, w, np.asarray(p0, float), np.asarray(p1, float))
        scn.ngeom += 1

    def build(scn):
        zt = float(np.median(cloud_base[sub][:, 2]))
        # z=0 reference as a WIREFRAME square (doesn't occlude the cloud), + the measured
        # table plane as a faint slab at its actual height (reveals the base->plate offset).
        c0 = [(0.0, -0.30), (0.50, -0.30), (0.50, 0.30), (0.0, 0.30)]
        for a, b in zip(c0, c0[1:] + c0[:1]):
            line(scn, [a[0], a[1], 0.0], [b[0], b[1], 0.0], [0.2, 0.95, 0.2, 0.9], w=0.0025)
        G(scn, mj.mjtGeom.mjGEOM_BOX, [0.25, 0.30, 0.0004], [0.25, 0.0, zt], [0.75, 0.7, 0.6, 0.30])
        # coloured point cloud (the world as the camera sees it, in the base frame)
        for i in sub:
            G(scn, mj.mjtGeom.mjGEOM_SPHERE, [0.0030, 0, 0], cloud_base[i], [*cloud_rgb[i], 1.0])
        # ball
        if ball is not None:
            G(scn, mj.mjtGeom.mjGEOM_SPHERE, [ball[1], 0, 0], ball[0], [0.95, 0.5, 0.15, 0.6])
        # camera as a pinhole: body box + frustum to the image corners at 0.45 m
        H, W = depth.shape
        G(scn, mj.mjtGeom.mjGEOM_BOX, [0.02, 0.03, 0.015], t_cb, [0.1, 0.1, 0.1, 1.0], R_cb.ravel())
        for (u, v) in [(0, 0), (W, 0), (W, H), (0, H)]:
            ray = np.array([(u - K["ppx"]) / K["fx"], (v - K["ppy"]) / K["fy"], 1.0])
            corner = R_cb @ (ray * 0.45) + t_cb
            line(scn, t_cb, corner, [0.2, 0.6, 1.0, 0.8], w=0.002)

    for name, (az, el, dist, look) in {
        "align_3q":   (135, -25, 1.1, [0.25, 0.0, -0.0]),
        "align_side": (90, -8, 1.0, [0.30, 0.0, 0.0]),
        "align_top":  (90, -78, 0.95, [0.27, 0.0, -0.03]),
    }.items():
        cam_v.azimuth, cam_v.elevation, cam_v.distance = az, el, dist
        cam_v.lookat[:] = look
        ren.update_scene(d, cam_v)
        build(ren.scene)
        import imageio
        imageio.imwrite(str(OUT / f"{name}.png"), ren.render())
        print(f"  saved {OUT / (name + '.png')}")

    if ball is not None:
        zt = float(np.median(cloud_base[sub][:, 2])) * 1000
        print(f"  ball base {np.round(ball[0]*1000).astype(int)}mm  table≈{zt:.0f}mm  "
              f"ball bottom {(ball[0][2]-ball[1])*1000:.0f}mm")
    print(f"  camera at base {np.round(t_cb*1000).astype(int)}mm")


if __name__ == "__main__":
    main()
