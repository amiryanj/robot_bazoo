#!/usr/bin/env python
"""Shared camera + FK/geometry utilities for the vision stack.

NOTE: this file used to be the pink-heart hand-eye calibration tool. The heart marker is
gone and hand-eye calibration now lives in `vision/cam_calib.py` (reference-anchored:
white-plate plane + desk ArUco tag). What remains here are the generic, still-shared bits
that many tools import:
  - `Realsense`   — the top-down D455 wrapper (aligned color+depth+intrinsics per grab),
  - `backproject` — pixel + aligned depth -> 3-D point in the camera frame,
  - `make_fk`     — MuJoCo FK of the `gripper` body (base frame).

(The filename is kept only so the ~20 importers don't churn; it's no longer a calib tool.)
"""
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SERIAL = "117222251972"
XML = str(ROOT / "SO-ARM100/Simulation/SO101/scene.xml")
MARKER_BODY = "gripper"            # FK target: the wrist_roll part the fingers hang off
OUT = ROOT / "outputs/calib"
WORKSPACE_Z = (0.20, 1.2)          # metres; valid depth band for back-projection
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def backproject(u, v, depth_m, K, win=4):
    """Pixel + aligned depth -> 3-D point in the camera frame (metres). Uses the median
    valid depth in a small window for robustness. None if no valid depth there."""
    z = depth_m[max(v - win, 0):v + win, max(u - win, 0):u + win]
    z = z[(z > WORKSPACE_Z[0]) & (z < WORKSPACE_Z[1])]
    if len(z) < 5:
        return None
    z = float(np.median(z))
    return np.array([(u - K["ppx"]) * z / K["fx"],
                     (v - K["ppy"]) * z / K["fy"], z])


def make_fk():
    """Return fk(ang_deg) -> (R, t): pose of the `gripper` body (the wrist_roll part) in
    the base/world frame, from MuJoCo FK of the SO-101 model."""
    import mujoco
    mm = mujoco.MjModel.from_xml_path(XML)
    md = mujoco.MjData(mm)
    adr = {j: mm.jnt_qposadr[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, j)]
           for j in MOTOR_NAMES}
    bid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, MARKER_BODY)

    def fk(ang):
        for j, a in adr.items():
            md.qpos[a] = math.radians(ang[j])
        mujoco.mj_forward(mm, md)
        return md.xmat[bid].reshape(3, 3).copy(), md.xpos[bid].copy()
    return fk


# ── Realsense (own pipeline: aligned color+depth+intrinsics) ──────────────────────────

class Realsense:
    def __init__(self, color_res=(640, 480)):
        """color_res: bump to (1280, 720) for small-tag work — color and depth are
        independent streams; depth STAYS at 640x480 on purpose (its min-Z blind zone
        grows with depth resolution). Depth is align-projected onto the color grid,
        and intrinsics come per-grab, so consumers don't care about the choice."""
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(SERIAL)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, color_res[0], color_res[1], rs.format.bgr8, 30)
        prof = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        for _ in range(30):
            self.pipe.wait_for_frames()                       # warm up auto-exposure

    def grab(self):
        f = self.align.process(self.pipe.wait_for_frames())
        d, c = f.get_depth_frame(), f.get_color_frame()
        intr = c.get_profile().as_video_stream_profile().get_intrinsics()
        K = dict(fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy)
        return (np.asarray(c.get_data()),
                np.asarray(d.get_data(), np.float32) * self.scale, K)

    def stop(self):
        self.pipe.stop()
