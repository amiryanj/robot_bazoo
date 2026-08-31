#!/usr/bin/env python
"""The top-down Realsense D455, and pixel+depth -> 3-D.

  - `Realsense`   — D455 wrapper, aligned color+depth+intrinsics per grab,
  - `backproject` — pixel + aligned depth -> 3-D point in the camera frame.

Was `realsense.py`, the pink-heart hand-eye tool. The heart is gone and hand-eye
calibration lives in `vision/cam_calib.py` now (reference-anchored: white-plate plane +
desk ArUco tag). Only the camera layer was left, so the file is named for what it is.
`make_fk` went with the rename: it was MuJoCo FK of the `gripper` body, and nothing
called it -- `tag_sweep.py` has its own `make_fk2`, and `pick_ball.Kin` is the FK
everything else uses.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SERIAL = "117222251972"
XML = str(ROOT / "SO-ARM100/Simulation/SO101/scene.xml")   # tag_sweep imports this
OUT = ROOT / "outputs/calib"
WORKSPACE_Z = (0.20, 1.2)          # metres; valid depth band for back-projection


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
