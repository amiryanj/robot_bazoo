#!/usr/bin/env python
"""
Grab one aligned color+depth frame from the top-down Realsense and save it, so
vision code (plane fit, ball localization) can be developed/verified offline
against real data — no arm, no noise.

Saves to outputs/vision/<timestamp>/:
    color.png        BGR image (640×480)
    depth.npy        float32 depth in METERS, aligned to color
    intrinsics.json  fx, fy, ppx, ppy, width, height  (color stream)

Usage:
    python vision/capture_frame.py            # put the ball on the table first
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RS_SERIAL, OUT as OUT_ROOT   # noqa: E402

SERIAL = RS_SERIAL
OUT = OUT_ROOT / "vision"        # was an absolute /home/javad/... path


def main():
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(SERIAL)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    try:
        for _ in range(30):                     # warm up auto-exposure
            pipe.wait_for_frames()
        frames = align.process(pipe.wait_for_frames())
        depth_f, color_f = frames.get_depth_frame(), frames.get_color_frame()
        if not depth_f or not color_f:
            print("No frame — is the camera free (station.py not running)?", file=sys.stderr)
            return
        intr = color_f.get_profile().as_video_stream_profile().get_intrinsics()
        depth_m = np.asarray(depth_f.get_data(), dtype=np.float32) * depth_scale
        color = np.asarray(color_f.get_data())
    finally:
        pipe.stop()

    out = OUT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "color.png"), color)
    np.save(out / "depth.npy", depth_m)
    json.dump({"fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy,
               "width": intr.width, "height": intr.height},
              open(out / "intrinsics.json", "w"), indent=2)

    valid = depth_m[depth_m > 0]
    print(f"Saved {out}")
    print(f"  depth: {valid.min():.3f}–{valid.max():.3f} m valid, "
          f"{100*len(valid)/depth_m.size:.0f}% of pixels have depth")
    print(f"  intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"ppx={intr.ppx:.1f} ppy={intr.ppy:.1f}")


if __name__ == "__main__":
    main()
