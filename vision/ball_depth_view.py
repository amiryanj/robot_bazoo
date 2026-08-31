#!/usr/bin/env python
"""Live look at the RealSense depth right where the ball is detected.

Streams to Rerun: full color (with the detected ball box), full depth, a ZOOMED depth
crop of just the ball box, and a live valid-depth-% scalar for that crop. Use it to see
whether the ball is a depth hole (the D455 min-Z blind zone).

    python vision/ball_depth_view.py        # Ctrl-C to stop
"""
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))


def main():
    import cv2
    import json
    import rerun as rr
    import rerun.blueprint as rrb
    from ball import WORKSPACE_Z
    from cloud import crop_z
    from cloud import deproject as cloud_deproject
    from realsense import Realsense
    from pick_ball import BallDetector, HANDEYE

    he = json.load(open(HANDEYE))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])

    rr.init("ball_depth_view", spawn=True)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Horizontal(rrb.Spatial2DView(origin="color", name="color + box"),
                               rrb.Spatial2DView(origin="depth", name="depth (full)")),
                rrb.Horizontal(rrb.Spatial2DView(origin="ball/depth", name="ball depth (zoom)"),
                               rrb.TimeSeriesView(origin="ball/valid_pct", name="valid depth %")),
            ),
            rrb.Spatial3DView(origin="cloud", name="point cloud (base frame)"),
            column_shares=[1, 1])))

    det = BallDetector()
    cam = Realsense()
    print(f"Detector on {det.device}. Ctrl-C to stop.")
    rr.log("cloud", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    try:
        while True:
            color, depth, K = cam.grab()          # depth: float32 metres, 640x480
            rr.set_time_seconds("t", time.time())
            rr.log("color", rr.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)))
            rr.log("depth", rr.DepthImage(depth, meter=1.0))

            # 3-D point cloud in the BASE frame, coloured with the RGB image, so you can
            # see the ball region's depth (or the hole) relative to the plate in 3-D.
            ys, xs = np.where(depth > 0)
            z = depth[ys, xs]
            ok = (z > WORKSPACE_Z[0]) & (z < WORKSPACE_Z[1])
            ys, xs, z = ys[ok], xs[ok], z[ok]
            X = (xs - K["ppx"]) * z / K["fx"]
            Y = (ys - K["ppy"]) * z / K["fy"]
            cam_pts = np.stack([X, Y, z], axis=1)
            base_pts = (R_cb @ cam_pts.T).T + t_cb
            cols = color[ys, xs][:, ::-1]                     # BGR -> RGB
            rr.log("cloud/scene", rr.Points3D(base_pts[::4], colors=cols[::4], radii=0.0015))

            box, score = det.detect(color)
            if box:
                x1, y1, x2, y2 = box
                rr.log("color/ball", rr.Boxes2D(
                    array=[[x1, y1, x2 - x1, y2 - y1]],
                    array_format=rr.Box2DFormat.XYWH, labels=[f"ball {score:.2f}"]))
                crop = depth[max(y1, 0):y2, max(x1, 0):x2]
                rr.log("ball/depth", rr.DepthImage(crop, meter=1.0))
                valid = crop[crop > 0]
                pct = 100.0 * valid.size / max(crop.size, 1)
                rr.log("ball/valid_pct", rr.Scalars(pct))
                # red: whatever depth points fall inside the ball box (the hole shows as none)
                inb = (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
                rr.log("cloud/ball_box_pts", rr.Points3D(base_pts[inb], colors=(255, 40, 40),
                                                         radii=0.003))
                med = float(np.median(valid)) * 1000 if valid.size else 0.0
                print(f"  ball {score:.2f}  depth valid {pct:5.1f}%  median {med:4.0f}mm  "
                      f"box_pts_in_cloud={int(inb.sum())}   ", end="\r", flush=True)
            else:
                rr.log("color/ball", rr.Clear(recursive=False))
                rr.log("cloud/ball_box_pts", rr.Clear(recursive=False))
                print("  no ball detected                              ", end="\r", flush=True)
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()


if __name__ == "__main__":
    main()
