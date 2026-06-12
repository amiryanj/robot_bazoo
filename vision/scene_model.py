#!/usr/bin/env python
"""Scene model v0 — the perception↔twin contract (vision/PERCEPTION3D.md roadmap).

A scan turns one RealSense frame into a persistent, base-frame description:
  - planes: from cloud.extract_planes, classified as SUPPORT surfaces when horizontal
    and facing up (|n·ẑ| > 0.95) — "things can rest here", with height and xy extent;
  - objects: cards for what the detectors found (v0: the ball — class, centre, radius,
    which support plane it rests on, or 'raised').

Saved to outputs/calib/scene_model.json; consumers (twin, scene_debug, future state
machine) render/read the file instead of re-deriving geometry. Re-scan whenever the
scene changes; the file is keyed to the hand-eye calibration that produced it.

    python vision/scene_model.py            # grab a frame, scan, save, print summary
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

OUT = ROOT / "outputs/calib/scene_model.json"
HANDEYE = ROOT / "outputs/calib/handeye.json"
WORKSPACE_Z = (0.20, 1.2)


def scan(color, depth, K, R_cb, t_cb, ball_detector=None):
    from ball_yolo import ball_from_box
    from cloud import crop_z, deproject, extract_planes

    pts = crop_z(deproject(depth, K), WORKSPACE_Z)
    base = (R_cb @ pts.T).T + t_cb

    planes = []
    for p in extract_planes(base[::4], max_planes=3):
        n = p["n"] if p["n"][2] >= 0 else -p["n"]          # normals face up
        d = -float(n @ p["centroid"])
        support = bool(abs(n[2]) > 0.95)
        planes.append(dict(
            n=n.tolist(), d=d,
            z_at_centroid=float(p["centroid"][2]),
            support=support,
            n_inliers=p["n_inliers"],
            extent_min=p["extent"][0].tolist(), extent_max=p["extent"][1].tolist()))

    objects = []
    if ball_detector is not None:
        box, conf = ball_detector.detect(color)
        if box:
            b = ball_from_box(box, conf, depth, K)
            if b is not None and b["fit_ok"]:
                c = R_cb @ np.asarray(b["center3d"]) + t_cb
                resting = None
                for i, pl in enumerate(planes):
                    if not pl["support"]:
                        continue
                    gap = (c[2] - b["radius_m"]) - pl["z_at_centroid"]
                    if abs(gap) < 0.012:
                        resting = i
                        break
                objects.append(dict(cls="ball", center=c.tolist(),
                                    radius=b["radius_m"], conf=float(conf),
                                    inliers=b["inliers"], resting_plane=resting))
    return dict(created=datetime.now().isoformat(),
                handeye=HANDEYE.stat().st_mtime if HANDEYE.exists() else None,
                planes=planes, objects=objects)


def save(model, path=OUT):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(model, open(path, "w"), indent=2)
    return path


def load(path=OUT):
    return json.load(open(path)) if Path(path).exists() else None


def summarize(model):
    lines = []
    for i, p in enumerate(model["planes"]):
        kind = "SUPPORT" if p["support"] else "plane"
        lines.append(f"  {kind} {i}: z={p['z_at_centroid'] * 1000:+.0f}mm  "
                     f"inliers={p['n_inliers']}")
    for o in model["objects"]:
        c = np.array(o["center"]) * 1000
        on = (f"on plane {o['resting_plane']}" if o["resting_plane"] is not None
              else "raised (holder?)")
        lines.append(f"  {o['cls']}: [{c[0]:.0f} {c[1]:.0f} {c[2]:.0f}]mm "
                     f"conf={o['conf']:.2f} — {on}")
    return "\n".join(lines)


def main():
    import torch
    from handeye_calib import Realsense
    from pick_ball import BallDetector

    he = json.load(open(HANDEYE))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])
    det = BallDetector("cuda" if torch.cuda.is_available() else "cpu")
    cam = Realsense()
    try:
        color, depth, K = cam.grab()
    finally:
        cam.stop()
    model = scan(color, depth, K, R_cb, t_cb, ball_detector=det)
    print(summarize(model))
    print("saved", save(model))


if __name__ == "__main__":
    main()
