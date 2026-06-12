#!/usr/bin/env python
"""Benchmark ball localization on saved sessions where the BALL WAS STATIC: the
spread of the estimated base-frame centre across frames is pure estimator error
(plus detection variation). Compares the legacy box-median estimator against the
known-radius sphere fit (vision/cloud.py).

    python vision/bench_localize.py <session_dir> [...]
Session dir = frames as *.png + depth as *.npy pairs (collect_marker_data or the
exposure-sweep layout)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ball import WORKSPACE_Z  # noqa: E402
from ball_yolo import BALL_RADIUS_M, ball_from_box  # noqa: E402

HANDEYE = Path(__file__).resolve().parent.parent / "outputs/calib/handeye.json"


def legacy_center(box, depth, K):
    """The old estimator: median depth of the central quarter + box-size radius."""
    x1, y1, x2, y2 = box
    u, v = (x1 + x2) // 2, (y1 + y2) // 2
    qx, qy = (x2 - x1) // 4, (y2 - y1) // 4
    win = depth[y1 + qy:y2 - qy, x1 + qx:x2 - qx]
    win = win[(win > WORKSPACE_Z[0]) & (win < WORKSPACE_Z[1])]
    if len(win) < 10:
        return None
    z = float(np.median(win))
    surface = np.array([(u - K["ppx"]) * z / K["fx"],
                        (v - K["ppy"]) * z / K["fy"], z])
    r = float(((x2 - x1) + (y2 - y1)) / 4 * z / K["fx"])
    return surface + np.array([0, 0, r])


def depth_for(png):
    cands = [png.with_name(png.stem.replace("frame_", "depth_") + ".npy"),
             png.with_name(png.name.replace(".png", "_depth.npy"))]
    for c in cands:
        if c.exists():
            return c
    return None


def main():
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pick_ball import BallDetector
    he = json.load(open(HANDEYE))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])
    det = BallDetector("cuda" if torch.cuda.is_available() else "cpu")

    for d in sys.argv[1:]:
        d = Path(d)
        K = None
        for jf in sorted(d.glob("joints_*.json")) + sorted(d.glob("*intrinsics*.json")):
            K = json.load(open(jf)).get("K") or json.load(open(jf))
            break
        old_pts, new_pts, n_det, n_gate = [], [], 0, 0
        for png in sorted(d.glob("*.png")):
            if "_det" in png.name or "overlay" in png.name:
                continue
            dn = depth_for(png)
            if dn is None:
                continue
            depth = np.load(dn)
            color = cv2.imread(str(png))
            box, conf = det.detect(color)
            if not box:
                continue
            n_det += 1
            if K is None:
                continue
            o = legacy_center(box, depth, K)
            b = ball_from_box(box, conf, depth, K)
            if o is not None:
                old_pts.append(R_cb @ o + t_cb)
            if b is not None and b["fit_ok"]:
                n_gate += 1
                new_pts.append(R_cb @ np.asarray(b["center3d"]) + t_cb)

        def spread(pts):
            if len(pts) < 2:
                return None
            P = np.array(pts) * 1000
            return (P.std(0).round(1).tolist(), np.round(P.mean(0)).astype(int).tolist())

        print(f"\n{d.name}: detected {n_det}, sphere-fit passed gate {n_gate}")
        for name, pts in (("legacy", old_pts), ("sphere-fit", new_pts)):
            s = spread(pts)
            if s:
                print(f"  {name:10s} n={len(pts):2d}  std(xyz)={s[0]} mm   mean={s[1]} mm")
            else:
                print(f"  {name:10s} n={len(pts)} — not enough")


if __name__ == "__main__":
    main()
