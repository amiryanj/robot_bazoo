#!/usr/bin/env python
"""
Core vision primitives for the top-down Realsense: fit the table plane and
localize the ball in 3D (camera frame). Robust = depth geometry + color, not
Hough on pixels — the depth + sphere model give a real metric 3D centre.

Functions (reusable):
    load_frame(dir)            -> color (BGR), depth (m), K
    deproject(depth, K)        -> HxWx3 point cloud in the camera frame (metres)
    fit_table_plane(points)    -> (normal, offset, inlier_mask)   RANSAC
    localize_ball(color, depth, K, plane=None)
                               -> dict(center3d, radius_m, uv, mask) or None

Run directly to verify on a captured frame (writes overlay.png):
    python vision/ball.py                       # latest outputs/vision/* capture
    python vision/ball.py outputs/vision/<ts>
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Orange mini-basketball in HSV (OpenCV H is 0-180). Tuned on the 01-50-23 frame;
# wide enough for the ball, tight enough to reject the red arm (H~0) and wood (low S).
ORANGE_LO = np.array([5, 110, 90])
ORANGE_HI = np.array([22, 255, 255])
WORKSPACE_Z = (0.15, 1.2)        # metres; ignore the far room / invalid depth


def load_frame(d):
    d = Path(d)
    color = cv2.imread(str(d / "color.png"))
    depth = np.load(d / "depth.npy")
    K = json.load(open(d / "intrinsics.json"))
    return color, depth, K


def deproject(depth, K):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z = depth
    x = (u - K["ppx"]) * z / K["fx"]
    y = (v - K["ppy"]) * z / K["fy"]
    return np.stack([x, y, z], axis=-1)


def fit_table_plane(points, iters=600, thresh=0.006, seed=0):
    """RANSAC plane fit. points: (N,3). Returns (normal unit, offset d, inlier idx)
    with plane defined by n·p + d = 0. Refit on inliers for accuracy."""
    rng = np.random.default_rng(seed)
    N = len(points)
    best = None
    for _ in range(iters):
        p = points[rng.choice(N, 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ p[0]
        inl = np.abs(points @ n + d) < thresh
        if best is None or inl.sum() > best[2].sum():
            best = (n, d, inl)
    n, d, inl = best
    # least-squares refit on inliers (centroid + SVD)
    P = points[inl]
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)
    n = Vt[2]
    d = -n @ c
    return n, d, inl


def localize_ball(color, depth, K, plane=None, min_area=80):
    """Find the orange ball -> 3D centre (camera frame) + radius. None if not found."""
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    n_lbl, lbl, stats, cents = cv2.connectedComponentsWithStats(mask)
    if n_lbl <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))     # largest blob
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        return None
    blob = (lbl == i)

    # 3D points of the ball surface that have valid, in-workspace depth
    pts = deproject(depth, K)[blob & (depth > WORKSPACE_Z[0]) & (depth < WORKSPACE_Z[1])]
    if len(pts) < 20:
        return None

    surface = np.median(pts, axis=0)                        # robust surface centroid
    # pixel radius -> metric radius at the surface depth
    area_px = stats[i, cv2.CC_STAT_AREA]
    r_px = np.sqrt(area_px / np.pi)
    radius_m = float(r_px * surface[2] / K["fx"])

    # true centre sits ~one radius behind the visible cap, along the plane normal
    # (toward the table). If we have the plane, push the surface centroid inward.
    center = surface.copy()
    if plane is not None:
        n, d = plane
        n = n if (n @ surface + d) > 0 else -n              # point n away from table
        center = surface - n * radius_m
    else:
        center = surface + np.array([0, 0, radius_m])       # fallback: +Z (into scene)

    uv = tuple(int(c) for c in cents[i])
    return dict(center3d=center, surface3d=surface, radius_m=radius_m, uv=uv, mask=blob)


# ── verification harness ──────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        d = sys.argv[1]
    else:
        caps = sorted(glob.glob("/home/javad/workspace/lerobot_all/outputs/vision/*"))
        d = caps[-1]
    print(f"frame: {d}")
    color, depth, K = load_frame(d)

    pts = deproject(depth, K)
    flat = pts.reshape(-1, 3)
    valid = flat[(flat[:, 2] > WORKSPACE_Z[0]) & (flat[:, 2] < WORKSPACE_Z[1])]
    n, pd, inl = fit_table_plane(valid)
    print(f"table plane: normal=({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f})  "
          f"offset={pd:+.3f}  inliers={inl.sum()}/{len(valid)} "
          f"({100*inl.sum()/len(valid):.0f}%)")

    ball = localize_ball(color, depth, K, plane=(n, pd))
    overlay = color.copy()
    if ball is None:
        print("ball: NOT FOUND")
    else:
        c = ball["center3d"]
        print(f"ball uv={ball['uv']}  surface_z={ball['surface3d'][2]:.3f}m  "
              f"radius={ball['radius_m']*1000:.0f}mm")
        print(f"ball center3d (cam frame) = "
              f"({c[0]*1000:+.0f}, {c[1]*1000:+.0f}, {c[2]*1000:+.0f}) mm")
        overlay[ball["mask"]] = (0, 255, 0)
        cv2.circle(overlay, ball["uv"], 6, (0, 0, 255), -1)
        cv2.circle(overlay, ball["uv"], int(np.sqrt(ball["mask"].sum() / np.pi)),
                   (255, 0, 0), 2)

    out = str(Path(d) / "overlay.png")
    cv2.imwrite(out, overlay)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
