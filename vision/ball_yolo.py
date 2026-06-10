#!/usr/bin/env python
"""
YOLO ball detection -> 3D ball point (camera frame) for the top-down Realsense.

Replaces the brittle color+depth localizer in ball.py: a pretrained basketball
YOLO (vision/models/basketball.pt, classes Basketball/Hoop) gives a clean 2-D box
even though the wooden table reads orange in HSV. We back-project the box centre
through the aligned depth to a metric 3-D point and keep fit_table_plane (from
ball.py) for the workspace reference.

Functions (reusable):
    load_model(path)                  -> YOLO
    localize_ball_yolo(color, depth, K, model, plane=None, conf=0.25)
        -> dict(center3d, surface3d, radius_m, uv, box, conf) or None

Run directly to verify on a captured frame (writes yolo_overlay.png):
    python vision/ball_yolo.py                      # latest outputs/vision/* capture
    python vision/ball_yolo.py outputs/vision/<ts>
"""
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ball import load_frame, deproject, fit_table_plane, WORKSPACE_Z  # noqa: E402

MODEL = Path(__file__).resolve().parent / "models" / "basketball.pt"
BALL_CLASS = "Basketball"


def load_model(path=MODEL):
    from ultralytics import YOLO
    return YOLO(str(path))


def localize_ball_yolo(color, depth, K, model, plane=None, conf=0.25):
    """Detect the ball with YOLO -> 3D centre (camera frame) + radius. None if not found.

    The box centre pixel is back-projected through its depth to the visible cap
    (top of the ball, top-down). The true centre sits ~one radius behind that,
    toward the table (along the plane normal if given, else +Z)."""
    res = model(color, conf=conf, verbose=False)[0]

    best = None                                          # highest-conf Basketball box
    for c, p, b in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy):
        if res.names[int(c)] == BALL_CLASS and (best is None or float(p) > best[0]):
            best = (float(p), [int(v) for v in b])
    if best is None:
        return None
    score, (x1, y1, x2, y2) = best
    u, v = (x1 + x2) // 2, (y1 + y2) // 2

    # robust surface depth: median of valid in-workspace depth in the box's central
    # quarter (mostly ball, avoids the table ring around the box edges)
    qx, qy = (x2 - x1) // 4, (y2 - y1) // 4
    win = depth[y1 + qy:y2 - qy, x1 + qx:x2 - qx]
    win = win[(win > WORKSPACE_Z[0]) & (win < WORKSPACE_Z[1])]
    if len(win) < 10:
        return None
    z = float(np.median(win))

    surface = np.array([(u - K["ppx"]) * z / K["fx"],
                        (v - K["ppy"]) * z / K["fy"], z])

    r_px = ((x2 - x1) + (y2 - y1)) / 4                   # box half-extent in pixels
    radius_m = float(r_px * z / K["fx"])

    center = surface.copy()
    if plane is not None:
        n, d = plane
        n = n if (n @ surface + d) > 0 else -n           # point n away from table
        center = surface - n * radius_m
    else:
        center = surface + np.array([0, 0, radius_m])     # fallback: +Z (into scene)

    return dict(center3d=center, surface3d=surface, radius_m=radius_m,
                uv=(u, v), box=(x1, y1, x2, y2), conf=score)


# ── verification harness ──────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        d = sys.argv[1]
    else:
        caps = sorted(glob.glob("/home/javad/workspace/lerobot_all/outputs/vision/*"))
        d = caps[-1]
    print(f"frame: {d}")
    color, depth, K = load_frame(d)

    pts = deproject(depth, K).reshape(-1, 3)
    valid = pts[(pts[:, 2] > WORKSPACE_Z[0]) & (pts[:, 2] < WORKSPACE_Z[1])]
    n, pd, inl = fit_table_plane(valid)
    print(f"table plane: normal=({n[0]:+.2f},{n[1]:+.2f},{n[2]:+.2f})  "
          f"offset={pd:+.3f}  inliers={inl.sum()}/{len(valid)} "
          f"({100 * inl.sum() / len(valid):.0f}%)")

    model = load_model()
    ball = localize_ball_yolo(color, depth, K, model, plane=(n, pd))

    overlay = color.copy()
    if ball is None:
        print("ball: NOT FOUND")
    else:
        c = ball["center3d"]
        x1, y1, x2, y2 = ball["box"]
        print(f"ball conf={ball['conf']:.2f}  box={ball['box']}  uv={ball['uv']}  "
              f"surface_z={ball['surface3d'][2]:.3f}m  radius={ball['radius_m'] * 1000:.0f}mm")
        print(f"ball center3d (cam frame) = "
              f"({c[0] * 1000:+.0f}, {c[1] * 1000:+.0f}, {c[2] * 1000:+.0f}) mm")
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(overlay, ball["uv"], 4, (0, 0, 255), -1)
        label = f"ball {ball['conf']:.2f}  z={ball['surface3d'][2]:.2f}m"
        cv2.putText(overlay, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        xyz = f"xyz=({c[0]*1000:+.0f},{c[1]*1000:+.0f},{c[2]*1000:+.0f})mm"
        cv2.putText(overlay, xyz, (10, overlay.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    out = str(Path(d) / "yolo_overlay.png")
    cv2.imwrite(out, overlay)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
