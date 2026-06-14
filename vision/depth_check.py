"""One-shot depth validity check at max res: where does the D455 actually have depth?

Captures aligned color (1280x800) + depth, detects the finger ArUco tags, and reports
per-region depth coverage + median range: each tag quad, the white plate, and the
overall frame. Compares depth-at-tag vs PnP range. Saves overlays to outputs/vision/.

python vision/depth_check.py
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

TAG_SIDE = 0.0274          # fitted print size from tag_sweep (nominal 24mm printed big)
OUT = Path(__file__).resolve().parent.parent / "outputs/vision"


def region_stats(depth_m, mask):
    d = depth_m[mask]
    valid = d > 0
    cov = valid.mean() * 100 if d.size else 0.0
    med = np.median(d[valid]) * 1000 if valid.any() else float("nan")
    return cov, med


def main():
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, 1280, 800, rs.format.bgr8, 30)
    dw, dh = (int(v) for v in (sys.argv[1:3] or ["1280", "720"]))
    cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    for _ in range(15):                      # let auto-exposure settle
        frames = pipe.wait_for_frames()
    frames = align.process(frames)
    c, d = frames.get_color_frame(), frames.get_depth_frame()
    intr = c.profile.as_video_stream_profile().intrinsics
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]])
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    color = np.asanyarray(c.get_data()).copy()
    depth_m = np.asanyarray(d.get_data()).astype(np.float32) * scale
    pipe.stop()

    h, w = depth_m.shape
    print(f"frame {w}x{h}, depth valid overall: {(depth_m > 0).mean() * 100:.1f}%")

    # tags
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(dic, par)
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
    ids = ids.flatten().tolist() if ids is not None else []
    print(f"tags detected: {ids}")

    obj = TAG_SIDE / 2 * np.array(
        [[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], np.float32)
    for tid, quad in zip(ids, corners):
        quad = quad.reshape(4, 2)
        mask = np.zeros((h, w), bool)
        cv2.fillPoly(mask.view(np.uint8).reshape(h, w), [quad.astype(np.int32)], 1)
        cov, med = region_stats(depth_m, mask)
        ok, rvec, tvec = cv2.solvePnP(obj, quad.astype(np.float32), K, None,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        pnp_z = tvec[2, 0] * 1000 if ok else float("nan")
        print(f"  tag{tid}: depth coverage {cov:5.1f}%  median {med:6.1f}mm"
              f"  | PnP range {pnp_z:6.1f}mm  (diff {med - pnp_z:+.1f}mm)")
        cv2.polylines(color, [quad.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(color, f"tag{tid} d={med:.0f} pnp={pnp_z:.0f}",
                    tuple(quad[0].astype(int) - [0, 10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # white plate sample: a patch around the frame centre-ish where the plate is
    cx, cy = w // 2, int(h * 0.55)
    plate = np.zeros((h, w), bool)
    plate[cy - 60:cy + 60, cx - 80:cx + 80] = True
    cov, med = region_stats(depth_m, plate)
    print(f"  plate patch @({cx},{cy}): coverage {cov:.1f}%  median {med:.1f}mm")
    cv2.rectangle(color, (cx - 80, cy - 60), (cx + 80, cy + 60), (255, 200, 0), 2)

    # overlays: holes tinted red on color; turbo-colored depth
    holes = depth_m == 0
    color[holes] = (color[holes] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)
    dvis = cv2.applyColorMap(
        cv2.convertScaleAbs(np.clip(depth_m, 0, 0.8) / 0.8 * 255), cv2.COLORMAP_TURBO)
    dvis[holes] = 0

    out = OUT / f"depth_check_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    out.mkdir(parents=True)
    cv2.imwrite(str(out / "holes_on_color.png"), color)
    cv2.imwrite(str(out / "depth_turbo.png"), dvis)
    print(f"overlays -> {out}")


if __name__ == "__main__":
    main()
