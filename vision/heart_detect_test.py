#!/usr/bin/env python
"""Capture N frames from the top-down Realsense and run the heart detector on each,
to validate detection across poses before trusting it for hand-eye calibration.

Torque is off after a calibration run, so between grabs just HAND-MOVE the limp arm to
a new pose. Each frame: detect hearts (Grounding DINO), pick the teal wrist marker, save
an annotated overlay + the raw frame + per-frame JSON to outputs/vision/<ts>_hearts/.

    python vision/heart_detect_test.py            # 8 frames, 3 s apart
    python vision/heart_detect_test.py -n 12 --interval 4
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vision"))
from handeye_calib import HeartDetector, Realsense, MIN_TEAL_PX  # noqa: E402


def annotate(bgr, boxes, uv):
    vis = bgr.copy()
    for (x1, y1, x2, y2), conf, teal, frac in boxes:
        pick = uv is not None and ((x1 + x2) // 2, (y1 + y2) // 2) == uv
        col = (0, 255, 0) if pick else (0, 200, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 3 if pick else 2)
        cv2.putText(vis, f"{conf:.2f} f={frac:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    if uv is not None:
        cv2.circle(vis, uv, 5, (0, 255, 0), -1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--frames", type=int, default=8)
    ap.add_argument("--interval", type=float, default=3.0, help="seconds between grabs")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading heart detector on {device} ...")
    det = HeartDetector(device)
    cam = Realsense()

    out = ROOT / "outputs/vision" / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_hearts"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Capturing {args.frames} frames -> {out}\n"
          f"Hand-move the limp arm between grabs to vary the pose.\n")

    summary = []
    try:
        for i in range(args.frames):
            for s in range(int(args.interval), 0, -1):
                print(f"  frame {i+1}/{args.frames} in {s}s ...", end="\r", flush=True)
                time.sleep(1)
            color, depth, K = cam.grab()
            uv, boxes = det.teal_uv(color)
            cv2.imwrite(str(out / f"frame_{i:02d}.png"), color)
            cv2.imwrite(str(out / f"frame_{i:02d}_det.png"), annotate(color, boxes, uv))
            rec = dict(frame=i, teal_uv=uv, n_hearts=len(boxes),
                       boxes=[dict(box=b, conf=c, teal_px=t, teal_frac=round(f, 3))
                              for b, c, t, f in boxes])
            summary.append(rec)
            tag = f"teal@{uv}" if uv else "NO teal marker"
            print(f"  frame {i+1}/{args.frames}: {len(boxes)} heart(s), {tag}        ")
    finally:
        cam.stop()

    json.dump(summary, open(out / "summary.json", "w"), indent=2)
    hit = sum(1 for r in summary if r["teal_uv"])
    print(f"\nTeal marker found in {hit}/{len(summary)} frames. Overlays: {out}/frame_*_det.png")


if __name__ == "__main__":
    main()
