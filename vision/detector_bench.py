#!/usr/bin/env python
"""Benchmark the fast student (yolov8n fine-tune) against the slow teacher (GDINO).

For every val frame: GDINO "basketball." boxes are the reference; report the student's
per-class agreement (hit = IoU>0.5 with the reference box) and the speed of both on this
machine. Single class: the ball (the heart_pink class is gone — gripper marker is AprilTags).

    python vision/detector_bench.py <weights.pt> <val_images_dir>
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autolabel import Gdino, label_frame  # noqa: E402  (GDINO reference labeling)

CLASSES = ["ball"]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def main():
    import torch
    from ultralytics import YOLO
    weights, val_dir = sys.argv[1], Path(sys.argv[2])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    student = YOLO(weights)
    teacher = Gdino(device)
    frames = sorted(val_dir.glob("*.png"))
    print(f"{len(frames)} frames, device={device}\n")

    hits = {c: [0, 0] for c in range(len(CLASSES))}      # cls -> [agree, ref_total]
    fp = 0
    t_s = t_t = 0.0
    for fpth in frames:
        img = cv2.imread(str(fpth))
        t0 = time.perf_counter()
        ref = label_frame(teacher, img)                   # [(cls,x1,y1,x2,y2)]
        t_t += time.perf_counter() - t0
        t0 = time.perf_counter()
        res = student(img, conf=0.4, verbose=False)[0]
        t_s += time.perf_counter() - t0
        pred = [(int(c), *[int(v) for v in b])
                for c, b in zip(res.boxes.cls, res.boxes.xyxy)]
        for rc, *rb in ref:
            hits[rc][1] += 1
            if any(pc == rc and iou(rb, pb) > 0.5 for pc, *pb in pred):
                hits[rc][0] += 1
        for pc, *pb in pred:
            if not any(rc == pc and iou(pb, rb) > 0.5 for rc, *rb in ref):
                fp += 1

    for c, name in enumerate(CLASSES):
        a, t = hits[c]
        print(f"{name:12s} agreement with GDINO: {a}/{t}")
    print(f"student extra boxes (no GDINO match): {fp}")
    print(f"\nspeed/frame: yolov8n {1000 * t_s / len(frames):.0f} ms   "
          f"GDINO {1000 * t_t / len(frames):.0f} ms   "
          f"({t_t / max(t_s, 1e-9):.0f}x)")


if __name__ == "__main__":
    main()
