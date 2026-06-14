#!/usr/bin/env python
"""Auto-label scene frames with Grounding DINO -> YOLO-format dataset.

GDINO is the slow-but-trusted teacher (validated on this scene: ball 0.75+); the goal is
a fast yolov8n student for the real-time loop. Single class now: the ball (the old
heart_pink class is gone — the gripper marker is AprilTags, detected by ArUco, not YOLO).

Classes:
    0 ball         GDINO "basketball." best box

Usage:
    python vision/autolabel.py <frames_dir> <out_dir>   # writes images/ labels/ overlays/
Every frame gets an overlay for human spot-checking. Frames with no detections are
still kept (negatives help).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

BALL_THR = 0.35
GDINO_ID = "IDEA-Research/grounding-dino-tiny"


class Gdino:
    """Minimal Grounding DINO teacher (zero-shot boxes for a text prompt)."""

    def __init__(self, device):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.device = device
        self.proc = AutoProcessor.from_pretrained(GDINO_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(device).eval()


def gdino_boxes(det, color_bgr, prompt, thr):
    from PIL import Image
    img = Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
    inp = det.proc(images=img, text=prompt, return_tensors="pt").to(det.device)
    with det.torch.no_grad():
        out = det.model(**inp)
    res = det.proc.post_process_grounded_object_detection(
        out, inp.input_ids, threshold=thr, text_threshold=thr,
        target_sizes=[img.size[::-1]])[0]
    return [([int(v) for v in b], float(s))
            for b, s in zip(res["boxes"].tolist(), res["scores"].tolist())]


def label_frame(det, color):
    """Return list of (cls, x1,y1,x2,y2) — ball only (class 0)."""
    H, W = color.shape[:2]
    out = []
    balls = gdino_boxes(det, color, "basketball.", BALL_THR)
    if balls:
        (x1, y1, x2, y2), s = max(balls, key=lambda b: b[1])
        if max(x2 - x1, y2 - y1) < 0.4 * W:               # reject whole-table boxes
            out.append((0, x1, y1, x2, y2))
    return out


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    for sub in ("images", "labels", "overlays"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    import torch
    det = Gdino("cuda" if torch.cuda.is_available() else "cpu")
    frames = sorted(p for p in src.rglob("*.png")
                    if not any(t in p.name for t in ("overlay", "_det", "depth")))
    n_ball = 0
    for i, fp in enumerate(frames):
        color = cv2.imread(str(fp))
        if color is None:
            continue
        H, W = color.shape[:2]
        anns = label_frame(det, color)
        name = f"{i:03d}_{fp.stem}"
        cv2.imwrite(str(dst / "images" / f"{name}.png"), color)
        with open(dst / "labels" / f"{name}.txt", "w") as f:
            for cls, x1, y1, x2, y2 in anns:
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        vis = color.copy()
        for cls, x1, y1, x2, y2 in anns:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(vis, "ball", (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.imwrite(str(dst / "overlays" / f"{name}.png"), vis)
        n_ball += len(anns)
        print(f"{name}: {len(anns)} labels")
    print(f"\n{len(frames)} frames: {n_ball} ball -> {dst}")
    print("Spot-check overlays/ before training.")


if __name__ == "__main__":
    main()
