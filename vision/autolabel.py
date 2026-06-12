#!/usr/bin/env python
"""Auto-label scene frames with Grounding DINO -> YOLO-format dataset.

GDINO is the slow-but-trusted teacher (validated on this scene: ball 0.75+, hearts
boxed reliably); the goal is a fast yolov8n student for the real-time loop.

Classes:
    0 ball         GDINO "basketball." best box
    1 heart_pink   GDINO "heart." boxes gated by the pink wrap-around band
                   (same gates as handeye_calib — yellow heart excluded for now:
                   it collides with the guitar pick / wood tones, needs human QC)

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
from handeye_calib import HeartDetector  # noqa: E402  (pink gates live there)
import handeye_calib  # noqa: E402

BALL_THR = 0.35


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


def label_frame(det, color, gripper_uv=None, gate_px=90):
    """Return list of (cls, x1,y1,x2,y2). If gripper_uv (the FK-projected gripper
    pixel) is given, a heart label must sit within gate_px of it — kinematics-backed
    QC that kills color false-positives (the red clamp incident)."""
    H, W = color.shape[:2]
    out = []
    balls = gdino_boxes(det, color, "basketball.", BALL_THR)
    if balls:
        (x1, y1, x2, y2), s = max(balls, key=lambda b: b[1])
        if max(x2 - x1, y2 - y1) < 0.4 * W:               # reject whole-table boxes
            out.append((0, x1, y1, x2, y2))
    uv, boxes = det.marker_uv(color)                       # hearts + pink gating
    if uv is not None:
        if gripper_uv is not None and np.hypot(uv[0] - gripper_uv[0],
                                               uv[1] - gripper_uv[1]) > gate_px:
            return out                                     # heart far from gripper: junk
        for (x1, y1, x2, y2), conf, px, frac in boxes:
            if ((x1 + x2) // 2, (y1 + y2) // 2) == uv:
                out.append((1, x1, y1, x2, y2))
                break
    return out


def fk_gripper_uv(joints, K):
    """Project the FK gripper position into the image via the hand-eye transform."""
    import json
    from handeye_calib import make_fk
    he = json.load(open(Path(__file__).resolve().parent.parent / "outputs/calib/handeye.json"))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])
    if not hasattr(fk_gripper_uv, "_fk"):
        fk_gripper_uv._fk = make_fk()
    _, t_w = fk_gripper_uv._fk(joints)
    p_cam = R_cb.T @ (t_w - t_cb)                          # base -> camera frame
    if p_cam[2] <= 0.05:
        return None
    return (int(K["fx"] * p_cam[0] / p_cam[2] + K["ppx"]),
            int(K["fy"] * p_cam[1] / p_cam[2] + K["ppy"]))


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    for sub in ("images", "labels", "overlays"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    import torch
    det = HeartDetector("cuda" if torch.cuda.is_available() else "cpu")
    frames = sorted(p for p in src.rglob("*.png")
                    if not any(t in p.name for t in ("overlay", "_det", "depth")))
    n_ball = n_heart = 0
    for i, fp in enumerate(frames):
        color = cv2.imread(str(fp))
        if color is None:
            continue
        H, W = color.shape[:2]
        guv = None
        jf = fp.parent / fp.name.replace("frame_", "joints_").replace(".png", ".json")
        if jf.exists():                                    # collect_marker_data session
            import json as _json
            meta = _json.load(open(jf))
            guv = fk_gripper_uv(meta["joints"], meta["K"])
        anns = label_frame(det, color, gripper_uv=guv)
        name = f"{i:03d}_{fp.stem}"
        cv2.imwrite(str(dst / "images" / f"{name}.png"), color)
        with open(dst / "labels" / f"{name}.txt", "w") as f:
            for cls, x1, y1, x2, y2 in anns:
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                bw, bh = (x2 - x1) / W, (y2 - y1) / H
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        vis = color.copy()
        for cls, x1, y1, x2, y2 in anns:
            col = (0, 165, 255) if cls == 0 else (255, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, ["ball", "heart_pink"][cls], (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        cv2.imwrite(str(dst / "overlays" / f"{name}.png"), vis)
        n_ball += sum(1 for a in anns if a[0] == 0)
        n_heart += sum(1 for a in anns if a[0] == 1)
        print(f"{name}: {len(anns)} labels")
    print(f"\n{len(frames)} frames: {n_ball} ball, {n_heart} heart_pink -> {dst}")
    print("Spot-check overlays/ before training.")


if __name__ == "__main__":
    main()
