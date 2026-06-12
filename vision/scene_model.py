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
    from cloud import (crop_z, deproject, euclidean_clusters, extract_planes,
                       voxel_downsample)

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

    # detector-free object discovery: voxelize, drop plane-adjacent points, cluster
    # what sticks up. Class-agnostic "blob" cards — the robot's own arm shows up too
    # (v0 doesn't subtract it; consumers can match blobs against FK later).
    vox = voxel_downsample(base, voxel=0.008)
    off_plane = np.ones(len(vox), bool)
    for pl in planes:
        n = np.array(pl["n"])
        off_plane &= np.abs(vox @ n + pl["d"]) > 0.010
    floor = min((pl["z_at_centroid"] for pl in planes if pl["support"]), default=0.0)
    blobs_src = vox[off_plane & (vox[:, 2] > floor + 0.008)]
    blobs = []
    if len(blobs_src) > 30:
        for cl in euclidean_clusters(blobs_src, radius=0.022, min_pts=12)[:8]:
            P = blobs_src[cl]
            lo, hi = P.min(0), P.max(0)
            rest = None
            for i, pl in enumerate(planes):
                if pl["support"] and abs(lo[2] - pl["z_at_centroid"]) < 0.015:
                    rest = i
                    break
            blobs.append(dict(cls="blob", n_pts=int(len(cl)),
                              centroid=P.mean(0).tolist(),
                              extent_min=lo.tolist(), extent_max=hi.tolist(),
                              height=float(hi[2] - lo[2]), resting_plane=rest))

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
                planes=planes, objects=objects, blobs=blobs)


def add_captions(model_dict, color, K, R_cb, t_cb):
    """Name the blobs: Florence-2-base region captions on each blob's projected bbox
    (probe 2026-06-12: 'a printer', 'bunch of wires', 'a basketball sitting in a
    basketball hoop' for the ball-on-spool — geometry from depth, semantics from RGB).
    Lazy-loads the model (~0.5 GB fp16); needs `timm`."""
    import cv2
    import torch
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor
    mid = "microsoft/Florence-2-base"
    proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
    net = (AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True,
                                                torch_dtype=torch.float16)
           .to("cuda" if torch.cuda.is_available() else "cpu").eval())
    rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]

    def proj(p):
        pc = R_cb.T @ (np.asarray(p) - t_cb)
        return int(K["fx"] * pc[0] / pc[2] + K["ppx"]), int(K["fy"] * pc[1] / pc[2] + K["ppy"])

    for b in model_dict.get("blobs", []):
        lo, hi = np.array(b["extent_min"]), np.array(b["extent_max"])
        cs = [proj([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
              for z in (lo[2], hi[2])]
        us, vs = [c[0] for c in cs], [c[1] for c in cs]
        x1, y1 = max(min(us) - 8, 0), max(min(vs) - 8, 0)
        x2, y2 = min(max(us) + 8, W - 1), min(max(vs) + 8, H - 1)
        if x2 - x1 < 16 or y2 - y1 < 16:
            continue
        inp = proc(text="<CAPTION>", images=Image.fromarray(rgb[y1:y2, x1:x2]),
                   return_tensors="pt").to(net.device, torch.float16)
        out = net.generate(input_ids=inp["input_ids"], pixel_values=inp["pixel_values"],
                           max_new_tokens=40, num_beams=3)
        b["caption"] = proc.batch_decode(out, skip_special_tokens=True)[0].strip()
    return model_dict


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
    for b in model.get("blobs", []):
        c = np.array(b["centroid"]) * 1000
        fx = (np.array(b["extent_max"]) - np.array(b["extent_min"])) * 1000
        on = (f"on plane {b['resting_plane']}" if b["resting_plane"] is not None
              else "floating/attached")
        cap = f" — \"{b['caption']}\"" if b.get("caption") else ""
        lines.append(f"  blob: ctr=[{c[0]:.0f} {c[1]:.0f} {c[2]:.0f}]mm "
                     f"footprint {fx[0]:.0f}x{fx[1]:.0f} h={b['height'] * 1000:.0f}mm "
                     f"({b['n_pts']} vox) — {on}{cap}")
    return "\n".join(lines)


def main():
    import argparse
    import torch
    from handeye_calib import Realsense
    from pick_ball import BallDetector

    ap = argparse.ArgumentParser()
    ap.add_argument("--caption", action="store_true",
                    help="Name the blobs with Florence-2 (downloads ~0.5 GB once).")
    args = ap.parse_args()

    he = json.load(open(HANDEYE))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])
    det = BallDetector("cuda" if torch.cuda.is_available() else "cpu")
    cam = Realsense()
    try:
        color, depth, K = cam.grab()
    finally:
        cam.stop()
    model = scan(color, depth, K, R_cb, t_cb, ball_detector=det)
    if args.caption:
        model = add_captions(model, color, K, R_cb, t_cb)
    print(summarize(model))
    print("saved", save(model))


if __name__ == "__main__":
    main()
