#!/usr/bin/env python
"""
Hand-eye calibration (eye-to-hand): estimate T_cam->base for the top-down Realsense.

Marker = the PINK heart sticker on a gripper finger. The fingers move with the gripper
joint, so KEEP THE GRIPPER OPENING FIXED (ideally closed) for the whole capture run —
then the heart is rigid w.r.t. the `gripper` body (the wrist_roll part) and the existing
FK + constant-offset solver apply unchanged. We drive the arm to many poses (gamepad
teleop), and at each captured pose:
  - detect the pink heart in the color image -> centroid pixel,
  - back-project through the aligned depth -> 3-D point in the CAMERA frame,
  - read joint angles -> MuJoCo FK of the `gripper` body -> its pose in the BASE frame.

Then solve, jointly, the camera->base transform AND the (unknown, constant) offset
of the heart in the gripper-body frame:

    R_cb . p_cam_i + t_cb  =  R_body_i . o + t_body_i        (per pose i)

9 unknowns (R_cb:3, t_cb:3, o:3), 3 eqs/pose -> ~10-15 poses is plenty.
Output: outputs/calib/handeye.json  (R, t, marker offset, RMS residual).

Usage:
    python vision/handeye_calib.py --selftest      # offline: verify the solver, no hardware
    python vision/handeye_calib.py                 # full calibration (arm + realsense + gamepad)

Live preview streams to Rerun so you can aim. Typed commands while it runs:
    c / <Enter>  capture the current pose      u  undo last      q  finish & solve
"""
import argparse
import json
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SERIAL = "117222251972"
XML = str(ROOT / "SO-ARM100/Simulation/SO101/scene.xml")
MARKER_BODY = "gripper"            # finger heart, rigid w.r.t. this body at fixed opening
OUT = ROOT / "outputs/calib"

WORKSPACE_Z = (0.20, 1.2)          # metres; valid marker depth band

# The calibration marker is the PINK finger heart. We find hearts with a learned zero-shot
# detector (Grounding DINO) — robust to clutter (piano keys, wood) that fools colour
# segmentation — then pick the pink one. Colour ID *inside* a confirmed heart box is
# reliable; it's colour-only segmentation across the whole frame that's not.
GDINO_ID = "IDEA-Research/grounding-dino-tiny"
GDINO_THR = 0.12                   # low: catches faint heart boxes (conf 0.18-0.44). Safe
                                   # because the HSV band below — not the threshold —
                                   # rejects non-pink clutter.
# Pink is the HSV wrap-around colour: the sticker measured H 160-166 in midday light but
# H~0-2 under warmer light (hue wraps past 180). So gate BOTH hue ends. What separates it
# from the red arm parts is SATURATION: the pale sticker reads S 75-103, the red plastic
# S 136-180 — cap S at 130. (Wood/ball are H 10-25 and/or high-S: excluded.)
PINK_BANDS = (((0, 40, 80), (10, 130, 255)),       # red-side pink (warm light)
              ((150, 40, 80), (180, 130, 255)))    # magenta-side pink (cool light)
# The marker is a SMALL box that is MOSTLY pink. Gating on pink *fraction* (not raw
# count) + a size cap cleanly rejects clutter that GDINO mislabels "heart" (white table,
# printer, the ball): those are big and/or only incidentally pink.
MAX_HEART_PX = 70                  # a heart sticker is small top-down; table/printer aren't
MIN_PINK_PX = 20                   # absolute floor, guards tiny high-fraction flukes
MIN_PINK_FRAC = 0.10               # pink pixels / box area (GDINO boxes run loose)

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


# ── Detection ──────────────────────────────────────────────────────────────────────

class HeartDetector:
    """Zero-shot 'heart' detector (Grounding DINO). `hearts()` returns every heart box
    with its pink-pixel count; `marker_uv()` picks the pink one (the finger marker)."""

    def __init__(self, device):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.torch = torch
        self.device = device
        self.proc = AutoProcessor.from_pretrained(GDINO_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(device).eval()

    def hearts(self, color_bgr):
        """List of ((x1,y1,x2,y2), conf, pink_px, pink_frac) for every detected heart."""
        import cv2
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
        inp = self.proc(images=img, text="heart.", return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inp)
        res = self.proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=GDINO_THR, text_threshold=GDINO_THR,
            target_sizes=[img.size[::-1]])[0]
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        boxes = []
        for box, score in zip(res["boxes"].tolist(), res["scores"].tolist()):
            x1, y1, x2, y2 = (int(v) for v in box)
            sub = hsv[max(y1, 0):y2, max(x1, 0):x2]
            pink = int(sum(cv2.inRange(sub, lo, hi).sum() for lo, hi in PINK_BANDS) / 255) \
                if sub.size else 0
            frac = pink / max((x2 - x1) * (y2 - y1), 1)
            boxes.append(((x1, y1, x2, y2), float(score), pink, frac))
        return boxes

    def marker_uv(self, color_bgr):
        """Return ((u, v) of the pink heart centre or None, all heart boxes for preview).
        The marker is the small, mostly-pink box (max fraction among gated boxes)."""
        boxes = self.hearts(color_bgr)
        cand = [b for b in boxes
                if max(b[0][2] - b[0][0], b[0][3] - b[0][1]) <= MAX_HEART_PX
                and b[2] >= MIN_PINK_PX and b[3] >= MIN_PINK_FRAC]
        if not cand:
            return None, boxes
        (x1, y1, x2, y2), _, _, _ = max(cand, key=lambda b: b[3])
        return ((x1 + x2) // 2, (y1 + y2) // 2), boxes


def backproject(u, v, depth_m, K, win=4):
    """Pixel + aligned depth -> 3-D point in the camera frame (metres). Uses the median
    valid depth in a small window for robustness. None if no valid depth there."""
    z = depth_m[max(v - win, 0):v + win, max(u - win, 0):u + win]
    z = z[(z > WORKSPACE_Z[0]) & (z < WORKSPACE_Z[1])]
    if len(z) < 5:
        return None
    z = float(np.median(z))
    return np.array([(u - K["ppx"]) * z / K["fx"],
                     (v - K["ppy"]) * z / K["fy"], z])


# ── Forward kinematics (MuJoCo) ──────────────────────────────────────────────────────

def make_fk():
    """Return fk(ang_deg) -> (R, t): pose of the `gripper` body (the wrist_roll part,
    in the base/world frame. The finger heart is rigid w.r.t. this body as long as the
    gripper opening stays FIXED during the capture run (the offset absorbs the rest)."""
    import mujoco
    mm = mujoco.MjModel.from_xml_path(XML)
    md = mujoco.MjData(mm)
    adr = {j: mm.jnt_qposadr[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, j)]
           for j in MOTOR_NAMES}
    bid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, MARKER_BODY)

    def fk(ang):
        for j, a in adr.items():
            md.qpos[a] = math.radians(ang[j])
        mujoco.mj_forward(mm, md)
        return md.xmat[bid].reshape(3, 3).copy(), md.xpos[bid].copy()
    return fk


# ── Solver ───────────────────────────────────────────────────────────────────────────

def solve_handeye(P_cam, R_w, t_w):
    """Jointly fit R_cb, t_cb (camera->base) and o (marker offset in wrist frame).
    Returns dict with R, t, offset, per-pose residual norms (m), and RMS (m)."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    P_cam = np.asarray(P_cam, float)
    R_w = np.asarray(R_w, float)
    t_w = np.asarray(t_w, float)

    def resid(x):
        Rcb = Rotation.from_rotvec(x[:3]).as_matrix()
        tcb, o = x[3:6], x[6:9]
        pred = (Rcb @ P_cam.T).T + tcb                       # marker in base via camera
        meas = np.einsum("nij,j->ni", R_w, o) + t_w          # marker in base via FK
        return (pred - meas).ravel()

    best = None
    # top-down camera: seed R_cb near a 180° flip about X (cam +Z -> base -Z), 4 yaws
    for yaw in (0.0, np.pi / 2, np.pi, -np.pi / 2):
        seed = (Rotation.from_rotvec([np.pi, 0, 0]) * Rotation.from_rotvec([0, 0, yaw])).as_rotvec()
        x0 = np.concatenate([seed, [0, 0, 0.3], [0, 0, 0]])
        sol = least_squares(resid, x0, method="lm", max_nfev=4000)
        if best is None or sol.cost < best.cost:
            best = sol

    err = best.fun.reshape(-1, 3)
    norms = np.linalg.norm(err, axis=1)
    Rcb = Rotation.from_rotvec(best.x[:3]).as_matrix()
    return dict(R=Rcb, t=best.x[3:6], offset=best.x[6:9],
                residuals_m=norms, rms_m=float(np.sqrt(np.mean(norms ** 2))))


# ── Offline self-test (no hardware) ───────────────────────────────────────────────────

def selftest():
    from scipy.spatial.transform import Rotation
    print("FK model load + `gripper` body ...", end=" ")
    fk = make_fk()
    R0, t0 = fk({n: 0.0 for n in MOTOR_NAMES})
    print(f"OK  (gripper at zero pose t={np.round(t0, 3)})")

    rng = np.random.default_rng(0)
    Rcb_t = Rotation.from_rotvec([np.pi + 0.08, 0.05, -0.12]).as_matrix()
    tcb_t = np.array([0.10, -0.20, 0.52])
    o_t = np.array([0.02, -0.015, 0.03])

    P_cam, R_w, t_w = [], [], []
    for _ in range(15):
        # plausible wrist orientations: random small tilts about a downward-ish axis
        Rw = Rotation.from_rotvec(rng.uniform(-1, 1, 3) * 0.6).as_matrix()
        tw = rng.uniform([-0.2, -0.2, 0.05], [0.2, 0.2, 0.30])
        marker_base = Rw @ o_t + tw
        p_cam = Rcb_t.T @ (marker_base - tcb_t) + rng.normal(0, 0.001, 3)  # 1 mm noise
        P_cam.append(p_cam); R_w.append(Rw); t_w.append(tw)

    s = solve_handeye(P_cam, R_w, t_w)
    dR = np.degrees(np.arccos(np.clip((np.trace(Rcb_t.T @ s["R"]) - 1) / 2, -1, 1)))
    dt = np.linalg.norm(s["t"] - tcb_t) * 1000
    do = np.linalg.norm(s["offset"] - o_t) * 1000
    print(f"solver recovery: rotation err={dR:.2f}°, translation err={dt:.1f} mm, "
          f"offset err={do:.1f} mm, residual RMS={s['rms_m']*1000:.2f} mm")
    ok = dR < 1.0 and dt < 5 and do < 5
    print("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


# ── Realsense (own pipeline: aligned color+depth+intrinsics) ──────────────────────────

class Realsense:
    def __init__(self):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(SERIAL)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        prof = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        for _ in range(30):
            self.pipe.wait_for_frames()                       # warm up auto-exposure

    def grab(self):
        f = self.align.process(self.pipe.wait_for_frames())
        d, c = f.get_depth_frame(), f.get_color_frame()
        intr = c.get_profile().as_video_stream_profile().get_intrinsics()
        K = dict(fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy)
        return (np.asarray(c.get_data()),
                np.asarray(d.get_data(), np.float32) * self.scale, K)

    def stop(self):
        self.pipe.stop()


# ── stdin command thread ──────────────────────────────────────────────────────────────

def input_thread(q):
    for line in sys.stdin:
        q.put(line.strip().lower())


def preview_loop(detector, frame_slot, preview, stop):
    """Run the (slow, esp. on CPU) heart detector OFF the control loop so gamepad teleop
    stays responsive. Reads the latest frame, writes the latest (uv, boxes) for preview."""
    while not stop.is_set():
        fr = frame_slot[0]
        if fr is None:
            time.sleep(0.05)
            continue
        try:
            preview[0] = detector.marker_uv(fr)
        except Exception as e:
            print(f"  detector error: {e!r}")
            time.sleep(0.2)


# ── Main calibration loop ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Hand-eye calibration via the pink finger heart.")
    ap.add_argument("--selftest", action="store_true", help="Offline solver/FK check, no hardware.")
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--min-poses", type=int, default=10)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"
    import cv2
    import pygame
    import rerun as rr
    sys.path.insert(0, str(ROOT))
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from gamepad_utils import (get_joint_deltas, apply_deltas, detect_profile,
                               graceful_shutdown, DeltaSmoother)

    import torch
    fk = make_fk()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading heart detector ({GDINO_ID}) on {device} ...")
    detector = HeartDetector(device)
    detector.hearts(np.zeros((480, 640, 3), np.uint8))     # warm up CUDA before Rerun/Vulkan
    if device == "cpu":
        print("  ⚠ running on CPU — live preview will be slow (a few s/frame).")

    cam = Realsense()
    robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
    print("Connecting arm...")
    robot.connect()

    rr.init("handeye_calib", spawn=True)
    # explicit blueprint: one camera view (image + heart boxes + green marker dot).
    # Overrides the viewer's saved layout from older runs (e.g. the dead cam/mask panel).
    import rerun.blueprint as rrb
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial2DView(origin="cam", name="camera"),
                                    collapse_panels=True))
    pygame.init()
    joystick = None
    smoother = DeltaSmoother(alpha=0.45)
    obs = robot.get_observation()
    goal = {n: obs.get(f"{n}.pos", 0.0) for n in MOTOR_NAMES}

    cmd_q = queue.Queue()
    threading.Thread(target=input_thread, args=(cmd_q,), daemon=True).start()

    # detection runs in its own thread so teleop never waits on it (esp. on CPU)
    frame_slot = [None]                # latest color frame for the detector
    preview = [(None, [])]             # latest (uv, boxes) from the detector
    det_stop = threading.Event()
    threading.Thread(target=preview_loop,
                     args=(detector, frame_slot, preview, det_stop), daemon=True).start()

    samples = []                       # list of dict(p_cam, R_w, t_w, ang)
    print(f"\nDrive the arm so the PINK finger heart is well in view, then 'c' to capture. "
          f"Need >= {args.min_poses}. 'u' undo, 'q' finish & solve.\n"
          f"  ⚠ Keep the gripper opening FIXED (ideally closed) for the whole run.\n")

    try:
        while True:
            for e in pygame.event.get():
                if e.type == pygame.JOYDEVICEADDED:
                    joystick = pygame.joystick.Joystick(e.device_index)
                    joystick.init()
                    profile = detect_profile(joystick)
                    print(f"  Gamepad: {joystick.get_name()}")
                elif e.type == pygame.JOYDEVICEREMOVED:
                    joystick = None

            if joystick is not None:
                deltas = smoother(get_joint_deltas(joystick, profile))
                goal = apply_deltas(goal, deltas)
                robot.send_action({f"{n}.pos": goal[n] for n in MOTOR_NAMES})

            obs = robot.get_observation()
            color, depth, K = cam.grab()
            frame_slot[0] = color          # hand the frame to the detector thread

            # live preview + marker lock to Rerun (detection result comes from the thread)
            uv, boxes = preview[0]
            rr.log("cam/image", rr.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)))
            if boxes:
                rr.log("cam/hearts", rr.Boxes2D(
                    array=[[x1, y1, x2 - x1, y2 - y1] for (x1, y1, x2, y2), *_ in boxes],
                    array_format=rr.Box2DFormat.XYWH,
                    labels=[f"{c:.2f} pink={f:.2f}" for _, c, _, f in boxes]))
            else:
                rr.log("cam/hearts", rr.Clear(recursive=False))
            if uv is not None:
                rr.log("cam/marker", rr.Points2D([list(uv)], radii=8, colors=[(0, 255, 0)]))
            else:
                rr.log("cam/marker", rr.Clear(recursive=False))

            # commands
            handled = False
            while not cmd_q.empty():
                c = cmd_q.get_nowait()
                if c in ("q", "quit"):
                    handled = "q"
                elif c == "u":
                    if samples:
                        samples.pop()
                        print(f"  undo -> {len(samples)} poses")
                elif c in ("c", "", "cap"):
                    if uv is None:
                        n_h = len(boxes)
                        print(f"  no pink heart found ({n_h} heart(s) detected, none pink enough) — "
                              f"reposition so the pink finger heart faces the camera")
                        continue
                    p_cam = backproject(*uv, depth, K)
                    if p_cam is None:
                        print("  no valid depth at the marker — reposition")
                        continue
                    ang = {n: obs.get(f"{n}.pos", 0.0) for n in MOTOR_NAMES}
                    R_w, t_w = fk(ang)
                    samples.append(dict(p_cam=p_cam, R_w=R_w, t_w=t_w, ang=ang))
                    print(f"  captured #{len(samples)}: uv={uv}  "
                          f"p_cam={np.round(p_cam*1000).astype(int)} mm  "
                          f"wrist_base={np.round(t_w*1000).astype(int)} mm")
            if handled == "q":
                break

            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nUnexpected error: {e!r}\nLanding the arm safely...")
    finally:
        det_stop.set()
        cam.stop()
        graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception:
            pass
        pygame.quit()

    # ── solve ──────────────────────────────────────────────────────────────────────
    if len(samples) < args.min_poses:
        print(f"\nOnly {len(samples)} poses (< {args.min_poses}) — not solving. Re-run to collect more.")
        return
    s = solve_handeye([x["p_cam"] for x in samples],
                      [x["R_w"] for x in samples],
                      [x["t_w"] for x in samples])
    print(f"\nSolved T_cam->base from {len(samples)} poses:")
    print(f"  residual RMS = {s['rms_m']*1000:.1f} mm   "
          f"(per-pose {np.round(s['residuals_m']*1000, 1)} mm)")
    print(f"  marker offset in wrist frame = {np.round(s['offset']*1000, 1)} mm")
    if s["rms_m"] > 0.010:
        print("  ⚠ RMS > 10 mm — check for a mis-detected pose (high per-pose residual) and re-run.")

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "handeye.json"
    json.dump(dict(R=s["R"].tolist(), t=s["t"].tolist(), marker_offset=s["offset"].tolist(),
                   rms_m=s["rms_m"], n_poses=len(samples),
                   created=datetime.now().isoformat()), open(out, "w"), indent=2)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
