#!/usr/bin/env python
"""FK-vs-optical sweep: is the gripper z gap CONSTANT or POSE-DEPENDENT?

Per pose we compare each finger tag's optical position (depth-at-pixel -> base via handeye)
against its FK position (encoder angles + JOINT_OFFSETS -> MuJoCo body). Across many poses:
  - small spread of (meas - FK)  -> CONSTANT offset  -> handeye t / base-z shift (one number)
  - spread that tracks reach/height -> POSE-DEPENDENT -> kinematic offsets -> needs a 3-D refit

Two capture modes:
  --teleop (default): drive with the joystick, servo-HELD poses (tight), press the
      Capture button to log a sample. Home button / Ctrl-C = save + land.
  --handmove: torque OFF, hand-move the limp arm, auto-captures when still ~1s.

    python vision/twin_sweep.py            # joystick (recommended)
    python vision/twin_sweep.py --handmove
Raw color/depth + angles are saved per sample for a later 3-D offset refit.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")          # headless gamepad

import argparse
import json
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

HANDEYE = ROOT / "outputs/calib/handeye.json"
TAG_CALIB = ROOT / "outputs/calib/tag_calib.json"
XML = str(ROOT / "SO-ARM100/Simulation/SO101/scene.xml")
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
               "wrist_flex", "wrist_roll", "gripper"]


def analyze(samples, run_dir, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(samples) < 3:
        sys.exit(f"\nonly {len(samples)} samples — capture more poses and rerun.")
    gaps, gripz, reach, rows = [], [], [], []
    for s in samples:
        for tid, (meas, fkp) in s["tags"].items():
            g = (np.array(meas) - np.array(fkp)) * 1000
            gaps.append(g)
            gripz.append(s["grip_fk"][2] * 1000)
            reach.append(math.hypot(s["grip_fk"][0], s["grip_fk"][1]) * 1000)
            rows.append((tid, gripz[-1], reach[-1], g))
    gaps = np.array(gaps); gripz = np.array(gripz); reach = np.array(reach)
    mean, std = gaps.mean(0), gaps.std(0)
    rep = [f"{len(samples)} poses, {len(gaps)} tag sightings\n",
           "per sighting (meas - FK), mm:",
           "  tid  grip_fk_z  reach |  gap_x  gap_y  gap_z"]
    for tid, gz, rc, g in rows:
        rep.append(f"   {tid}    {gz:+6.0f}   {rc:5.0f} | {g[0]:+6.0f} {g[1]:+6.0f} {g[2]:+6.0f}")
    rep += ["", f"gap mean = [{mean[0]:+.1f} {mean[1]:+.1f} {mean[2]:+.1f}] mm",
            f"gap std  = [{std[0]:.1f} {std[1]:.1f} {std[2]:.1f}] mm"]
    if len(gaps) >= 4:
        cz = np.corrcoef(gripz, gaps[:, 2])[0, 1]
        cr = np.corrcoef(reach, gaps[:, 0])[0, 1]
        rep.append(f"corr(gap_z, grip_fk_z) = {cz:+.2f}   corr(gap_x, reach) = {cr:+.2f}")
    verdict = ("CONSTANT offset -> handeye t / base-z shift (apply -mean to fix)"
               if np.all(std < 6) else
               "POSE-DEPENDENT -> kinematic offsets wrong -> needs a 3-D refit")
    rep.append(f"\nVERDICT: {verdict}")
    print("\n" + "\n".join(rep))
    (run_dir / "report.txt").write_text("\n".join(rep) + "\n")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, (c, lab) in enumerate([("tab:red", "gap_x"), ("tab:green", "gap_y"),
                                  ("tab:blue", "gap_z")]):
        ax[0].scatter(range(len(gaps)), gaps[:, k], label=lab, c=c)
        ax[0].axhline(mean[k], color=c, ls=":", alpha=0.6)
    ax[0].set_xlabel("sighting"); ax[0].set_ylabel("gap (mm)")
    ax[0].set_title("gap per sighting (dotted = mean)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].scatter(gripz, gaps[:, 2], c="tab:blue", label="gap_z vs height")
    ax[1].scatter(reach, gaps[:, 0], c="tab:red", marker="x", label="gap_x vs reach")
    ax[1].set_xlabel("FK gripper z / reach (mm)"); ax[1].set_ylabel("gap (mm)")
    ax[1].set_title("pose dependence?"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(run_dir / "sweep.png", dpi=110); plt.close(fig)

    json.dump([dict(ang=s["ang"], grip_fk=list(s["grip_fk"]),
                    tags={t: dict(meas=list(mv), fk=list(fv))
                          for t, (mv, fv) in s["tags"].items()},
                    color=s.get("color"), depth=s.get("depth")) for s in samples],
              open(run_dir / "samples.json", "w"), indent=2)
    print(f"\nwrote: {run_dir}/  (report.txt, sweep.png, samples.json + raw frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--n", type=int, default=10, help="poses to capture (teleop: soft cap)")
    ap.add_argument("--maxsec", type=float, default=600.0)
    ap.add_argument("--handmove", action="store_true", help="limp/auto-capture instead of joystick")
    args = ap.parse_args()

    import cv2
    import mujoco
    from handeye_calib import Realsense, backproject
    from pick_ball import JOINT_OFFSETS

    he = json.load(open(HANDEYE))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])
    tc = json.load(open(TAG_CALIB))
    tag_sides = {int(k): float(v["side_m"]) for k, v in tc["tags"].items()}
    tag_body = {int(k): v["body"] for k, v in tc["tags"].items()}
    tag_off = {int(k): np.array(v["t"]) for k, v in tc["tags"].items()}

    def to_base(p):
        return (R_cb @ np.asarray(p)) + t_cb

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    adr = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
           for j in MOTOR_NAMES}
    bid = {tid: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
           for tid, b in tag_body.items()}
    grip_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper")

    def fk(ang):
        for j, a in adr.items():
            d.qpos[a] = math.radians(ang[j] + JOINT_OFFSETS.get(j, 0.0))
        mujoco.mj_forward(m, d)
        tags = {tid: (d.xmat[b].reshape(3, 3) @ tag_off[tid] + d.xpos[b]).copy()
                for tid, b in bid.items()}
        return tags, d.xpos[grip_bid].copy()

    def make_det():
        par = cv2.aruco.DetectorParameters()
        par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        return cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), par)

    def detect_tags(color, depth, K, det):
        out = {}
        c, ids, _ = det.detectMarkers(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
        if ids is not None:
            for c4, tid in zip(c, ids.ravel()):
                tid = int(tid)
                if tid in tag_sides:
                    uv = c4[0].mean(0)
                    pc = backproject(int(round(uv[0])), int(round(uv[1])), depth, K)
                    if pc is not None:
                        out[tid] = to_base(pc)
        return out

    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from pick_ball import read_angles
    robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
    robot.connect()
    run_dir = ROOT / f"outputs/calib/twin_sweep_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    samples = []

    def record(color, depth, ang, seen):
        n = len(samples) + 1
        cv2.imwrite(str(run_dir / f"color_{n:02d}.png"), color)
        np.save(run_dir / f"depth_{n:02d}.npy", depth)
        fk_tags, grip_fk = fk(ang)
        samples.append(dict(ang=ang, grip_fk=grip_fk,
                            tags={t: (seen[t], fk_tags[t]) for t in seen},
                            color=f"color_{n:02d}.png", depth=f"depth_{n:02d}.npy"))
        print(f"\n  captured {n}: " + "  ".join(
            f"t{t} gap=[{(seen[t]-fk_tags[t])[0]*1000:+.0f},"
            f"{(seen[t]-fk_tags[t])[1]*1000:+.0f},{(seen[t]-fk_tags[t])[2]*1000:+.0f}]mm"
            for t in sorted(seen)))

    cam = Realsense(color_res=(1280, 720))
    det = make_det()
    try:
        if args.handmove:
            robot.bus.disable_torque()
            print("Arm torque DISABLED — hand-move it; hold a pose still ~1s to capture.\n")
            hist = deque(maxlen=14); t0 = time.time()
            while len(samples) < args.n and time.time() - t0 < args.maxsec:
                color, depth, K = cam.grab()
                ang = {n: float(v) for n, v in
                       ((nm, robot.get_observation().get(f"{nm}.pos", 0.0))
                        for nm in MOTOR_NAMES)}
                av = np.array([ang[n] for n in MOTOR_NAMES]); hist.append((time.time(), av))
                still = (len(hist) >= 8 and hist[-1][0] - hist[0][0] > 0.7
                         and max(np.ptp([h[1][i] for h in hist]) for i in range(6)) < 1.5)
                seen = detect_tags(color, depth, K, det)
                _, grip_fk = fk(ang)
                new = all(np.linalg.norm(grip_fk - s["grip_fk"]) > 0.03 for s in samples)
                print(f"  [{len(samples)}/{args.n}] {'STILL' if still else 'moving'} "
                      f"tags{sorted(seen)} z={grip_fk[2]*1000:+.0f}"
                      + ("  CAPTURE" if (still and seen and new) else "   "), end="\r")
                if still and seen and new:
                    record(color, depth, ang, seen); hist.clear(); time.sleep(0.4)
                time.sleep(0.03)
        else:
            samples_teleop(robot, cam, det, detect_tags, record, read_angles, args, samples)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cam.stop()
        if not args.handmove:
            from gamepad_utils import graceful_shutdown
            graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception:
            pass

    analyze(samples, run_dir, args)


def samples_teleop(robot, cam, det, detect_tags, record, read_angles, args, samples):
    """Joystick teleop with a background tag-detector; Capture button logs a held pose."""
    import pygame
    from gamepad_utils import (detect_profile, get_joint_deltas, ButtonDebouncer,
                               DeltaSmoother, JOINT_LIMITS, button_index)

    pygame.init(); pygame.joystick.init()
    for _ in range(50):
        pygame.event.pump()
        if pygame.joystick.get_count():
            break
        time.sleep(0.1)
    if not pygame.joystick.get_count():
        sys.exit("no joystick — plug in the Pro Controller (or use --handmove).")
    js = pygame.joystick.Joystick(0); js.init()
    profile = detect_profile(js)
    cap_btn = button_index(profile, "Cap")
    if cap_btn is None:
        cap_btn = 4
    quit_btn = button_index(profile, "Home")

    # background camera/detector thread so 50Hz teleop stays crisp
    latest = {"v": None}; lock = threading.Lock(); stop = {"s": False}

    def cam_loop():
        while not stop["s"]:
            color, depth, K = cam.grab()
            seen = detect_tags(color, depth, K, det)
            with lock:
                latest["v"] = (color, depth, seen)
            time.sleep(0.05)
    th = threading.Thread(target=cam_loop, daemon=True); th.start()

    goal = read_angles(robot)
    smoother = DeltaSmoother(alpha=0.15); deb = ButtonDebouncer()
    target = 1.0 / 50; last = time.perf_counter(); cap_prev = 0; t0 = time.time()
    print(f"\n  Teleop: sticks/shoulders = joints, A/B = gripper.")
    print(f"  Capture (#{cap_btn}) = log a pose (needs >=1 finger tag visible).")
    print(f"  Home (#{quit_btn}) / Ctrl-C = save + land. Vary HEIGHT and REACH.\n")
    while len(samples) < args.n and time.time() - t0 < args.maxsec:
        tnow = time.perf_counter()
        dt = min(max(tnow - last, 0.005), 0.1); last = tnow
        pygame.event.pump()
        deltas = smoother(get_joint_deltas(js, profile, dt, debounce=deb))
        for n in JOINT_LIMITS:
            lo, hi = JOINT_LIMITS[n]
            goal[n] = max(lo, min(hi, goal[n] + deltas[n]))
        robot.send_action({f"{n}.pos": goal[n] for n in MOTOR_NAMES})

        cap = js.get_button(cap_btn)
        if cap and not cap_prev:
            with lock:
                snap = latest["v"]
            if snap and snap[2]:
                color, depth, seen = snap
                record(color, depth, read_angles(robot), seen)
            else:
                print("  skip: no finger tag visible — reorient the gripper.")
        cap_prev = cap
        if quit_btn is not None and js.get_button(quit_btn):
            print("  Home — save + land."); break
        time.sleep(max(target - (time.perf_counter() - tnow), 0.0))
    stop["s"] = True; time.sleep(0.2)


if __name__ == "__main__":
    main()
