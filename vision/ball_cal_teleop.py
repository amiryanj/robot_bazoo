#!/usr/bin/env python
"""Teleop + button-capture for ball-vs-finger-tag calibration data.

You drive the arm with the joystick (grab the ball, move it around). A live Rerun camera
view overlays the detected ball + the 2 finger tags and shows READY when all three are
visible. Press the controller's Capture button (the "screenshot" button) to record a
sample: the 2 finger tags + detected ball (pixel + 3D base via handeye) + joint angles +
FK gripper. The ball is rigid to the gripper, so detected-ball vs the finger-tag midpoint
(and vs FK) maps the calibration error -- data to refine it later.

Controls: sticks/shoulders drive the joints, A/B the gripper (standard station mapping),
Capture button = record a sample, Home button (or Ctrl-C) = save + land.

    python vision/ball_cal_teleop.py
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")          # headless gamepad

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vision"))

HANDEYE = ROOT / "outputs/calib/handeye.json"


def _aruco():
    import cv2
    par = cv2.aruco.DetectorParameters()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), par)


class CamThread(threading.Thread):
    """Grab the Realsense, detect ball + 2 finger tags, draw an overlay to Rerun, and
    publish the latest detections for the capture handler. Decoupled from the 50Hz teleop."""

    def __init__(self, R_cb, t_cb, detector):
        super().__init__(daemon=True)
        self.R_cb, self.t_cb, self.detector = R_cb, t_cb, detector
        self.lock = threading.Lock()
        self.latest = None
        self.stop = False

    def run(self):
        import cv2
        import rerun as rr
        from handeye_calib import Realsense, backproject
        from ball_yolo import ball_from_box
        cam = Realsense(); det = _aruco()
        try:
            while not self.stop:
                color, depth, K = cam.grab()
                ov = color.copy()
                tags = {}
                c, ids, _ = det.detectMarkers(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
                if ids is not None:
                    for c4, tid in zip(c, ids.ravel()):
                        if int(tid) not in (1, 2):
                            continue
                        uv = c4[0].mean(0)
                        pc = backproject(int(round(uv[0])), int(round(uv[1])), depth, K)
                        if pc is not None:
                            tags[int(tid)] = dict(px=uv.tolist(),
                                                  base3d=(self.R_cb @ pc + self.t_cb).tolist())
                            cv2.polylines(ov, [c4[0].astype(int)], True, (0, 180, 255), 2)
                            cv2.putText(ov, f"id{int(tid)}", tuple(c4[0][0].astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2)
                ball = None
                box, score = self.detector.detect(color)
                if box:
                    b = ball_from_box(box, score, depth, K)
                    if b is not None:
                        ball = dict(px=[float(b["uv"][0]), float(b["uv"][1])],
                                    base3d=(self.R_cb @ b["center3d"] + self.t_cb).tolist(),
                                    radius=float(b["radius_m"]))
                        x1, y1, x2, y2 = b["box"]
                        cv2.rectangle(ov, (x1, y1), (x2, y2), (0, 255, 0), 2)
                ready = len(tags) == 2 and ball is not None
                cv2.putText(ov, f"tags {len(tags)}/2  ball {'y' if ball else 'n'}"
                            + ("   READY - press Capture" if ready else "   reposition"),
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 230, 0) if ready else (0, 0, 255), 2)
                rr.log("cam", rr.Image(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)))
                with self.lock:
                    self.latest = dict(tags=tags, ball=ball, color=color, depth=depth, K=K)
                time.sleep(0.06)            # cap ~8Hz: yield the GIL so teleop stays crisp
        except Exception as e:
            print(f"  cam thread stopped: {e!r}")
        finally:
            cam.stop()


def main():
    import cv2
    import pygame
    import rerun as rr
    import torch
    from gamepad_utils import (detect_profile, get_joint_deltas, ButtonDebouncer,
                               DeltaSmoother, graceful_shutdown, JOINT_LIMITS, MOTOR_NAMES,
                               button_index)
    from pick_ball import Kin, BallDetector, read_angles
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    he = json.load(open(HANDEYE)); R_cb, t_cb = np.array(he["R"]), np.array(he["t"])

    pygame.init(); pygame.joystick.init()
    for _ in range(50):
        pygame.event.pump()
        if pygame.joystick.get_count():
            break
        time.sleep(0.1)
    if not pygame.joystick.get_count():
        sys.exit("no joystick found — plug in the Pro Controller.")
    js = pygame.joystick.Joystick(0); js.init()
    profile = detect_profile(js)
    cap_btn = button_index(profile, "Cap")
    if cap_btn is None:
        cap_btn = 4
    quit_btn = button_index(profile, "Home")

    kin = Kin()
    detector = BallDetector("cuda" if torch.cuda.is_available() else "cpu")
    rr.init("ball_cal_teleop", spawn=True)
    cam_thread = CamThread(R_cb, t_cb, detector); cam_thread.start()

    robot = SOFollower(SOFollowerRobotConfig(port="/dev/ttyACM1", id="so101", cameras={}))
    robot.connect()
    goal = read_angles(robot)
    smoother = DeltaSmoother(alpha=0.15); deb = ButtonDebouncer()
    run_dir = ROOT / f"outputs/calib/ball_cal_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    records = []; cap_prev = 0; target = 1.0 / 50; last = time.perf_counter()

    print("\n  Teleop: sticks/shoulders = joints, A/B = gripper.")
    print(f"  Capture button (#{cap_btn}) = record a sample (needs tags 2/2 + ball).")
    print(f"  Home button (#{quit_btn}) or Ctrl-C = save + land.\n  Watch the Rerun 'cam' view for READY.\n")

    try:
        while True:
            t0 = time.perf_counter()
            dt = min(max(t0 - last, 0.005), 0.1)           # measured tick -> correct deg/s
            last = t0
            pygame.event.pump()
            deltas = smoother(get_joint_deltas(js, profile, dt, debounce=deb))
            for n in MOTOR_NAMES:
                lo, hi = JOINT_LIMITS[n]
                goal[n] = max(lo, min(hi, goal[n] + deltas[n]))
            robot.send_action({f"{n}.pos": goal[n] for n in MOTOR_NAMES})

            cap = js.get_button(cap_btn)
            if cap and not cap_prev:
                with cam_thread.lock:
                    snap = dict(cam_thread.latest) if cam_thread.latest else None
                if snap and len(snap["tags"]) == 2 and snap["ball"]:
                    ang = read_angles(robot)
                    _, t_w = kin.fk(ang)
                    mid = (np.array(snap["tags"][1]["base3d"]) +
                           np.array(snap["tags"][2]["base3d"])) / 2
                    err = (np.array(snap["ball"]["base3d"]) - mid) * 1000
                    n = len(records) + 1                       # raw frames for re-processing
                    cv2.imwrite(str(run_dir / f"color_{n:02d}.png"), snap["color"])
                    np.save(run_dir / f"depth_{n:02d}.npy", snap["depth"])
                    records.append(dict(angles=ang, tcp_fk=t_w.tolist(), tags=snap["tags"],
                                        ball=snap["ball"], ball_minus_tagmid_mm=err.tolist(),
                                        K=snap["K"], color=f"color_{n:02d}.png",
                                        depth=f"depth_{n:02d}.npy"))
                    print(f"  captured #{n}: ball-tagmid = {np.round(err).astype(int)} mm")
                else:
                    have = len(snap["tags"]) if snap else 0
                    print(f"  skip: need tags 2/2 + ball (have tags {have}, "
                          f"ball {bool(snap and snap['ball'])})")
            cap_prev = cap

            if quit_btn is not None and js.get_button(quit_btn):
                print("  Home pressed — saving + landing.")
                break
            time.sleep(max(target - (time.perf_counter() - t0), 0.0))
    except KeyboardInterrupt:
        print("\n  Ctrl-C — saving + landing.")
    finally:
        cam_thread.stop = True; time.sleep(0.3)
        graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception:
            pass

    out = run_dir / "samples.json"
    json.dump(dict(records=records, n=len(records), created=datetime.now().isoformat()),
              open(out, "w"), indent=2)
    print(f"\n  {len(records)} samples (+ raw color/depth) -> {run_dir}")
    if records:
        e = np.array([r["ball_minus_tagmid_mm"] for r in records])
        print(f"  ball-vs-tagmid: mean {np.round(e.mean(0)).astype(int)} mm, "
              f"std {np.round(e.std(0)).astype(int)} mm")


if __name__ == "__main__":
    main()
