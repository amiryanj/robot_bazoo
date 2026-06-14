#!/usr/bin/env python
"""Localize the mini-basketball with the top-down Realsense and grab it.

Pipeline (critical-path step 3, first iteration):
  ball_yolo (2-D box + depth -> ball centre, cam frame)
  -> handeye.json (T_cam->base from vision/handeye_calib.py)
  -> MuJoCo numeric IK (same model the calibration validated end-to-end)
  -> slow scripted sequence: above ball -> descend -> close -> lift -> put back -> rest.

Usage:
    python pick_ball.py --selftest    # offline IK check, no hardware
    python pick_ball.py --dry-run     # detect + print base-frame ball position, no arm
    python pick_ball.py               # the real thing (asks once before moving)

The arm ALWAYS lands via graceful_shutdown, whatever happens.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

XML = str(ROOT / "SO-ARM100/Simulation/SO101/scene.xml")
HANDEYE = ROOT / "outputs/calib/handeye.json"
TAG_CALIB = ROOT / "outputs/calib/tag_calib.json"   # wrist_roll mapping offset (delta)
# THE real-degrees -> model-qpos correction (the "layer in between"): the real joint zeros
# drift from the CAD/model zeros -- eyeballed lerobot homing on every joint, plus the
# wrist_roll horn sitting ~-85deg off. Every consumer that feeds live/command angles into
# MuJoCo (FK/IK, the Twin, scene_debug) must add these, or sim != reality. This does NOT
# touch lerobot control -- only the sim/twin interpretation of the reported angles.
TCP_SITE = "gripperframe"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
MOTOR_NAMES = ARM_JOINTS + ["gripper"]
OFFSET_CALIB = ROOT / "outputs/calib/offset_calib.json"


def _load_joint_offsets():
    """Per-joint command->model offsets (deg). Source: vision/offset_calib.py (GTSAM, all
    5 arm joints incl. the desk-tag-resolved pan); falls back to tag_calib's wrist_roll-
    only delta if that file is absent."""
    try:
        oc = json.load(open(OFFSET_CALIB))["delta_deg"]
        return {j: float(oc[j]) for j in ARM_JOINTS}          # gripper omitted (unobservable)
    except Exception:
        try:
            return {"wrist_roll": float(json.load(open(TAG_CALIB))["delta_deg"])}
        except Exception:
            return {}


JOINT_OFFSETS = _load_joint_offsets()
ROLL_DELTA_DEG = JOINT_OFFSETS.get("wrist_roll", 0.0)         # back-compat (tag_handeye)

# workspace sanity bounds for the detected ball, base frame (metres)
BALL_X = (0.12, 0.42)
BALL_Y = (-0.30, 0.30)
BALL_Z = (-0.02, 0.12)

APPROACH_CLEAR = 0.05      # pre-grasp clearance back along the fingers (m)
LIFT_CLEAR = 0.08          # lift height above grasp (m)
GRIP_OPEN = 95.0           # gripper command while approaching (0=closed, 100=open)
GRIP_GRASP = 30.0          # partial-close floor: don't fully shut (no crush). Placeholder
                           # until the close target is sized to the object (radius -> mm).
MOVE_SECONDS = 2.5         # per segment, linear joint interpolation
RATE = 20                  # interpolation steps/s


# ── Kinematics: FK + damped-least-squares IK on the validated MuJoCo model ──────────

class Kin:
    def __init__(self):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        self.adr = {j: self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                    for j in MOTOR_NAMES}
        self.sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        self.lim = {j: np.degrees(self.m.jnt_range[
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS}
        # dof (velocity-space) indices of the arm joints, for jacobian columns
        self.dof = [self.m.jnt_dofadr[self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_JOINT, j)]
                    for j in ARM_JOINTS]
        # command->model roll mapping: the real wrist_roll horn sits ~-85 deg vs the
        # model (measured by vision/tag_sweep.py, verified gauge-free via the jaw
        # axis). Applied here only — IK in/out stays in command space. Without it
        # the planned grasp point is ~11 mm off (missed grasp, 2026-06-12).
        self.roll_delta = ROLL_DELTA_DEG
        # Collision model for the grasp-centre search: scene + a movable (mocap) ball geom.
        # We aim the JAW-OPENING CENTRE at the ball, NOT the TCP "gripperframe" site — the
        # site sits at the fixed finger, so aiming the TCP plants the ball on/through that
        # finger. gap_center_offset() finds where a ball of given radius sits centred
        # between the two fingers (equal clearance, no penetration) via mj_geomDistance.
        scene_dir = Path(XML).parent
        wrap = scene_dir / "_grasp_search.xml"
        wrap.write_text('<mujoco model="grasp_search"><include file="scene.xml"/><worldbody>'
                        '<body name="ballm" mocap="true" pos="0 0 0">'
                        '<geom name="ballg" type="sphere" size="0.02" contype="0" '
                        'conaffinity="0"/></body></worldbody></mujoco>')
        try:
            self.sm = mujoco.MjModel.from_xml_path(str(wrap))
        finally:
            wrap.unlink(missing_ok=True)
        self.sd = mujoco.MjData(self.sm)
        self.s_adr = {j: self.sm.jnt_qposadr[mujoco.mj_name2id(self.sm, mujoco.mjtObj.mjOBJ_JOINT, j)]
                      for j in MOTOR_NAMES}
        self.s_sid = mujoco.mj_name2id(self.sm, mujoco.mjtObj.mjOBJ_SITE, TCP_SITE)
        self.s_ballg = mujoco.mj_name2id(self.sm, mujoco.mjtObj.mjOBJ_GEOM, "ballg")
        gf = mujoco.mj_name2id(self.sm, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        gm = mujoco.mj_name2id(self.sm, mujoco.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
        self.s_fixed = [g for g in range(self.sm.ngeom) if self.sm.geom_bodyid[g] == gf]
        self.s_moving = [g for g in range(self.sm.ngeom) if self.sm.geom_bodyid[g] == gm]
        self._gap_cache = {}

    def fk(self, ang_deg):
        for j, a in self.adr.items():
            off = JOINT_OFFSETS.get(j, 0.0)
            self.d.qpos[a] = math.radians(ang_deg.get(j, 0.0) + off)
        self.mj.mj_forward(self.m, self.d)
        R = self.d.site_xmat[self.sid].reshape(3, 3).copy()
        return R, self.d.site_xpos[self.sid].copy()

    def _ik_pass(self, p_target, ang, iters, damping, w_rot, approach_dir):
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        cols = self.dof[:4]                              # pan, lift, elbow, wrist_flex
        for _ in range(iters):
            R, p = self.fk(ang)
            e_pos = p_target - p
            e_rot = np.cross(R[:, 0], approach_dir)      # pull finger-axis onto approach_dir
            if np.linalg.norm(e_pos) < 5e-4 and w_rot * np.linalg.norm(e_rot) < 0.01:
                break
            self.mj.mj_jacSite(self.m, self.d, jacp, jacr, self.sid)
            J = np.vstack([jacp[:, cols + [self.dof[4]]],     # position: all 5 joints
                           w_rot * jacr[:, cols + [self.dof[4]]]])
            e = np.concatenate([e_pos, w_rot * e_rot])
            dq = np.linalg.solve(J.T @ J + damping * np.eye(J.shape[1]), J.T @ e)
            for k, j in enumerate(ARM_JOINTS[:4]):
                ang[j] = float(np.clip(ang[j] + math.degrees(dq[k]),
                                       self.lim[j][0], self.lim[j][1]))
            # wrist_roll (dq[4]) intentionally not applied

    def ik(self, p_target, ang0, iters=400, damping=2e-3, w_rot=0.5, approach_dir=None):
        """Joint angles (deg) putting the TCP at p_target with the fingers pointing along
        approach_dir. Default = HORIZONTAL, radially OUTWARD from the base toward the
        target: a SIDE approach (wrist level, jaws grab the ball's equator). Orientation
        is kept (only lightly eased) so the wrist stays horizontal. wrist_roll is held at
        ang0's value. Returns (angles, pos_err_m, axis_err_deg)."""
        if approach_dir is None:
            approach_dir = np.array([p_target[0], p_target[1], 0.0])
            n = np.linalg.norm(approach_dir)
            approach_dir = approach_dir / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        ang = dict(ang0)
        self._ik_pass(p_target, ang, iters, damping, w_rot, approach_dir)
        for w in (0.3, 0.15):                            # ease, but keep the wrist level
            _, p = self.fk(ang)
            if np.linalg.norm(p_target - p) < 0.006:
                break
            self._ik_pass(p_target, ang, 150, damping, w, approach_dir)
        R, p = self.fk(ang)
        axis_err = math.degrees(math.acos(np.clip(float(R[:, 0] @ approach_dir), -1, 1)))
        return ang, float(np.linalg.norm(p_target - p)), axis_err

    def approach_axis(self, ang):
        """Unit vector the fingers point along (site x-axis) at this pose."""
        R, _ = self.fk(ang)
        return R[:, 0]

    def gap_center_offset(self, radius):
        """Where a ball of this radius sits CENTRED between the two fingers, expressed in
        the gripper frame (offset from the TCP site). Found by sweeping ball positions in
        the finger pocket (gripper at GRIP_OPEN) and keeping the one with equal, positive
        clearance to both fingers (centred, no penetration). Pose-independent (the fingers
        are rigid to the gripper), so computed at a canonical pose and cached per radius."""
        key = round(float(radius), 4)
        if key in self._gap_cache:
            return self._gap_cache[key]
        m, d, mj = self.sm, self.sd, self.mj
        m.geom_size[self.s_ballg] = [radius, 0, 0]
        for j, a in self.s_adr.items():                       # canonical pose + GRIP_OPEN
            d.qpos[a] = math.radians(JOINT_OFFSETS.get(j, 0.0))
        d.qpos[self.s_adr["gripper"]] = math.radians(GRIP_OPEN + JOINT_OFFSETS.get("gripper", 0.0))
        mj.mj_forward(m, d)
        R = d.site_xmat[self.s_sid].reshape(3, 3)
        tcp = d.site_xpos[self.s_sid]
        best = None
        for dx in np.linspace(-0.075, -0.01, 12):
            for dy in np.linspace(-0.015, 0.04, 10):
                for dz in np.linspace(-0.01, 0.07, 14):
                    off = np.array([dx, dy, dz])
                    d.mocap_pos[0] = tcp + R @ off
                    mj.mj_forward(m, d)
                    df = min(mj.mj_geomDistance(m, d, self.s_ballg, g, 0.5, None) for g in self.s_fixed)
                    dm = min(mj.mj_geomDistance(m, d, self.s_ballg, g, 0.5, None) for g in self.s_moving)
                    if df < 0.003 or dm < 0.003:              # must clear BOTH fingers
                        continue
                    score = abs(df - dm) + 0.5 * (df + dm)    # centred + snug in the pocket
                    if best is None or score < best[0]:
                        best = (score, off)
        off = best[1] if best else np.zeros(3)
        self._gap_cache[key] = off
        return off


# SIDE approach: fingers level (horizontal) when shoulder_lift + elbow_flex + wrist_flex = 0
# (model FK). DLS is local, so try several postures along that constraint and keep the best.
# wrist_roll = 0 -> jaw opens left/right around the ball's equator (clears the holder).
IK_SEEDS = [{"shoulder_lift": lift, "elbow_flex": elbow, "wrist_flex": -lift - elbow,
             "shoulder_pan": 0.0, "wrist_roll": 0.0, "gripper": 0.0}
            for lift, elbow in ((45, 45), (30, 30), (60, 20), (40, 60), (55, 5))]


def ik_best(kin, p_target):
    best = None
    for seed in IK_SEEDS:
        sol = kin.ik(np.array(p_target), seed)
        if best is None or (sol[1] + 0.01 * sol[2]) < (best[1] + 0.01 * best[2]):
            best = sol
    return best


def plan_waypoints(kin, p_base, radius=0.0245):
    """Three via points for the side grasp (NO collision checking yet — these waypoints
    are the poor-man's substitute):
      high   — beside the ball but raised LIFT_CLEAR, so the big move in from rest goes
               UP and over rather than sweeping the gripper low across the table;
      above  — beside the ball at grasp height, backed off APPROACH_CLEAR along the fingers;
      grasp  — the precise grasp pose (only this one needs precision).
    Returns (high, above, grasp, grasp_err_m, axis_err_deg, above_err_m)."""
    p_base = np.asarray(p_base, float)
    # rough aim, then re-aim so the JAW-OPENING CENTRE (not the TCP) lands on the ball:
    grasp0, _, _ = ik_best(kin, p_base)
    R0, _ = kin.fk(grasp0)
    g_off = kin.gap_center_offset(radius)                 # gap centre in gripper frame
    tcp_target = p_base - R0 @ g_off                      # so gap centre hits the ball
    grasp, e_g, tilt = ik_best(kin, tcp_target)
    u = kin.approach_axis(grasp)                          # horizontal, points wrist -> ball
    p_beside = tcp_target - APPROACH_CLEAR * u
    above, e_a, _ = kin.ik(p_beside, grasp)
    high, _, _ = kin.ik(p_beside + np.array([0, 0, LIFT_CLEAR]), above)
    return high, above, grasp, e_g, tilt, e_a


def selftest():
    kin = Kin()
    ok = True
    # SIDE approach needs the ball raised off the table (it sits on a holder, ~50mm),
    # so the level gripper can reach the equator without the body hitting the surface.
    for target in ([0.30, 0.05, 0.05], [0.33, -0.07, 0.055], [0.28, 0.12, 0.05],
                   [0.36, 0.0, 0.05]):
        high, above, grasp, e_g, tilt, e_a = plan_waypoints(kin, target)
        line = (f"  ball={np.round(target, 3)}  grasp_err={e_g * 1000:.1f}mm  "
                f"axis_err={tilt:.0f}deg  approach_short={e_a * 1000:.0f}mm  "
                f"q={[round(grasp[j], 1) for j in ARM_JOINTS[:4]]}")
        good = e_g < 0.008 and tilt < 30 and e_a < 0.05
        ok &= good
        print(("OK " if good else "FAIL") + line)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


# ── Perception: ball in the base frame ───────────────────────────────────────────────

YOLO_WEIGHTS = ROOT / "vision/models/scene_yolov8n.pt"


class BallDetector:
    """Fast student first, slow teacher as fallback.

    Student: yolov8n fine-tuned on our scene (vision/DETECTOR.md) — ~5 ms/frame.
    Fallback: Grounding DINO zero-shot "basketball." for frames the student has
    never seen (it's single-scene v1). The broadcast basketball.pt YOLO stays out:
    it scored ~0 on the mini ball against the white plate."""

    def __init__(self, device=None):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.student = None
        if YOLO_WEIGHTS.exists():
            from ultralytics import YOLO
            self.student = YOLO(str(YOLO_WEIGHTS))
        self._gdino = None                                # lazy: only if student fails

    def _gdino_detect(self, color_bgr, thr):
        import cv2
        from PIL import Image
        if self._gdino is None:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            mid = "IDEA-Research/grounding-dino-tiny"
            self._gdino = (AutoProcessor.from_pretrained(mid),
                           AutoModelForZeroShotObjectDetection.from_pretrained(mid)
                           .to(self.device).eval())
        proc, model = self._gdino
        img = Image.fromarray(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
        inp = proc(images=img, text="basketball.", return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = model(**inp)
        res = proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=thr, text_threshold=thr,
            target_sizes=[img.size[::-1]])[0]
        best = None
        for box, score in zip(res["boxes"].tolist(), res["scores"].tolist()):
            if best is None or score > best[1]:
                best = ([int(v) for v in box], float(score))
        return best if best else (None, 0.0)

    def detect(self, color_bgr, thr=0.3):
        """Highest-conf ball box -> ((x1,y1,x2,y2), conf) or (None, 0)."""
        if self.student is not None:
            res = self.student(color_bgr, conf=0.45, verbose=False)[0]
            best = None
            for c, p, b in zip(res.boxes.cls, res.boxes.conf, res.boxes.xyxy):
                if res.names[int(c)] == "ball" and (best is None or float(p) > best[1]):
                    best = ([int(v) for v in b], float(p))
            if best:
                return best
        return self._gdino_detect(color_bgr, thr)


def localize_base(detector=None):
    """Grab a frame, detect the ball, return (p_base, info dict). Always saves the
    frame + detection overlay to outputs/vision/pick_<ts>/ for debugging."""
    import cv2
    from datetime import datetime
    from ball import WORKSPACE_Z
    from ball_yolo import ball_from_box
    from cloud import crop_z, extract_planes
    from cloud import deproject as cloud_deproject
    from handeye_calib import Realsense

    he = json.load(open(HANDEYE))
    R_cb, t_cb = np.array(he["R"]), np.array(he["t"])

    detector = detector or BallDetector()
    cam = Realsense()
    try:
        color, depth, K = cam.grab()
    finally:
        cam.stop()

    box, score = detector.detect(color)
    ball = ball_from_box(box, score, depth, K) if box else None

    out = ROOT / "outputs/vision" / f"pick_{datetime.now():%Y-%m-%d_%H-%M-%S}"
    out.mkdir(parents=True, exist_ok=True)
    overlay = color.copy()
    if ball is not None:
        x1, y1, x2, y2 = ball["box"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(overlay, ball["uv"], 4, (0, 0, 255), -1)
    cv2.imwrite(str(out / "frame.png"), color)
    cv2.imwrite(str(out / "overlay.png"), overlay)
    print(f"  debug frames -> {out}")

    if ball is None:
        return None, None
    p_base = R_cb @ ball["center3d"] + t_cb
    # z comes from the sphere fit. The scene's SUPPORT planes (plate, desk, spool top —
    # extract_planes separates them) are reported as context: which one the ball rests
    # on, or that it's raised.
    pts = crop_z(cloud_deproject(depth, K), WORKSPACE_Z)
    base_pts = (R_cb @ pts.T).T + t_cb
    supports = [pl for pl in extract_planes(base_pts[::4], max_planes=3)
                if abs(pl["n"][2]) > 0.95]
    z_table, resting = None, None
    bottom = p_base[2] - ball["radius_m"]
    for pl in supports:
        z = float(pl["centroid"][2])
        if z_table is None or abs(bottom - z) < abs(bottom - z_table):
            z_table = z
    if z_table is not None:
        resting = abs(bottom - z_table) < 0.012
        print(f"  supports at {[round(float(pl['centroid'][2]) * 1000) for pl in supports]}mm; "
              f"ball bottom {bottom * 1000:.0f}mm -> "
              f"{'resting on the ' + str(round(z_table * 1000)) + 'mm plane' if resting else 'raised'}")
    ball["z_table"] = z_table
    return p_base, ball


def watch():
    """Live Rerun preview: frame + ball box + conf, until Ctrl-C. For aiming/setup."""
    import cv2
    import rerun as rr
    import rerun.blueprint as rrb
    from handeye_calib import Realsense

    detector = BallDetector()
    print(f"Detector on {detector.device}. Ctrl-C to stop.")
    cam = Realsense()
    rr.init("pick_ball", spawn=True)
    rr.send_blueprint(rrb.Blueprint(rrb.Spatial2DView(origin="cam", name="ball watch"),
                                    collapse_panels=True))
    try:
        while True:
            color, depth, K = cam.grab()
            box, score = detector.detect(color)
            rr.log("cam/image", rr.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB)))
            if box:
                x1, y1, x2, y2 = box
                rr.log("cam/ball", rr.Boxes2D(array=[[x1, y1, x2 - x1, y2 - y1]],
                                              array_format=rr.Box2DFormat.XYWH,
                                              labels=[f"ball {score:.2f}"]))
            else:
                rr.log("cam/ball", rr.Clear(recursive=False))
    except KeyboardInterrupt:
        pass
    finally:
        cam.stop()


# ── Digital twin (MuJoCo viewer: scene + detected ball + mirrored joints) ────────────

class Twin:
    def __init__(self, p_ball, radius, z_table=None):
        import mujoco
        import mujoco.viewer
        self.mj = mujoco
        scene_dir = Path(XML).parent
        # the scene's floor is at z=0 but the REAL table measures ~-29mm in base coords
        # (model origin is partway up the base plate) — draw the measured table too,
        # else the ball renders buried in the visual floor and looks mislocalized.
        table = ""
        if z_table is not None:
            table = (f'<geom name="table_twin" type="box" size="0.45 0.45 0.002" '
                     f'pos="0.25 0 {z_table - 0.002:.4f}" rgba="0.8 0.75 0.65 0.6" '
                     f'contype="0" conaffinity="0"/>')
        wrapper = scene_dir / "_pick_twin.xml"   # in scene dir so includes/assets resolve
        wrapper.write_text(f"""<mujoco model="pick_twin">
  <include file="scene.xml"/>
  <worldbody>
    <geom name="ball_twin" type="sphere" size="{radius:.4f}"
          pos="{p_ball[0]:.4f} {p_ball[1]:.4f} {p_ball[2]:.4f}"
          rgba="0.95 0.5 0.15 1" contype="0" conaffinity="0"/>
    {table}
  </worldbody>
</mujoco>""")
        try:
            self.m = mujoco.MjModel.from_xml_path(str(wrapper))
        finally:
            wrapper.unlink(missing_ok=True)
        self.d = mujoco.MjData(self.m)
        self.adr = {j: self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                    for j in MOTOR_NAMES}
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)

    def set(self, ang_deg):
        for j, a in self.adr.items():
            off = JOINT_OFFSETS.get(j, 0.0)
            self.d.qpos[a] = math.radians(ang_deg.get(j, 0.0) + off)
        self.mj.mj_forward(self.m, self.d)
        self.viewer.sync()

    def close(self):
        try:
            self.viewer.close()
        except Exception:
            pass


# ── Motion helpers ───────────────────────────────────────────────────────────────────

def read_angles(robot):
    obs = robot.get_observation()
    return {n: obs.get(f"{n}.pos", 0.0) for n in MOTOR_NAMES}


def move_to(robot, start, goal, seconds=MOVE_SECONDS, twin=None):
    """Slow linear joint-space interpolation start -> goal."""
    steps = max(int(seconds * RATE), 1)
    for i in range(1, steps + 1):
        a = i / steps
        cmd = {n: start[n] + a * (goal[n] - start[n]) for n in MOTOR_NAMES}
        robot.send_action({f"{n}.pos": cmd[n] for n in MOTOR_NAMES})
        if twin:
            twin.set(cmd)
        time.sleep(1.0 / RATE)
    return dict(goal)


def close_on_ball(robot, pose, twin=None):
    """Close the gripper in small steps until the jaws stall on the ball (contact) or reach
    the partial-close floor GRIP_GRASP — NOT fully shut, and no extra squeeze (gentle hold).
    Object-adaptive sizing is a TODO. Returns the final command pose."""
    cmd = dict(pose)
    trace = []
    for g in np.arange(pose["gripper"], GRIP_GRASP, -3.0):
        cmd["gripper"] = float(g)
        robot.send_action({f"{n}.pos": cmd[n] for n in MOTOR_NAMES})
        if twin:
            twin.set(cmd)
        time.sleep(0.15)
        meas = read_angles(robot)["gripper"]
        trace.append((round(float(g), 1), round(float(meas), 1)))
        if meas - g > 6.0:                                # jaws stalled on the ball
            cmd["gripper"] = float(meas)                  # hold at contact, no squeeze
            robot.send_action({f"{n}.pos": cmd[n] for n in MOTOR_NAMES})
            print(f"  contact: gripper held at {meas:.0f} (no squeeze)")
            return cmd
    cmd["gripper"] = GRIP_GRASP                           # no hard contact: hold partial
    robot.send_action({f"{n}.pos": cmd[n] for n in MOTOR_NAMES})
    if twin:
        twin.set(cmd)
    print(f"  no clear contact; holding partial close at {GRIP_GRASP:.0f}; trace {trace}")
    return cmd


# ── Main ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Localize and grab the mini-basketball.")
    ap.add_argument("--selftest", action="store_true", help="Offline IK check, no hardware.")
    ap.add_argument("--dry-run", action="store_true", help="Detect + print only, no arm.")
    ap.add_argument("--watch", action="store_true", help="Live Rerun detection preview, no arm.")
    ap.add_argument("--no-twin", action="store_true", help="Skip the MuJoCo twin window.")
    ap.add_argument("--record", action="store_true",
                    help="Save a camera frame at each grasp stage (forensics).")
    ap.add_argument("--port", default="/dev/ttyACM1")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if args.watch:
        watch()
        return

    print("Localizing ball...")
    p_base, ball = localize_base()
    if p_base is None:
        sys.exit("Ball not found in the frame — aborting (nothing moved).")
    if ball["fit_ok"]:
        fit_msg = f"OK ({ball['inliers']} inliers, {ball['fit_rms'] * 1000:.1f}mm rms)"
    else:
        fit_msg = "FAILED — fallback estimate"
    print(f"  conf={ball['conf']:.2f}  sphere fit: {fit_msg}")
    print(f"  ball centre, BASE frame = {np.round(p_base * 1000).astype(int)} mm")
    if not args.dry_run and not ball["fit_ok"]:
        sys.exit("Sphere fit failed — localization not trustworthy enough to grasp. "
                 "Re-run; if persistent, check depth coverage on the ball.")

    inside = (BALL_X[0] <= p_base[0] <= BALL_X[1] and BALL_Y[0] <= p_base[1] <= BALL_Y[1]
              and BALL_Z[0] <= p_base[2] <= BALL_Z[1])
    print(f"  workspace check: {'OK' if inside else 'OUT OF BOUNDS'} "
          f"(x{BALL_X} y{BALL_Y} z{BALL_Z})")
    if args.dry_run:
        return
    if not inside:
        sys.exit("Refusing to move to an out-of-bounds target.")

    # IK both waypoints before touching the arm.
    # SIDE approach: aim the TCP (fingertip plane) straight at the ball centre — the level
    # jaws then close around the equator. (The old top-down code aimed below centre and
    # applied corner_fkerr.json; both were tuned for the vertical grasp and don't apply.)
    kin = Kin()
    target = np.asarray(p_base, float)
    high, above, grasp, e_g, tilt, e_a = plan_waypoints(kin, target, ball["radius_m"])
    print(f"  IK: grasp_err={e_g * 1000:.1f}mm  axis_err={tilt:.0f}deg  "
          f"approach_short={e_a * 1000:.0f}mm")
    if e_g > 0.008 or tilt > 30 or e_a > 0.05:
        sys.exit("IK did not converge well — target out of reach. Aborting.")
    for w in (high, above, grasp):
        w["gripper"] = GRIP_OPEN

    twin = None
    if not args.no_twin:
        try:
            twin = Twin(np.asarray(p_base, float), ball["radius_m"], ball.get("z_table"))
            twin.set(grasp)                              # preview the planned grasp pose
            print("  Twin window: planned GRASP pose vs the detected ball — check it straddles.")
        except Exception as e:
            print(f"  (twin unavailable: {e!r})")

    if input("Move the arm? [y/N] ").strip().lower() != "y":
        if twin:
            twin.close()
        sys.exit("Aborted (nothing moved).")

    sys.path.insert(0, str(ROOT))
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from gamepad_utils import graceful_shutdown

    rec = None
    if args.record:
        import cv2
        from datetime import datetime
        from handeye_calib import Realsense
        rec_dir = ROOT / "outputs/vision" / f"grasp_{datetime.now():%Y-%m-%d_%H-%M-%S}"
        rec_dir.mkdir(parents=True, exist_ok=True)
        rec_cam = Realsense()

        def rec(stage):
            c, _, _ = rec_cam.grab()
            cv2.imwrite(str(rec_dir / f"{stage}.png"), c)
        print(f"  recording stages -> {rec_dir}")

    robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
    robot.connect()
    try:
        cur = read_angles(robot)
        print("1/7 raise to safe height"); cur = move_to(robot, cur, high, twin=twin)
        if rec: rec("1_high")
        print("2/7 beside ball (pre-grasp)"); cur = move_to(robot, cur, above, twin=twin)
        if rec: rec("2_beside")
        print("3/7 move in (side)"); cur = move_to(robot, cur, grasp, seconds=2.0, twin=twin)
        if rec:
            import cv2 as _cv
            time.sleep(0.4)
            _c, _, _K = rec_cam.grab()
            he3 = json.load(open(HANDEYE))
            _R, _t = np.array(he3["R"]), np.array(he3["t"])
            _, _ptcp = kin.fk(read_angles(robot))

            def _proj(p):
                pc = _R.T @ (np.asarray(p) - _t)
                return (int(_K["fx"] * pc[0] / pc[2] + _K["ppx"]),
                        int(_K["fy"] * pc[1] / pc[2] + _K["ppy"]))
            _cv.drawMarker(_c, _proj(_ptcp), (0, 255, 0), _cv.MARKER_CROSS, 24, 3)
            _cv.circle(_c, _proj(p_base), 8, (0, 165, 255), 3)
            _cv.imwrite(str(rec_dir / "3_movein_aim.png"), _c)

        # (depth-cloud finger servoing at the grasp pose was tried and removed: the
        # wrist occludes the scene below it — the top-down camera is blind exactly
        # there. Closed-loop correction lives in the ball-displacement pursuit instead.)
        print("4/7 close");       cur = close_on_ball(robot, cur, twin=twin)
        if rec: rec("4_closed")
        lift = dict(cur); lift.update({k: high[k] for k in ARM_JOINTS})   # raise to safe height
        print("5/7 lift");        cur = move_to(robot, cur, lift, seconds=2.0, twin=twin)
        time.sleep(1.0)
        if rec: rec("5_lifted")
        down = dict(cur); down.update({k: grasp[k] for k in ARM_JOINTS})
        print("6/7 put back");    cur = move_to(robot, cur, down, seconds=2.0, twin=twin)
        rel = dict(cur); rel["gripper"] = GRIP_OPEN
        print("7/7 release");     cur = move_to(robot, cur, rel, seconds=1.0, twin=twin)
        cur = move_to(robot, cur, dict(cur, **{k: high[k] for k in ARM_JOINTS}),
                      seconds=1.5, twin=twin)         # retract up to safe height
        if rec: rec("7_done")
        print("Done — grabbed, lifted, put back.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nUnexpected error: {e!r}\nLanding the arm safely...")
    finally:
        graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception:
            pass
        if rec:
            rec_cam.stop()
        if twin:
            twin.close()


if __name__ == "__main__":
    main()
