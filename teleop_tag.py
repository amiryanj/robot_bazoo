#!/usr/bin/env python
"""Teleoperate the SO-101 with the tag-marked joystick (teleop step 3).

The webcam tracks the tagged controller (vision/tag_body.py), and its motion drives the
arm's TCP. Mapping is CLUTCHED and RELATIVE: hold L to engage, and from that instant the
tag's displacement is added to the TCP pose the arm had at engage. Release to freeze and
re-anchor wherever your hand ended up -- like lifting a mouse. That deliberately needs no
camera->base calibration: only the axis convention below, plus a gain.

  L (hold)  engage / clutch. Press = "I'm ready, anchor here". Release = arm freezes.
  A         ease the arm to the ready pose (do this first)
  twist     rotating the controller rolls the wrist (--roll-gain 0 / negative to flip)
  R-stick   up / down = open / close the gripper (proportional)
  Home      stop and land gracefully

POSITION + WRIST ROLL. The SO-101 is 5-DoF and cannot track a full 6-D hand pose, but
roll is free: the IK never touches wrist_roll (`_ik_pass` skips dq[4]), so twisting the
controller about the camera's viewing axis maps straight onto the wrist at zero cost to
the position solution. Hand PITCH and YAW are still only measured, not commanded -- those
would have to be bought out of the same 5 joints holding the position, so they belong in
ik(approach_dir=...) once position teleop is boring. --roll-gain 0 turns roll back off.

What stops a perception glitch from snapping the arm:
  - --gain scales hand displacement to arm displacement (1.0 = 1:1; measured 0.99)
  - the commanded TCP is speed-limited (--max-speed), per MEASURED tick time
  - poses that jump more than JUMP_MAX in one frame are DROPPED as outliers, and the
    pose is smoothed with an EMA before anything is commanded
  - an IK step that would move a joint more than MAX_JOINT_JUMP is re-solved with more
    damping (DAMP_LADDER) rather than refused -- a refusal deadlocks
  - the LEASH: the target only advances while the arm is actually keeping up. This, not
    a workspace box, is what handles the edge of the arm's reach (--cage to add the box)
  - the hand position is Kalman-fused (IMU predicts, tags correct); when its 1-sigma
    passes SIGMA_MAX_MM the clutch drops rather than following a guess

Not yet done, and worth adding once this runs: a proper OOD guard on the hand motion
itself (velocity/acceleration bounds, tag-count changes, reprojection spikes) rather than
the single per-frame jump test used here.

Every run logs a CSV episode to outputs/teleop/<ts>/ (--no-log to skip). It carries the
things that make the arm stop following you -- edge_px (tag about to leave the FOV),
ik_damp (how hard the IK had to work, i.e. how close to the reach limit), leash_held (the
arm falling behind) and clamped_mm (the cage, if enabled) -- because from the operator's
seat they all feel identical: "it went stiff". The status line names whichever is active.

    python teleop_tag.py --dry-run     # NO robot: verify the axis mapping in Rerun first
    python teleop_tag.py --sim         # then the MuJoCo twin: watch the whole arm move
    python teleop_tag.py               # only then the real arm
    python teleop_tag.py --view top    # + the top-down Realsense in Rerun to operate from
"""
import argparse
import csv
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")          # pygame: input only, no window

CAM_SOURCE = "9"
RATE_HZ = 30.0                     # nominal only -- every per-tick budget below uses
                                   # the MEASURED dt instead. The loop actually ran at a
                                   # median 13.3 Hz on the real arm (2026-08-20 log), so
                                   # dividing rates by an assumed 30 delivered 2.3x LESS
                                   # than asked: --max-speed 0.2 moved at 0.089 m/s, and
                                   # the gripper and wrist crawled at the same discount
DT_MAX = 0.25                      # s: cap the measured dt, so one long hitch cannot
                                   # cash in as a single huge step
CMD_RATE_HZ = 50.0                 # servo command rate, decoupled from the camera.
                                   # station.py has run teleop at 50 Hz on this bus for
                                   # months, so the hardware is known to take it
INTERP_S = 0.06                    # s: ramp onto each new goal over this long. Vision
                                   # delivers a goal only every ~77 ms, and at
                                   # --max-speed 0.2 that is a 15 mm JUMP each time --
                                   # the arm gets a staircase and steps through it.
                                   # Ramping costs ~30 ms of average lag and is the
                                   # cheapest smoothness available; 0 disables
GAIN = 1.0                         # arm metres per hand metre
JUMP_MAX = 0.06                    # m: a bigger frame-to-frame move is a detection glitch
TAU = 0.12                         # s: position smoothing TIME CONSTANT. Not a per-tick
                                   # alpha -- the loop's dt swung 28..152 ms in the
                                   # 2026-08-20 log, so a fixed alpha made the actual
                                   # filter cutoff wander 5.5x (3.8 Hz down to 0.7 Hz)
                                   # tick to tick, which by itself feels like uneven
                                   # following. 0.12 s reproduces the old median feel.
                                   # Same bug class as the RATE_HZ one above: a rate
                                   # expressed per TICK is wrong on a jittery loop
MAX_TCP_SPEED = 0.30               # m/s ceiling on the COMMANDED tcp. This is the ONE
                                   # number that decides how laggy teleop feels: the
                                   # target itself is rate-limited, so at the old 0.06
                                   # a brisk 20 cm hand move took 3 s to play out and
                                   # kept crawling after the hand stopped. Perception
                                   # is NOT the lag; only the newest frame is ever used
DIAG_HZ = 10.0                     # Rerun scalar rate for DIAGNOSTICS. ~30 series at
                                   # full loop rate is what makes the viewer expensive
                                   # to redraw; 10 Hz is plenty to read a plot by eye
IMG_HZ = 15.0                      # Rerun image rate. It is for eyeballing, not
                                   # control: encoding + the viewer's GPU work at
                                   # full rate competes with the MuJoCo window.
                                   # It is also ~95% of everything Rerun retains --
                                   # measured 2026-08-21, RSS grew 0.43 MB/s with images
                                   # vs 0.02 MB/s with scalars only and 0.00 MB/s with
                                   # logging off (so nothing in this file or MuJoCo
                                   # leaks). Rerun keeps the whole recording so you can
                                   # scrub back, so the image stream sets how long a
                                   # session stays cheap. --no-rr-images drops it.
LEASH = 0.040                      # m: how far the target may get ahead of the
                                   # ACHIEVED tcp before it stops advancing
MAX_JOINT_JUMP = 12.0              # deg: the most an IK step may move any one joint
DAMP_LADDER = (2e-3, 3e-2, 1e-1, 3e-1, 1.0)
HOPELESS = 0.030                   # m: a position error this big is UNREACHABILITY, not
                                   # a discontinuity. Damping shortens the step; it cannot
                                   # pull an out-of-range point into range, so walking the
                                   # rest of the ladder costs ~330 iterations per rung and
                                   # returns the same answer (409 ms measured 2026-08-21).
TWIN_HZ = 10.0                     # the MuJoCo viewer redraws at screen rate anyway, and
                                   # viewer.sync() BLOCKS on the render thread -- syncing
                                   # every tick couples the control loop to the compositor
                                   # (and to Rerun's window, which competes for the GPU).
                                   # Levenberg-Marquardt damping to retry with when a
                                   # step is too big. Near full stretch the arm is
                                   # redundant AND ill-conditioned, so the IK swings
                                   # joints ~19 deg/tick while the TCP moves 8 mm; more
                                   # damping buys a gentler joint step for almost no
                                   # position error. Replayed against the 99 refusals in
                                   # the 2026-08-20 log: the ladder rescues 100% of them,
                                   # median 4.9 mm / max 6.0 mm position error
MAX_REPROJ = 4.0                   # px: reject sloppy body poses. Now one consistent
                                   # unit (per-corner Euclidean RMS) for 1 and 2 tags --
                                   # tag_body.body_pose used to report the single-tag
                                   # case 1/sqrt(2) smaller, so 3.0 here really meant
                                   # 4.24 px on one tag and 3.0 on two. 4.0 keeps the
                                   # single-tag behaviour the operator already tuned
                                   # against while making the two-tag case no stricter
ROLL_RATE = 120.0                  # deg/s ceiling on the commanded wrist_roll
ROLL_RANGE = (-95.0, 95.0)         # deg, absolute
GRIP_RATE = 60.0                   # deg/s while ZL/ZR held
GRIP_RANGE = (2.0, 90.0)
BUTTONS = ("L", "A", "Home", "B")
GRIP_AXIS = 3                      # right stick Y: up = open, down = close.
                                   # NOT ZL/ZR - this pad's ZR is worn and
                                   # fires spuriously (see ButtonDebouncer)
MAX_MISS = 5                       # consecutive jump-rejects before we believe it
LOST_MAX = 8                       # frames without a tag before the clutch drops
EDGE_WARN_PX = 60                  # tag corner this close to the frame border = about
                                   # to leave the FOV. The operator cannot see the
                                   # camera's framing while watching the ARM, so the
                                   # tag leaving the view is a silent failure; this
                                   # is the early warning, logged and spoken aloud
BLIND_MAX_S = 0.4                  # s: drop the clutch after this long with no tag.
                                   # Time, not covariance: sigma is only meaningful if
                                   # the loop rate is healthy, and gating on it meant a
                                   # SINGLE missed frame at 3 Hz (dt 0.35 s) disengaged
                                   # instantly -- "lost confidence (125 mm)" over and
                                   # over on 2026-08-21, which reads as the arm simply
                                   # refusing to move. A clock is something the operator
                                   # can predict; a covariance is not.
SIGMA_MAX_MM = 6.0                 # mm: (diagnostic only now) fused-estimate 1-sigma
                                   # 1-sigma. The filter grows its own uncertainty while
                                   # no tag is visible, at a rate set by the calibration
                                   # (see HandKF), so this single number replaces the old
                                   # separate time and distance caps -- and it relaxes by
                                   # itself when `pad_imu.py check` improves.
                                   # 6 mm is set from MEASURED error, not from sigma's own
                                   # claim: over 12 trials sigma reads ~1.4x optimistic at
                                   # every gap length (5.0 mm true vs 3.5 mm claimed at
                                   # 150 ms; 7.9 vs 5.5 at 500 ms). 6 mm of claimed sigma
                                   # is therefore ~8 mm of real error and ~500 ms of
                                   # blindness -- past the 268 ms p75 of observed dropouts.
TWIST_TAU = 0.4                    # s: how fast the VISION angle pulls the gyro-
                                   # integrated wrist twist back. Longer than TAU on
                                   # purpose: with a gyro the fast response comes from
                                   # integration and vision only has to kill the drift.
                                   # With no gyro this falls back to TAU (pure vision).
G_TAU = 2.0                        # s: gravity-direction estimate. Hand acceleration
                                   # averages to zero over seconds; gravity does not
LOG_ROOT = ROOT / "outputs/teleop"
WRIST_INDEX = 15                   # the wrist webcam (CLAUDE.md); 640x480 only
# Workspace box in the BASE frame (x fwd, y left, z up) + a reach band. This is a coarse
# SAFETY CAGE, not the thing the operator should feel -- the arm's own envelope is a
# curved shell, and the LEASH already handles hitting it gracefully (the target stops
# advancing when the achieved TCP falls behind). Sized from a measured IK sweep
# (2026-08-20): x reaches 0.15-0.40 at z=0.195 and 0.25-0.46 down at z=0.05, y +-0.33 low
# / +-0.26 at working height, z tops out ~0.28. The old x=(0.08, 0.36) left just 36 mm of
# forward headroom from the ready pose (TCP [324, 31, 195] mm, radius 379) against 244 mm
# back, and its 0.08 inner / 0.30 top were never reachable at all -- so fore/aft felt
# walled-in while back/down felt free. z min stays a table guard: the plate is at -0.029.
BOX = dict(x=(0.15, 0.46), y=(-0.28, 0.28), z=(0.02, 0.28))
REACH = (0.15, 0.46)
Z_FLOOR = 0.02                     # m: the ONE limit kept when the cage is off. The
                                   # plate is at -0.029, and a target under it just
                                   # presses the gripper into the desk -- which the
                                   # servos will hold against until they overheat.
                                   # Everything else is now the arm's own envelope,
                                   # handled by the leash instead of a hard clip.
READY_POSE = {"shoulder_pan": 0.0, "shoulder_lift": -18.0, "elbow_flex": 60.0,
              "wrist_flex": -40.0, "wrist_roll": 0.0, "gripper": 40.0}
# base_axis = AXES[i] applied to the CAMERA delta. Camera frame is RDF (x right, y down,
# z forward), and it FACES the operator, so image-right is the operator's LEFT and
# "push the hand away from the camera" is +z. base_y = +cam_x looks backwards written down
# -- but the camera MIRRORS you: moving your hand to your own right sends the tag to
# image-LEFT (-cam_x), which must become arm-right (-base_y). Verified in sim 2026-08-20;
# base_z = -cam_y (hand up -> arm up) was confirmed earlier. Flip with --axes.
AXES_DEFAULT = "-z,x,-y"


def parse_axes(spec):
    """'-z,-x,-y' -> 3x3 matrix M with base_delta = M @ cam_delta."""
    M = np.zeros((3, 3))
    for row, part in enumerate(spec.split(",")):
        part = part.strip()
        sign = -1.0 if part[0] == "-" else 1.0
        M[row, "xyz".index(part[-1])] = sign
    return M


def alpha_for(dt, tau):
    """Per-tick EMA weight giving a dt-independent `tau`-second time constant."""
    return 1.0 - math.exp(-dt / max(tau, 1e-6)) if tau > 0 else 1.0


class Filter:
    """Outlier drop + EMA. Returns the smoothed position, or None while untrusted.

    The jump test runs on the RAW pose: an EMA alone would happily swallow a 20 cm
    detection flip over a few frames and drive the arm there smoothly.

    It MUST be able to recover. A rejection test with no recovery path latches: one
    relocation (tag occluded, hand moved, re-acquired elsewhere) leaves `last_raw`
    stale, and every later frame is measured against a position the hand will never
    return to -- 1021 drops in a row, observed 2026-08-20. So after `max_miss`
    consecutive rejections we believe the tag and re-seed, raising `relocated` so the
    caller can drop the clutch instead of quietly driving to the new anchor."""

    def __init__(self, tau=TAU, jump=JUMP_MAX, max_miss=MAX_MISS):
        self.tau, self.jump, self.max_miss = tau, jump, max_miss
        self.p = None
        self.last_raw = None
        self.dropped = 0
        self.miss = 0
        self.relocated = False

    def __call__(self, p, dt):
        if self.last_raw is not None and np.linalg.norm(p - self.last_raw) > self.jump:
            self.dropped += 1
            self.miss += 1
            if self.miss < self.max_miss:
                return self.p                   # transient glitch: hold the last good one
            self.reset()                        # sustained: it really is somewhere else
            self.relocated = True
        self.miss = 0
        self.last_raw = p
        a = alpha_for(dt, self.tau)
        self.p = p if self.p is None else (1 - a) * self.p + a * p
        return self.p

    def reset(self):
        self.p = self.last_raw = None
        self.miss = 0


COLUMNS = [
    "time_s", "hz", "engaged", "paused", "tags", "reproj_px", "edge_px",
    "raw_x", "raw_y", "raw_z", "hand_x", "hand_y", "hand_z", "dropped", "lost",
    "tgt_x", "tgt_y", "tgt_z", "tcp_x", "tcp_y", "tcp_z", "track_mm",
    "clamped_mm", "leash_held", "ik_ok", "ik_jump_deg", "ik_perr_mm", "ik_axis_deg",
    "at_limit", "ik_damp", "twist_deg", "bridge_mm",
    # per-stage tick timing (ms). Which part of the loop got slow is not something to
    # infer from the outside -- 2026-08-21 cost a round of guessing about Rerun.
    "ms_grab", "ms_detect", "ms_ik", "ms_twin", "ms_rerun", "ms_rest",
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper",
]


def edge_margin(dets, shape):
    """Smallest distance from any detected corner to the frame border, in px.

    This is the 'you are about to lose the tag' number. Losing it is not a graceful
    degradation -- the clutch drops and the arm freezes mid-motion -- and the operator
    is looking at the arm, not at the camera view, so they get no warning at all."""
    if not dets:
        return float("nan")
    h, w = shape[:2]
    return min(min(x, y, w - 1 - x, h - 1 - y)
               for _, c in dets for x, y in c)


def clamp_target(p, cage=False):
    """Table floor always; the box + reach cage only with --cage.

    The cage is off by default because it was fighting the operator rather than
    protecting anything: a box and a sphere are the wrong SHAPE for this arm's envelope
    (x reaches 0.46 low down but only ~0.40 at working height), so a cage loose enough
    not to wall you in at one height admits unreachable targets at another. The leash
    already handles the true boundary gracefully -- the target stops advancing when the
    arm falls behind -- and unlike a clip it degrades instead of snapping."""
    q = np.array(p, float)
    q[2] = max(q[2], Z_FLOOR)
    if not cage:
        return q
    q = np.array([np.clip(q[0], *BOX["x"]), np.clip(q[1], *BOX["y"]),
                  np.clip(q[2], *BOX["z"])])
    r = np.linalg.norm(q)
    if r > 1e-6:
        q *= np.clip(r, *REACH) / r
    return q


def build_blueprint(args, has_imu):
    """One panel per question, instead of every scalar in one pile.

    Ordered by what you actually look at while flying the arm: is it following me, why
    did it stop, and (with the IMU) how late is the camera."""
    import rerun.blueprint as rrb
    follow = rrb.TimeSeriesView(
        origin="/", name="1 · Following  (target vs actual, mm)",
        contents=["+ tcp/**"])
    why = rrb.TimeSeriesView(
        origin="/", name="2 · Why it stopped  (track/edge/limits)",
        contents=["+ teleop/track_mm", "+ teleop/edge_px", "+ teleop/leash_held",
                  "+ teleop/ik_ok", "+ teleop/at_limit", "+ teleop/clamped_mm",
                  "+ teleop/dropped", "+ teleop/lost", "+ teleop/engaged"])
    views = [follow, why]
    if has_imu:
        # THE delay panel: two sensors, one rotation. The horizontal gap between the
        # curves IS the pipeline delay.
        views.append(rrb.TimeSeriesView(
            origin="/", name="3 · DELAY  (gyro leads, vision lags)",
            contents=["+ delay/omega_gyro_dps", "+ delay/omega_vision_dps"]))
        views.append(rrb.TimeSeriesView(
            origin="/", name="4 · Wrist roll  (fused vs vision)",
            contents=["+ delay/twist_vision_deg", "+ delay/twist_fused_deg"]))
        views.append(rrb.TimeSeriesView(
            origin="/", name="5 · Fusion confidence  (tag hidden)",
            contents=["+ teleop/bridge_mm", "+ teleop/blind_s", "+ teleop/tags"]))
    views.append(rrb.TimeSeriesView(origin="joints", name="6 · Joints (deg)"))
    cams = ["+ world/cam/**"]
    if args.view != "none":
        cams.append(f"+ view/{args.view}")
    views.append(rrb.Spatial2DView(origin="/", name="Camera", contents=cams))
    return rrb.Blueprint(rrb.Grid(*views), collapse_panels=True)


def at_limit(kin, ang, tol=1.0):
    """Arm joints pinned against their model limits, i.e. the IK has run out of road.

    _ik_pass CLIPS every joint into `kin.lim`, so a saturated joint silently stops
    contributing and the TCP stalls short of the target while the solver still reports
    convergence. That is the 'IK feels stuck' case: not a solver failure, a joint that
    physically cannot go further."""
    return [j for j, (lo, hi) in kin.lim.items()
            if j in ang and (ang[j] <= lo + tol or ang[j] >= hi - tol)]


class HandKF:
    """Kalman filter on hand position. State [p, v, b] in the camera frame, 9-D.

    LINEAR, deliberately. The nonlinearity in inertial fusion is always rotation --
    rotations are not a vector space, so an orientation state forces an extended /
    error-state filter. We keep orientation OUT: it is estimated separately (gyro +
    vision complementary filter in InertialAid), and the accelerometer, once rotated to
    the camera frame and gravity-subtracted, enters as a known INPUT. A double integrator
    with a known input is linear -- no Jacobians, nothing to linearise.

    `b` is an accelerometer BIAS state, and it is the part that makes this work. The
    extrinsic calibration is good to ~2 deg, which leaks sin(2 deg) * 9.81 = 0.34 m/s^2
    of gravity into the measured acceleration. That leak is a near-constant OFFSET, not
    white noise, and the two must not be confused: white noise averages away, an offset
    integrates into a position error that grows as t^2. Modelling the leak as process
    noise instead of as a state gave 16 mm of tracking error while the filter cheerfully
    reported 2.9 mm of uncertainty (measured 2026-08-21) -- overconfident and wrong,
    which is the dangerous combination. With `b` in the state the vision updates observe
    the offset and cancel it, and Q only carries what is genuinely random: hand jerk the
    constant-acceleration model cannot follow, plus slow drift of the bias itself.

    A dropout needs no special case: it is a tick with a predict and no update. The
    covariance grows on its own and says when the estimate is no longer usable, which is
    a better stopping rule than the fixed time and distance caps it replaces -- and two
    of the nastier bugs in this file lived in those caps."""

    def __init__(self, jerk=15.0, bias_walk=0.3, meas_sigma=0.004):
        self.jerk = jerk                  # m/s^3, unmodelled hand dynamics. 15 is what a
                                          # hand actually produces weaving at ~1 Hz; the
                                          # filter is insensitive to it anyway (6 -> 40
                                          # moved the 150 ms gap error 5.0 -> 5.7 mm)
        self.bias_walk = bias_walk        # m/s^3, how fast the leak may drift
        self.meas_var = meas_sigma ** 2
        self.p = self.v = self.b = None
        self.P = np.eye(9)

    def start(self, p):
        self.p = np.asarray(p, float).copy()
        self.v = np.zeros(3)
        self.b = np.zeros(3)
        # the bias is unknown at first, so say so: 0.5 m/s^2 covers a ~3 deg calibration
        self.P = np.diag([1e-6] * 3 + [1e-2] * 3 + [0.25] * 3)

    def predict(self, dt, a_cam):
        if self.p is None:
            return
        a = np.zeros(3) if a_cam is None else np.asarray(a_cam, float) - self.b
        self.p = self.p + self.v * dt + 0.5 * a * dt * dt
        self.v = self.v + a * dt
        I = np.eye(3)
        F = np.eye(9)
        F[:3, 3:6] = I * dt
        F[:3, 6:] = -I * 0.5 * dt * dt        # bias pushes position, with a minus sign
        F[3:6, 6:] = -I * dt
        # White-noise-JERK discretisation. Writing sigma_a = jerk*dt and then reusing
        # the white-acceleration form gives Q_pp ~ dt^6, which explodes when the loop
        # stutters: one 350 ms tick produced a 46 METRE covariance (2026-08-21). The
        # correct terms are dt^5/20, dt^4/8, dt^3/3.
        j2 = self.jerk ** 2
        Q = np.zeros((9, 9))
        Q[:3, :3] = I * j2 * dt ** 5 / 20.0
        Q[:3, 3:6] = Q[3:6, :3] = I * j2 * dt ** 4 / 8.0
        Q[3:6, 3:6] = I * j2 * dt ** 3 / 3.0
        Q[6:, 6:] = I * (self.bias_walk * dt) ** 2
        self.P = F @ self.P @ F.T + Q

    def update(self, z, var=None):
        """Fold in a vision position fix."""
        if self.p is None:
            self.start(z)
            return
        r = np.asarray(z, float) - self.p
        S = self.P[:3, :3] + np.eye(3) * (self.meas_var if var is None else var)
        K = self.P[:, :3] @ np.linalg.inv(S)
        dx = K @ r
        self.p, self.v, self.b = self.p + dx[:3], self.v + dx[3:6], self.b + dx[6:]
        H = np.zeros((3, 9))
        H[:, :3] = np.eye(3)
        self.P = (np.eye(9) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return float(np.linalg.norm(r))

    @property
    def sigma_p(self):
        """1-sigma position uncertainty (m) — the filter's own confidence."""
        return float(np.sqrt(max(np.trace(self.P[:3, :3]) / 3.0, 0.0)))


class InertialAid:
    """Fuses the controller's own gyro/accel with the tag pose (see pad_imu.py).

    Vision stays in charge: it is the only drift-free source, and it is better than the
    IMU at exactly what this teleop does most -- slow, small, deliberate motion. The IMU
    contributes the two things vision is bad at.

      1. ROLL RATE. The gyro is a single integration (no double integral, no gravity), so
         wrist twist can run at the IMU's ~77 Hz instead of the camera's 13 Hz, corrected
         toward the vision angle whenever a tag is visible. This is a complementary
         filter, not a replacement.
      2. POSITION, through a Kalman filter (HandKF). The IMU drives the predict step
         every tick; each tag pose is a measurement update. A dropout is then not a
         special case at all -- just a tick with no update -- and the filter's own
         covariance says when the estimate has decayed too far to use.

    Gravity is never assumed: it is MEASURED as the low-passed accelerometer vector
    rotated into the (fixed) camera frame, which works because hand acceleration averages
    to zero over G_TAU seconds and gravity does not."""

    def __init__(self, imu, X, calib_deg=2.2):
        self.imu, self.X = imu, np.asarray(X, float)
        self.g_cam = None          # gravity in the camera frame (camera is fixed)
        self.R = None              # body rotation, propagated by gyro while blind
        self.kf = HandKF()
        self.blind = 0.0           # seconds since the last vision fix
        self.w_cam = np.zeros(3)   # latest angular velocity, camera frame (rad/s)
        self.speed = 0.0           # |omega|, deg/s
        self.a_cam = np.zeros(3)   # latest gravity-removed acceleration, camera frame

    def _rodrigues(self, w, dt):
        import cv2
        return cv2.Rodrigues(w * dt)[0]

    def update(self, dt, hand_R, hand_p, _unused=None):
        """Advance one tick. `hand_R`/`hand_p` are None when no tag was seen.

        Returns (fused position or None, twist rate deg/s). There is no longer a
        'bridging' special case: a dropout is just a tick with a predict and no update."""
        _, w_imu, a_imu = self.imu.latest()
        w_body = self.X @ w_imu
        a_body = self.X @ a_imu

        if hand_R is not None:                       # vision fix: re-anchor rotation
            self.R = hand_R
            g = hand_R @ a_body
            a_g = alpha_for(dt, G_TAU)
            self.g_cam = g if self.g_cam is None else (1 - a_g) * self.g_cam + a_g * g
        elif self.R is not None:                     # blind: propagate with the gyro
            self.R = self.R @ self._rodrigues(w_body, dt)

        self.w_cam = self.R @ w_body if self.R is not None else np.zeros(3)
        self.speed = float(np.degrees(np.linalg.norm(w_body)))
        twist_rate = math.degrees(self.w_cam[2])     # about the camera viewing axis

        if self.R is not None and self.g_cam is not None:
            self.a_cam = self.R @ a_body - self.g_cam
        else:
            self.a_cam = np.zeros(3)

        # PREDICT every tick, UPDATE only when vision has something to say. That single
        # asymmetry is the whole of "IMU for fast detail, tags for slow truth".
        self.kf.predict(dt, self.a_cam)
        if hand_p is not None and hand_R is not None:
            self.kf.update(hand_p)
            self.blind = 0.0
        else:
            self.blind += dt
        return (self.kf.p.copy() if self.kf.p is not None else None), twist_rate

    @property
    def sigma_mm(self):
        return self.kf.sigma_p * 1e3


class Sender(threading.Thread):
    """Feeds the servos at a steady rate, ramping onto each new goal from the vision loop.

    Command smoothness is decoupled from camera rate here. The vision loop is both SLOW
    (13 Hz measured) and JITTERY (dt 28..152 ms), and sending one big joint step per
    camera frame makes the arm walk down a staircase. This thread runs on its own clock
    and interpolates, so a ragged perception pipeline cannot put steps into the servos.

    It owns ALL sending. move_to() and graceful_shutdown() drive the bus directly, so
    they must run inside pause()/resume() or the two will fight over the port."""

    def __init__(self, robot, rate=CMD_RATE_HZ, span=INTERP_S):
        super().__init__(daemon=True)
        self.robot, self.dt, self.span = robot, 1.0 / rate, span
        self._lock = threading.Lock()
        # NOT self._stop: threading.Thread has a private _stop() that join() calls, and
        # shadowing it with an Event makes join() raise 'Event object is not callable'.
        self._halt = threading.Event()
        self._paused = True                      # nothing to send until the first set()
        self._start = self._goal = None
        self._t0 = 0.0
        self.sent = 0

    def set(self, ang):
        with self._lock:
            self._start = dict(self._goal) if self._goal is not None else dict(ang)
            self._goal = dict(ang)
            self._t0 = time.perf_counter()
            self._paused = False

    def resync(self, ang):
        """Adopt `ang` as the current pose WITHOUT ramping (after move_to moved the arm)."""
        with self._lock:
            self._start = self._goal = dict(ang)
            self._t0 = time.perf_counter()

    def pause(self):
        with self._lock:
            self._paused = True

    def run(self):
        while not self._halt.is_set():
            t = time.perf_counter()
            with self._lock:
                send = None
                if not self._paused and self._goal is not None:
                    f = 1.0 if self.span <= 0 else min((t - self._t0) / self.span, 1.0)
                    send = {j: self._start[j] + f * (self._goal[j] - self._start[j])
                            for j in self._goal}
            if send is not None:
                try:
                    self.robot.send_action({f"{j}.pos": v for j, v in send.items()})
                    self.sent += 1
                except Exception as e:                # a dead bus must not kill teleop
                    print(f"\n  send failed: {e}")
            time.sleep(max(0.0, self.dt - (time.perf_counter() - t)))

    def stop(self):
        self.pause()
        self._halt.set()
        self.join(timeout=1.0)


class ViewCam:
    """A SECOND camera, streamed to Rerun purely so the operator can watch the workspace.

    Runs on its own thread and only ever hands back the newest frame: an operator view
    must never pace the control loop. Fault-tolerant on purpose -- the Realsense and the
    wrist webcam share a USB hub and stall together (CLAUDE.md), so failing to open one
    degrades to 'no extra view', never to 'no teleop'."""

    def __init__(self, spec, fov=70.0, size=(640, 480)):
        from tag_pose import Webcam, RS
        self.src = RS(size=size) if spec in ("top", "realsense") else Webcam(
            WRIST_INDEX if spec == "wrist" else spec, size=size, fov=fov)
        self._frame = None
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                f, _ = self.src.grab()
            except Exception:
                break
            if f is not None:
                self._frame = f

    def latest(self):
        return self._frame

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)
        try:
            self.src.stop()
        except Exception:
            pass


# ── fusion selftest (no camera, no hand, no arm) ───────────────────────────────────

class _FakeIMU:
    """Stands in for PadIMU: same `latest()` contract, values we control."""

    def __init__(self):
        self.w = np.zeros(3)
        self.a = np.zeros(3)

    def latest(self):
        return 0.0, self.w.copy(), self.a.copy()


def selftest(seed=0, verbose=True, calib_deg=1.4):
    """Drive InertialAid with a KNOWN hand motion and check what comes out.

    The fusion was written and tuned when only 0.4% of frames carried 2+ tags and the
    gyro read 12% high, so it has never been exercised on good input. This runs the real
    filter -- not a copy of it -- against a trajectory whose truth we know, with the
    error sources that actually exist: corner noise on the tag pose, tag dropouts, a
    tilted extrinsic leaking gravity into the accelerometer, and gyro bias.

    Checks three things, because they fail differently:
      1. tracking error while vision is present (should be at the noise floor),
      2. tracking error through a BLIND gap (this is what the IMU is for),
      3. whether the filter's own sigma is HONEST -- an overconfident filter is worse
         than a noisy one, since the caller trusts sigma to decide when to stop."""
    import cv2
    rng = np.random.default_rng(seed)

    HZ, T = 30.0, 40.0
    dt = 1.0 / HZ
    g_cam = np.array([0.0, 9.80665, 0.0])          # camera y-down: rest reading is +y
    X = cv2.Rodrigues(np.array([0.3, -0.8, 1.9]))[0]        # some IMU->body rotation
    # 1.4 deg: the alignment we actually measure now (pad_imu align, 2026-08-31, four
    # runs agreeing to 1.34 deg; live camera-vs-gyro gap 1.37 deg with 2+ tags).
    tilt = cv2.Rodrigues(rng.normal(0, np.radians(calib_deg), 3))[0]
    acc_bias = rng.normal(0, 0.05, 3)
    gyro_bias = rng.normal(0, np.radians(0.3), 3)

    def truth(t):
        """Hand position and orientation at time t: a smooth ~1 Hz weave."""
        p = np.array([0.06 * math.sin(2 * math.pi * 0.45 * t),
                      0.04 * math.sin(2 * math.pi * 0.31 * t + 1.0),
                      0.30 + 0.05 * math.sin(2 * math.pi * 0.23 * t + 2.0)])
        a = np.array([-0.06 * (2 * math.pi * 0.45) ** 2 * math.sin(2 * math.pi * 0.45 * t),
                      -0.04 * (2 * math.pi * 0.31) ** 2 * math.sin(2 * math.pi * 0.31 * t + 1.0),
                      -0.05 * (2 * math.pi * 0.23) ** 2 * math.sin(2 * math.pi * 0.23 * t + 2.0)])
        w = np.array([0.6 * math.sin(2 * math.pi * 0.17 * t),
                      0.5 * math.sin(2 * math.pi * 0.13 * t + 0.7),
                      0.4 * math.sin(2 * math.pi * 0.21 * t + 1.9)])
        return p, a, w

    # blind gaps: the box gives 2+ tags ~75% of the time, so gaps are short but real
    blind = np.zeros(int(T * HZ), bool)
    k = 0
    while k < len(blind):
        k += int(rng.uniform(1.0, 4.0) * HZ)
        n = int(rng.uniform(0.15, 0.60) * HZ)
        blind[k:k + n] = True
        k += n

    imu = _FakeIMU()
    aid = InertialAid(imu, X)
    R = cv2.Rodrigues(np.array([math.pi, 0.1, 0.0]))[0]
    rows = []
    for i in range(len(blind)):
        t = i * dt
        p_t, a_t, w_t = truth(t)
        R = R @ cv2.Rodrigues(w_t * dt)[0]

        # what the IMU would report: specific force in the body frame, then to IMU axes
        a_body = R.T @ (a_t + g_cam)
        imu.a = X.T @ (tilt @ a_body) + acc_bias + rng.normal(0, 0.02, 3)
        imu.w = X.T @ (R.T @ (R @ w_t)) + gyro_bias + rng.normal(0, np.radians(0.1), 3)

        if blind[i]:
            hand_R = hand_p = None
        else:
            hand_R = R @ cv2.Rodrigues(rng.normal(0, np.radians(1.4), 3))[0]
            hand_p = p_t + rng.normal(0, 0.002, 3)
        fused, _ = aid.update(dt, hand_R, hand_p)
        if fused is not None and t > 3.0:          # let the filter settle
            rows.append((t, float(blind[i]), float(np.linalg.norm(fused - p_t)),
                         aid.kf.sigma_p, aid.blind))

    rows = np.array(rows)
    err_seen = float(np.median(rows[rows[:, 1] == 0][:, 2])) * 1e3
    # honesty: how often does the true error exceed the filter's own 3-sigma?
    over = float(np.mean(rows[:, 2] > 3 * rows[:, 3]))
    # Error through a gap is dominated by the gravity that leaks in through the
    # extrinsic error: a tilt of d degrees leaks sin(d)*9.81 m/s^2, which integrates as
    # 0.5*a*t^2. So the useful number is not one figure but error vs GAP LENGTH -- it
    # says how long the IMU may be trusted before the arm should stop following.
    buckets = [(0.0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 1.0)]
    per_gap = []
    for lo, hi in buckets:
        m = (rows[:, 4] > lo) & (rows[:, 4] <= hi)
        if m.sum() > 5:
            per_gap.append((lo, hi, float(np.percentile(rows[m, 2], 90)) * 1e3,
                            float(np.median(rows[m, 3])) * 1e3, int(m.sum())))
    if verbose:
        print(f"  {len(rows)} ticks, {100*np.mean(rows[:,1]):.0f}% blind, "
              f"extrinsic error {calib_deg:.1f} deg")
        print(f"  with vision:  median error {err_seen:5.2f} mm")
        print(f"  blind for     p90 error   filter sigma   n")
        for lo, hi, e, sg, n in per_gap:
            print(f"   {lo:4.2f}-{hi:4.2f}s  {e:8.1f} mm  {sg:8.1f} mm  {n:5d}")
        print(f"  filter honesty: error > 3 sigma on {100*over:.1f}% of ticks (want < 1%)")
    # What to assert. The gap error is PHYSICS, not a bug: a tilt of d degrees leaks
    # sin(d)*9.81 m/s^2 into the acceleration and that integrates as 0.5*a*t^2, so at
    # 1.4 deg the error must quadruple when the gap doubles, and it does (8 -> 27 -> 77
    # mm). Pinning a number on the long buckets would only be pinning that formula. What
    # MUST hold is (a) vision-present accuracy, and (b) that the filter's own sigma stays
    # close to the true error -- the caller stops following when sigma grows, so an
    # optimistic sigma is the failure that actually hurts.
    assert err_seen < 8.0, f"tracking error with vision is {err_seen:.1f} mm"
    for lo, hi, e, sg, _ in per_gap:
        assert sg > 0.4 * e, (f"filter is optimistic in the {lo:.2f}-{hi:.2f}s bucket: "
                              f"says {sg:.0f} mm, really {e:.0f} mm")
    assert over < 0.01, f"filter is overconfident: {100*over:.1f}% beyond 3 sigma"
    short = [e for lo, hi, e, _, _ in per_gap if hi <= 0.15]
    assert not short or max(short) < 15.0, f"short-gap error is {max(short):.1f} mm"
    return err_seen, per_gap, over


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=CAM_SOURCE)
    ap.add_argument("--fov", type=float, default=70.0)
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--gain", type=float, default=GAIN)
    ap.add_argument("--max-speed", type=float, default=MAX_TCP_SPEED,
                    help="m/s ceiling on the commanded TCP (lower = safer, laggier)")
    ap.add_argument("--roll-gain", type=float, default=1.0,
                    help="hand twist -> wrist_roll (0 disables, negative flips)")
    ap.add_argument("--axes", default=AXES_DEFAULT, help="base<-cam axis map, e.g. '-z,-x,-y'")
    ap.add_argument("--sim", action="store_true",
                    help="drive the MuJoCo twin (sim_backend.SimRobot) instead of the arm")
    ap.add_argument("--no-twin", action="store_true",
                    help="--sim without the MuJoCo viewer window (much faster loop)")
    ap.add_argument("--cage", action="store_true",
                    help="re-enable the workspace box + reach clamp (off by default; "
                         "the table floor is always on)")
    ap.add_argument("--view", default="none", choices=["none", "top", "wrist"],
                    help="second camera into Rerun to operate from (top = Realsense)")
    ap.add_argument("--no-bridge", action="store_true",
                    help="use the IMU for wrist roll only — FREEZE on tag loss instead "
                         "of dead-reckoning through it (A/B against the default)")
    ap.add_argument("--no-imu", action="store_true",
                    help="ignore the controller's built-in IMU even if calibrated")
    ap.add_argument("--no-rr-images", action="store_true",
                    help="no camera images in Rerun (plots only) — ~95%% less data kept")
    ap.add_argument("--rr-scale", type=float, default=0.5,
                    help="size of the Rerun picture (1.0 = full; detection uses full)")
    ap.add_argument("--no-log", action="store_true", help="skip the episode CSV")
    ap.add_argument("--dry-run", action="store_true", help="no robot; verify the mapping")
    ap.add_argument("--model", default=str(ROOT / "outputs/calib/box_body.json"),
                    help="tag body model (default: the 3-face box)")
    ap.add_argument("--selftest", action="store_true",
                    help="check the fusion against a known trajectory; no hardware")
    args = ap.parse_args()

    if args.selftest:
        print("fusion selftest (synthetic hand, no camera / IMU / arm):")
        for seed in range(3):
            print(f"  --- seed {seed}")
            selftest(seed)
        print("\nselftest: PASS")
        return 0

    import cv2
    import pygame
    import rerun as rr
    from tag_pose import make_detector, detect, open_source, rpy_deg
    from tag_body import load_model, body_pose, Km
    from gamepad_utils import (button_index, graceful_shutdown, filtered_axis_value,
                               ButtonDebouncer)
    from station import JoystickManager
    from pick_ball import Kin, MOTOR_NAMES, read_angles, move_to

    M = parse_axes(args.axes)
    model = load_model(args.model)
    keep = set(model["tags"])
    det = make_detector("DICT_4X4_50")
    src = open_source(args.source, args.fov)
    kin = Kin()

    rr.init("teleop_tag")
    # NOT spawn=True: the viewer defaults to a 75% memory budget, so it keeps every
    # sample and the timeseries panels get slower to redraw as the session grows. That
    # is what collapsed the loop from 14.5 Hz to 2.8 Hz after ~80 s on 2026-08-21 --
    # the SDK side costs a flat 1.7 ms/tick (measured), it is the VIEWER competing for
    # the Intel iGPU. A small budget makes it drop old data and stay cheap.
    rr.spawn(memory_limit="256MB")
    rr.log("world", rr.ViewCoordinates.RDF, static=True)
    # Explicit colours: the pairs below are meant to be COMPARED, so they must not
    # come out in near-identical auto-assigned shades. Within each pair, blue = the
    # fast/commanded one, orange = the slow/measured one.
    BLUE, ORANGE = [70, 150, 255], [255, 150, 40]
    for path, col, nm, w in (
            ("delay/omega_gyro_dps",    BLUE,   "gyro (fast)",   2.0),
            ("delay/omega_vision_dps",  ORANGE, "vision (late)", 2.0),
            ("delay/twist_fused_deg",   BLUE,   "fused",         2.0),
            ("delay/twist_vision_deg",  ORANGE, "vision only",   2.0),
            ("tcp/target_x", [255, 120, 120], "target x", 2.0),
            ("tcp/target_y", [120, 255, 120], "target y", 2.0),
            ("tcp/target_z", [120, 160, 255], "target z", 2.0),
            ("tcp/actual_x", [140, 40, 40],   "actual x", 1.0),
            ("tcp/actual_y", [40, 140, 40],   "actual y", 1.0),
            ("tcp/actual_z", [40, 60, 140],   "actual z", 1.0)):
        rr.log(path, rr.SeriesLines(colors=col, names=nm, widths=w), static=True)

    pygame.init()
    pygame.joystick.init()
    jm = JoystickManager()                     # station.py's hot-plug handler
    if not jm.connected:
        print("Waiting for the gamepad (it carries the clutch and the gripper)... "
              "plug it in, or Ctrl-C.")
    while not jm.connected:                    # wait rather than exit: it may not be in yet
        for e in pygame.event.get():
            jm.handle_event(e)
        time.sleep(0.15)

    def buttons_for(profile):
        b = {n: button_index(profile, n) for n in BUTTONS}
        missing = [k for k, v in b.items() if v is None]
        if missing:
            print(f"  layout has no {missing} — those controls are dead")
        return b

    B = buttons_for(jm.profile)
    deb = ButtonDebouncer()                    # this pad chatters; debounce every button

    aid = imu = None
    if not args.no_imu:
        from pad_imu import PadIMU, load_calib
        cal = load_calib()
        if cal is None:
            print("  IMU: no outputs/calib/pad_imu.json — run `python pad_imu.py align` "
                  "to use the controller's own gyro (vision-only until then)")
        else:
            try:
                imu = PadIMU(bias=cal["bias"])
                imu.start()
                time.sleep(0.3)
                if imu.n == 0:
                    raise RuntimeError("no samples — controller asleep?")
                aid = InertialAid(imu, cal["X"])
                print(f"  IMU: {imu.name}, align residual {cal['rms_dps']:.2f} deg/s")
            except Exception as e:
                print(f"  IMU unavailable ({e}) — vision only")
                imu = aid = None

    view = None
    if args.view != "none":
        try:
            view = ViewCam(args.view)
            print(f"  operator view: {args.view}")
        except Exception as e:                     # shared USB hub; never fatal
            print(f"  operator view '{args.view}' unavailable ({e}) — continuing without")

    log = writer = None
    if not args.no_log:
        run = LOG_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run.mkdir(parents=True, exist_ok=True)
        log = open(run / "teleop.csv", "w", newline="")
        writer = csv.writer(log)
        writer.writerow(COLUMNS)
        print(f"  logging to {run}")

    robot, twin = None, None
    if args.sim:
        from sim_backend import SimRobot
        from pick_ball import Twin
        robot = SimRobot()
        robot.connect()
        # a viewer, so --sim actually SHOWS the arm rather than just moving numbers.
        # It renders on its own thread and competes with the Rerun window for the iGPU.
        if not args.no_twin:
            twin = Twin(np.array([0.30, 0.0, -0.029]), 0.0245, z_table=-0.029)
    elif not args.dry_run:
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
        robot = SOFollower(SOFollowerRobotConfig(port=args.port, id="so101", cameras={}))
        robot.connect()

    sender = Sender(robot) if robot else None
    if sender:
        sender.start()

    # An IK-CONSISTENT ready pose. A hand-picked one is not a fixed point of ik(): the
    # first call silently reconfigures elbow/wrist by ~86 deg to satisfy the approach
    # -direction preference while holding the same TCP. Harmless once, but it looks
    # exactly like a discontinuity and trips the jump guard, after which nothing moves.
    _, _p_ready = kin.fk(dict(READY_POSE))
    READY, _, _ = kin.ik(_p_ready, dict(READY_POSE), iters=400)

    ang = read_angles(robot) if robot else dict(READY)
    _, tcp = kin.fk(ang)
    tcp_now = tcp.copy()               # the clutch reads it before the first IK block
    filt = Filter()
    engaged = False
    anchor_hand = anchor_tcp = None
    guess = None
    target = tcp.copy()
    prev = {k: False for k in B}
    paused = False
    readied = False
    lost = 0
    n, t_fps, t0 = 0, time.perf_counter(), time.perf_counter()
    t_img = t_edge = t_diag = t_twin = 0.0
    t_prev = t0
    twist = 0.0
    hand_prev = hand_last = vis_R_prev = None
    t_hand_prev = 0.0
    imu_n_prev = 0
    pinned = []
    hz = 0.0
    damp = DAMP_LADDER[0]
    js = jm.joystick

    rr.send_blueprint(build_blueprint(args, aid is not None))

    mode = ("[SIM — MuJoCo twin]" if args.sim else
            "[DRY RUN — robot not connected]" if args.dry_run else "[REAL ARM]")
    print(f"\nREADY. axes '{args.axes}', gain {args.gain}, "
          f"max {args.max_speed * 100:.0f} cm/s  {mode}\n"
          f"  A = ready pose | hold L = engage | R-stick up/down = grip | "
          f"Home = pause/resume | B = quit")
    try:
        while True:
            _t_a = time.perf_counter()
            frame, K = src.grab()
            if frame is None:
                continue
            t = time.perf_counter()
            ms_grab = (t - _t_a) * 1e3
            dt = min(max(t - t_prev, 1e-3), DT_MAX)
            hz = 1.0 / dt
            t_prev = t
            rr.set_time("time", duration=t - t0)
            was = jm.connected
            for e in pygame.event.get():
                jm.handle_event(e)
            if jm.connected and not was:               # re-plugged: layout may differ
                B = buttons_for(jm.profile)
            js = jm.joystick
            if js is None:                             # unplugged mid-run: hold, don't drop
                down = {k: False for k in B}
                engaged, paused = False, True
            else:
                down = {k: deb(i, bool(js.get_button(i)))
                        if (i is not None and js.get_numbuttons() > i) else False
                        for k, i in B.items()}

            if down["B"] and not prev["B"]:
                print("\n[B] quitting")
                break
            if down["Home"] and not prev["Home"]:
                paused = not paused
                if paused:
                    engaged = False
                    filt.reset()
                    if sender:
                        sender.pause()             # hold wherever we are
                print(f"\n[Home] {'PAUSED — arm holding' if paused else 'resumed'}")
            if down["A"] and not prev["A"] and not engaged and not paused:
                print("\n[A] easing to the ready pose")
                readied = True
                if robot:
                    sender.pause()                 # move_to writes the bus itself
                    ang = move_to(robot, read_angles(robot), dict(READY), seconds=3.0)
                    sender.resync(ang)
                else:
                    ang = dict(READY)
                _, tcp = kin.fk(ang)
                target = tcp.copy()

            # ── perception ────────────────────────────────────────────────────────
            _t_b = time.perf_counter()
            dets = detect(det, frame, keep)
            edge = edge_margin(dets, frame.shape)
            out = body_pose(dets, K, model, guess) if dets else None
            hand = hand_R = None
            raw = None
            rep = float("nan")
            if out is not None and out[2] <= MAX_REPROJ:
                lost = 0
                R_b, t_b, err = out
                rep = err
                guess = (R_b, t_b)
                raw = t_b
                hand = filt(t_b, dt)
                hand_R = R_b
                rr.log("hand/reproj_px", rr.Scalars(err))
                if engaged and edge < EDGE_WARN_PX and t - t_edge > 1.0:
                    t_edge = t
                    print(f"\n  tag near the frame edge ({edge:.0f} px) — "
                          f"re-clutch before it leaves the view")
                yaw, pitch, roll = rpy_deg(R_b)                # measured, NOT commanded
                for k, v in (("yaw", yaw), ("pitch", pitch), ("roll", roll)):
                    rr.log(f"hand/rot_{k}_deg", rr.Scalars(float(v)))
            else:
                lost += 1
                if out is None:
                    guess = None
                if lost == LOST_MAX:      # tag gone long enough that the hand has moved
                    filt.reset()
                    if engaged:
                        engaged = False
                        print("\n  tag lost — disengaged, press L to re-anchor")
            ms_detect = (time.perf_counter() - _t_b) * 1e3

            # ── inertial aid ──────────────────────────────────────────────────────
            gyro_twist = None
            bridge_mm = 0.0
            if aid is not None:
                if hand is not None:
                    hand_prev, hand_last, t_hand_prev = hand, hand, t
                fused, rate = aid.update(dt, hand_R, hand)
                gyro_twist = rate
                bridge_mm = aid.sigma_mm          # the filter's OWN confidence, 1 sigma
                if not args.no_bridge and fused is not None and engaged:
                    # Use the fused estimate whenever the filter still trusts itself.
                    # This replaces the old time+distance caps: sigma grows on its own
                    # while blind, at a rate set by the calibration quality, so ONE
                    # number decides both "is the dead reckoning still good" and "how
                    # long may a dropout last" -- and it tightens automatically as the
                    # calibration improves.
                    if aid.blind <= BLIND_MAX_S:
                        hand = fused
                        lost = 0
                    elif hand is None and engaged:
                        engaged = False
                        print(f"\n  no tag for {aid.blind:.1f} s — disengaged, "
                              f"press L to re-anchor")

            if filt.relocated:            # re-acquired somewhere else: the anchor is stale
                filt.relocated = False
                if engaged:
                    engaged = False
                    print("\n  hand relocated — disengaged, press L to re-anchor")

            # ── clutch ────────────────────────────────────────────────────────────
            if down["L"] and not prev["L"] and not paused:
                if not readied:
                    print("\n  press A first — the arm must start from the ready pose")
                elif hand is None:
                    print("\n  can't engage: no tag in view")
                else:
                    # Anchor to the arm's REAL TCP, not to `target`. After a
                    # disengage `target` is frozen wherever it was -- often past the
                    # reach limit -- so anchoring to it re-engaged already slipping
                    # ("engaged ... trk 50.0 mm SLIP (out of reach)" in the 2026-08-21
                    # logs, on the very first tick).
                    target = tcp_now.copy()
                    anchor_hand, anchor_tcp = hand.copy(), target.copy()
                    anchor_rot, anchor_roll = hand_R.copy(), ang["wrist_roll"]
                    twist = 0.0
                    engaged = True
                    print(f"\n  [L] engaged at hand {np.round(hand * 1e3).astype(int)} mm")
            if not down["L"] and engaged:
                engaged = False
                filt.reset()
                print("\n  [L] released — holding")

            # ── map hand -> TCP ───────────────────────────────────────────────────
            tgt_cmd, roll_cmd, clamped = target, ang["wrist_roll"], 0.0
            if engaged and hand is not None and not paused:
                want = anchor_tcp + args.gain * (M @ (hand - anchor_hand))
                caged = clamp_target(want, args.cage)
                clamped = float(np.linalg.norm(caged - want))
                want = caged
                step = want - target
                lim = args.max_speed * dt
                if (d := np.linalg.norm(step)) > lim:
                    step *= lim / d
                tgt_cmd = target + step

                # Wrist roll is FREE: _ik_pass never writes dq[4], so wrist_roll is
                # whatever we hand it and costs the position solution nothing. That
                # makes it the one hand-orientation DoF we can follow without the
                # 5-DoF arm having to trade position away for it (pitch/yaw would).
                # Twist = rotation of the body about the camera's VIEWING axis,
                # measured relative to the engage pose -- a rotation vector, not an
                # Euler angle, so it does not gimbal as you tilt the controller.
                if args.roll_gain and (hand_R is not None or gyro_twist is not None):
                    # Complementary: the gyro PREDICTS at its own ~77 Hz, vision CORRECTS
                    # the drift at 13 Hz. Integrating omega_z is only an approximation of
                    # the twist angle when the controller is also pitching/yawing, which
                    # is precisely why the vision term has to stay.
                    if gyro_twist is not None:
                        twist += gyro_twist * dt
                    if hand_R is not None:
                        rv = cv2.Rodrigues(hand_R @ anchor_rot.T)[0].ravel()
                        tau_c = TWIST_TAU if gyro_twist is not None else TAU
                        vis_twist = math.degrees(rv[2])
                        twist += alpha_for(dt, tau_c) * (vis_twist - twist)
                        rr.log("delay/twist_vision_deg", rr.Scalars(vis_twist))
                    rr.log("delay/twist_fused_deg", rr.Scalars(twist))
                    roll_want = np.clip(anchor_roll + args.roll_gain * twist, *ROLL_RANGE)
                    dr = np.clip(roll_want - ang["wrist_roll"],
                                 -ROLL_RATE * dt, ROLL_RATE * dt)
                    roll_cmd = ang["wrist_roll"] + float(dr)

            # ── gripper: right stick Y, proportional (further = faster) ───────────
            v = (filtered_axis_value(js, jm.profile, GRIP_AXIS)
                 if js is not None and js.get_numaxes() > GRIP_AXIS else 0.0)
            g = ang["gripper"] - v * GRIP_RATE * dt            # stick up is -Y = open
            ang["gripper"] = float(np.clip(g, *GRIP_RANGE))

            # ── IK + send ─────────────────────────────────────────────────────────
            # 30 iterations, not the 400 default: warm-started from the last pose the
            # steps are tiny, and 400/120/60/30 all converge to the same 1.9 mm (measured)
            # while 120 would cost 20 ms of a 33 ms frame budget.
            _t_c = time.perf_counter()
            arm = [j for j in MOTOR_NAMES if j != "gripper"]
            # Walk UP the damping ladder until the joint step is small enough to apply.
            # A bare refusal DEADLOCKS: with `ang` unchanged the next tick re-solves from
            # the same state and gets the same answer, so the identical jump repeats
            # forever -- 94 consecutive refused ticks in the 2026-08-20 log, 7.4 s during
            # which the target advanced 34 mm and the arm moved 0.0 mm. More damping is
            # the honest fix, not a bigger tolerance: it asks the same question with a
            # shorter step instead of giving up.
            if not engaged:
                # IDLE: the arm holds, so there is nothing to solve -- the target is
                # frozen and `ang` already realises it. Re-solving it every tick was
                # 76-88% of all solves in the 2026-08-21 logs, at 135-230 ms each,
                # because the frozen target is often UNREACHABLE and so walks the whole
                # ladder to the same failure. Roll and gripper are direct joint commands
                # and need no IK, so they stay live while idle.
                perr = aerr = jump = 0.0
                damp = DAMP_LADDER[0]
                ok = True
                ang = dict(ang)
                ang["wrist_roll"] = roll_cmd
                if sender and not paused:
                    sender.set(ang)
            else:
                for damp in DAMP_LADDER:
                    sol, perr, aerr = kin.ik(tgt_cmd, ang, iters=30, damping=damp)
                    jump = max(abs(sol[j] - ang[j]) for j in arm)
                    if jump <= MAX_JOINT_JUMP or perr > HOPELESS:
                        break
                # Do NOT gate on perr. It is the IK's orientation-preference offset, not
                # a reachability measure: 2.8 mm at the settled ready pose but 27.8 mm at
                # the sim's zero pose, so a perr gate refuses every command forever and
                # the arm never moves (observed 2026-08-20). Gate only on discontinuity,
                # and hold the TARGET back on a leash if the arm is not keeping up --
                # that is what actually prevents a runaway.
                ok = jump <= MAX_JOINT_JUMP
                if ok:
                    sol["gripper"] = ang["gripper"]
                    sol["wrist_roll"] = roll_cmd
                    ang = sol
                    if sender and not paused:
                        sender.set(ang)
            ms_ik = (time.perf_counter() - _t_c) * 1e3

            _t_t = time.perf_counter()
            if twin and t - t_twin >= 1.0 / TWIN_HZ:
                t_twin = t
                twin.set(ang)
            ms_twin = (time.perf_counter() - _t_t) * 1e3
            _, tcp_now = kin.fk(ang)
            track = float(np.linalg.norm(tgt_cmd - tcp_now))
            held = track > LEASH
            if not held:                        # arm keeping up: let the target advance
                target = tgt_cmd
            if not ok and engaged and hand is not None:
                # The ladder rescued every refusal in replay, so this should never fire.
                # It exists because the failure it prevents is WINDUP: keep commanding a
                # target the arm cannot reach and the error grows without bound, then
                # discharges as a lurch the moment it becomes reachable. Re-anchoring
                # here makes the clutch SLIP at the limit, the way a mouse does at the
                # edge of the screen -- you push, nothing happens, you re-reference.
                anchor_hand, anchor_tcp = hand.copy(), tcp_now.copy()
                target = tcp_now.copy()
            pinned = at_limit(kin, ang)
            _t_d = time.perf_counter()
            if not args.no_rr_images and t - t_img >= 1.0 / IMG_HZ:
                t_img = t
                rr.log("world/cam",
                       rr.Pinhole(image_from_camera=Km(K),
                                  resolution=[frame.shape[1], frame.shape[0]]))
                sc = args.rr_scale
                small = frame if sc >= 0.999 else cv2.resize(frame, None, fx=sc, fy=sc)
                rr.log("world/cam/image",
                       rr.Image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                         .compress(jpeg_quality=70))
                if view is not None and (vf := view.latest()) is not None:
                    vs = vf if sc >= 0.999 else cv2.resize(vf, None, fx=sc, fy=sc)
                    rr.log(f"view/{args.view}",
                           rr.Image(cv2.cvtColor(vs, cv2.COLOR_BGR2RGB))
                             .compress(jpeg_quality=70))
            diag = t - t_diag >= 1.0 / DIAG_HZ
            if diag:
                t_diag = t
            for k, v in () if not diag else (("engaged", float(engaged)), ("tags", len(dets)),
                         ("track_mm", track * 1e3), ("ik_ok", float(ok)),
                         ("dropped", float(filt.dropped)),
                         # the three "why did it stop following me" signals, so the
                         # cause is visible in the plot instead of being guessed at
                         ("edge_px", float(edge)),          # about to leave the FOV
                         ("at_limit", float(len(pinned))),  # joint out of road
                         ("clamped_mm", clamped * 1e3),     # hit the safety cage
                         ("leash_held", float(held)),
                         ("ik_perr_mm", perr * 1e3), ("ik_jump_deg", jump),
                         ("ik_damp", damp),
                         ("lost", float(lost))):
                rr.log(f"teleop/{k}", rr.Scalars(v))
            if diag:
                for j in MOTOR_NAMES:
                    rr.log(f"joints/{j}", rr.Scalars(float(ang[j])))
            if diag and aid is not None:
                rr.log("teleop/bridge_mm", rr.Scalars(bridge_mm))
                rr.log("teleop/blind_s", rr.Scalars(float(aid.blind)))
                # THE DELAY PLOT. Both sensors watch the same rotation, so |omega| must
                # agree -- overlay them and any lag between the curves IS the pipeline
                # delay, in a form you can read straight off the timeline. The gyro is
                # the fast one; vision is what arrives late.
                rr.log("delay/omega_gyro_dps", rr.Scalars(aid.speed))
                if hand_R is not None and vis_R_prev is not None and dt > 1e-4:
                    dR = cv2.Rodrigues(vis_R_prev.T @ hand_R)[0].ravel()
                    rr.log("delay/omega_vision_dps",
                           rr.Scalars(float(np.degrees(np.linalg.norm(dR)) / dt)))
                for i, ax in enumerate("xyz"):
                    rr.log(f"imu/gyro_cam_{ax}_dps",
                           rr.Scalars(float(np.degrees(aid.w_cam[i]))))
                rr.log("imu/rate_hz", rr.Scalars(float(imu.n - imu_n_prev) / max(dt, 1e-6)))
            if hand_R is not None:
                vis_R_prev = hand_R
            imu_n_prev = imu.n if imu is not None else 0
            for i, axis in enumerate("xyz"):
                rr.log(f"tcp/target_{axis}", rr.Scalars(float(target[i])))
                rr.log(f"tcp/actual_{axis}", rr.Scalars(float(tcp_now[i])))

            ms_rerun = (time.perf_counter() - _t_d) * 1e3
            ms_rest = ((time.perf_counter() - t) * 1e3
                       - ms_detect - ms_ik - ms_twin - ms_rerun)

            if writer is not None:
                nan3 = (float("nan"),) * 3
                writer.writerow([f"{t - t0:.4f}", f"{hz:.1f}", int(engaged), int(paused),
                                 len(dets), f"{rep:.2f}", f"{edge:.1f}",
                                 *[f"{v:.5f}" for v in (raw if raw is not None else nan3)],
                                 *[f"{v:.5f}" for v in (hand if hand is not None else nan3)],
                                 filt.dropped, lost,
                                 *[f"{v:.5f}" for v in tgt_cmd],
                                 *[f"{v:.5f}" for v in tcp_now],
                                 f"{track * 1e3:.2f}", f"{clamped * 1e3:.2f}", int(held),
                                 int(ok), f"{jump:.2f}", f"{perr * 1e3:.2f}",
                                 f"{aerr:.1f}", ";".join(pinned), f"{damp:.4f}",
                                 f"{twist:.1f}", f"{bridge_mm:.1f}",
                                 f"{ms_grab:.2f}", f"{ms_detect:.2f}", f"{ms_ik:.2f}",
                                 f"{ms_twin:.2f}", f"{ms_rerun:.2f}", f"{ms_rest:.2f}",
                                 *[f"{ang[j]:.2f}" for j in MOTOR_NAMES]])

            prev = down                # rising-edge state; without this every held
                                       # button re-fires its action on every frame
            n += 1
            # 4 Hz while engaged: fast enough to WATCH the axis mapping respond as you
            # move your hand, which is the one thing that has to be right before the
            # arm is allowed to move.
            if t - t_fps >= (0.25 if engaged else 1.0):
                state = "PAUSED " if paused else ("ENGAGED" if engaged else "idle   ")
                if engaged and hand is not None:
                    dh = (hand - anchor_hand) * 1e3            # camera frame
                    dt_ = (target - anchor_tcp) * 1e3          # base frame
                    extra = (f"| hand d{np.round(dh).astype(int)} mm "
                             f"-> tcp d{np.round(dt_).astype(int)} mm ")
                else:
                    extra = f"| tcp {np.round(tcp_now * 1e3).astype(int)} mm "
                # Name the ACTIVE limiter. "It stopped following me" has four very
                # different causes and they are indistinguishable from the arm.
                why = ("FOV!" if edge < EDGE_WARN_PX else
                       "limit:" + ",".join(j[:5] for j in pinned) if pinned else
                       "caged" if clamped > 1e-4 else
                       "SLIP (out of reach)" if not ok else
                       "lagging" if held else
                       f"damped {damp:.2f}" if damp > DAMP_LADDER[0] else "")
                print(f"\r{n / (t - t_fps):4.1f} Hz | {state} | tags {len(dets)} "
                      f"{extra}| grip {ang['gripper']:.0f} | roll {ang['wrist_roll']:+4.0f} "
                      f"| trk {track * 1e3:4.1f} mm | drop {filt.dropped} {why:22s}",
                      end="", flush=True)
                n, t_fps = 0, t
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        src.stop()
        if view is not None:
            view.stop()
        if imu is not None:
            imu.stop()
        if log is not None:
            log.close()
            (run / "summary.txt").write_text(
                f"teleop_tag  {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                f"t0_unix_epoch {time.time() - (time.perf_counter() - t0):.6f}\n"
                f"mode {mode}  axes {args.axes}  gain {args.gain}  "
                f"roll_gain {args.roll_gain}  max_speed {args.max_speed}\n"
                f"source {args.source}  view {args.view}\n"
                f"TAU {TAU}  JUMP_MAX {JUMP_MAX}  LEASH {LEASH}  "
                f"MAX_JOINT_JUMP {MAX_JOINT_JUMP}\n"
                f"BOX {BOX}  REACH {REACH}\n"
                f"dropped {filt.dropped}\n")
            print(f"\nlog: {run}")
        if twin:
            twin.close()
        if sender:
            sender.stop()
        if robot:
            print("landing...")
            if not args.sim:              # graceful_shutdown drives Torque_Limit, a real
                graceful_shutdown(robot)  # Feetech RAM register the sim bus has no use for
            robot.disconnect()
        pygame.quit()
        print("done")


if __name__ == "__main__":
    main()
