#!/usr/bin/env python
"""pad_imu.py — the Switch Pro Controller's OWN 6-axis IMU, as a teleop sensor.

The controller you already hold has a factory-calibrated accelerometer + gyroscope, and
`hid_nintendo` publishes it as a SECOND evdev device named "Pro Controller (IMU)",
alongside the normal joystick node. So the tag-tracked controller is already an inertial
tracker: no board, no battery, no cable, and it is rigidly attached to the tags by
construction. Measured 2026-08-20 over Bluetooth:

    accel 4096 units/g, noise 17.6 mm/s^2 rms   gyro 14247 units per deg/s (+-2300),
    noise 0.06-0.22 deg/s, bias [3.75 4.27 -1.71] deg/s (constant -> calibratable)
    delivery is BURSTY: 3 samples every ~13 ms, so 200 Hz of samples but new
    information at ~77 Hz -- still 6x the 13 Hz vision loop

What it is good for, and what it is not
---------------------------------------
GOOD: bridging a tag dropout, and adding detail to FAST motion. Dead-reckoning drift,
measured by double-integrating a genuinely stationary recording, is 0.7 mm over 0.5 s and
2.1 mm over 1 s. Every one of the 59 tag-loss episodes in the 2026-08-20 teleop log was
under 1 s, so this sensor covers all of them.

NOT GOOD: slow, small, delicate motion. Double-integrating a small acceleration is
noise-dominated and the answer drifts; vision is the better sensor there and should stay
in charge. Nothing here replaces the tag -- it only fills the gaps between tag fixes.

Frames
------
The gyro reports in the IMU's own frame. Everything else in this project works in the tag
BODY frame (vision/tag_body.py). Those differ by a fixed rotation X (v_body = X @ v_imu)
set by how the IMU sits inside the shell, which no datasheet will tell us -- so `align`
solves it from a recording: rotate the controller in view of the camera, match the gyro's
angular velocity against the one vision measures, and take the Procrustes fit.

    python pad_imu.py probe            # live values -- check the sensor is alive
    python pad_imu.py bias             # gyro bias, controller STILL (2 s)
    python pad_imu.py align            # solve X: rotate the controller in view (~30 s)
    python pad_imu.py selftest         # solver check, no hardware

Access: the IMU node is root:input, so you must be in the `input` group:
    sudo usermod -aG input $USER          # then LOG OUT and back in
Group membership is fixed at login, so a shell opened before that still cannot read it.
Rather than re-login, any such shell can borrow the group for one command:
    sg input -c "$CONDA_PREFIX/bin/python pad_imu.py align"
(`setfacl` on the device node also works, but does NOT survive the controller
reconnecting -- udev makes a fresh node each time.)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

CALIB = ROOT / "outputs/calib/pad_imu.json"
# The tag body the controller carries. Now the 3-face box (vision/box_tags.py fit);
# the old two-tag joystick_body.json still loads via --model.
BODY_MODEL = ROOT / "outputs/calib/box_body.json"
# The driver's reported gyro resolution reads ~14% HIGH on this controller. Measured
# 2026-08-31 by two independent methods that agree to 0.001: vision (tag-body rotation vs
# gyro, median ratio 0.876) and gravity alone (still->still accelerometer transitions, no
# camera, 0.875). Correcting it takes the gravity residual from 12.32 deg to 1.72 deg over
# 94 deg rotations. Re-measure per controller with `pad_imu.py gyrocal`; the value in the
# calib file wins over this default.
GYRO_SCALE_DEFAULT = 0.877
STILL_SD_DPS = 3.0                 # gyro sd above this means it was NOT still
LOG_ROOT = ROOT / "outputs/pad_imu"
G = 9.80665


def _saved_gyr_scale():
    """Calibrated gyro scale if `gyrocal` has been run, else the measured default."""
    try:
        return float(json.loads(CALIB.read_text())["gyr_scale"])
    except Exception:
        return GYRO_SCALE_DEFAULT


def find_imu():
    """Path of the controller's IMU evdev node, or None.

    NOT evdev.list_devices(): that filters to devices openable READ-WRITE, and read-only
    is both all we need and all a setfacl grant gives -- so the device is silently absent
    from that list (cost me a debugging round trip on 2026-08-20)."""
    import evdev
    for path in sorted(glob.glob("/dev/input/event*")):
        try:
            d = evdev.InputDevice(path)
        except OSError:
            continue
        if "IMU" in d.name:
            return path
    return None


def _no_imu_reason():
    """Say which of the FOUR things actually went wrong, instead of always blaming groups.

    The previous message blamed the 'input' group unconditionally. On 2026-08-21 that was
    wrong in the most expensive way: the group was correct, `sg input` was being used, and
    the controller had simply been asleep -- its (IMU) node appears a moment AFTER the
    joystick node. A confident misdiagnosis costs more than no diagnosis."""
    import grp
    import os
    listed = Path("/proc/bus/input/devices").read_text()
    has_pad = 'Name="Pro Controller"' in listed
    has_imu = 'Name="Pro Controller (IMU)"' in listed
    denied = []
    for path in sorted(glob.glob("/dev/input/event*")):
        try:
            os.close(os.open(path, os.O_RDONLY))
        except PermissionError:
            denied.append(path)
        except OSError:
            pass
    try:
        in_group = grp.getgrnam("input").gr_gid in os.getgroups()
    except KeyError:
        in_group = False

    if has_imu and denied and not in_group:
        return ("the IMU node exists but this process is NOT in the 'input' group:\n"
                '    sg input -c "$CONDA_PREFIX/bin/python pad_imu.py ..."\n'
                "  (or `sudo usermod -aG input $USER` once, then log out and back in)")
    if has_imu and denied:
        return (f"'Pro Controller (IMU)' is listed and this process IS in the 'input' "
                f"group, yet {denied[0]} is unreadable — check the node's ACL:\n"
                "    getfacl /dev/input/event*\n"
                "    sudo setfacl -m u:$USER:r /dev/input/eventN")
    if has_pad and not has_imu:
        return ("the controller is connected but its '(IMU)' node has not appeared — it "
                "is created a moment after the joystick node, and not at all while the "
                "pad is asleep. Press Home, wait ~2 s, and retry.")
    return ("no 'Pro Controller' in /proc/bus/input/devices — the controller is not "
            "connected. Press Home on it to wake the Bluetooth link.")


class PadIMU(threading.Thread):
    """Background reader. Keeps the newest sample and integrates body rotation.

    Scale factors come from the driver's own `resolution` fields rather than a constant,
    because that is where the factory calibration lands: accel in units/g, gyro in units
    per deg/s (Linux input event-codes convention for INPUT_PROP_ACCELEROMETER)."""

    def __init__(self, path=None, bias=None, gyr_scale=None):
        super().__init__(daemon=True)
        import evdev
        from evdev import ecodes
        self.ecodes = ecodes
        path = path or find_imu()
        if path is None:
            raise RuntimeError(_no_imu_reason())
        self.dev = evdev.InputDevice(path)
        self.name = self.dev.name

        absinfo = dict(self.dev.capabilities(absinfo=True)).get(ecodes.EV_ABS, [])
        res = {c: i.resolution for c, i in absinfo}
        self.acc_per_g = float(res.get(ecodes.ABS_X) or 4096)
        self.gyr_per_dps = float(res.get(ecodes.ABS_RX) or 14247)
        self.gyr_scale = float(gyr_scale) if gyr_scale is not None else _saved_gyr_scale()
        self._slot = {ecodes.ABS_X: 0, ecodes.ABS_Y: 1, ecodes.ABS_Z: 2,
                      ecodes.ABS_RX: 3, ecodes.ABS_RY: 4, ecodes.ABS_RZ: 5}

        self.bias = np.zeros(3) if bias is None else np.asarray(bias, float)
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self._raw = [0] * 6
        self.acc = np.zeros(3)          # m/s^2, IMU frame
        self.gyr = np.zeros(3)          # rad/s, IMU frame, bias removed
        self.t = 0.0                    # host perf_counter of the newest sample
        self.n = 0
        self._hist = []                 # (t, gyr, acc) ring for align/replay
        self.keep_history = False

    # ── reader ───────────────────────────────────────────────────────────────────
    def run(self):
        ec = self.ecodes
        try:
            self._read(ec)
        except OSError:
            pass          # stop() closes the fd while read_loop blocks; that is the exit

    def _read(self, ec):
        for e in self.dev.read_loop():
            if self._halt.is_set():
                break
            if e.type == ec.EV_ABS and e.code in self._slot:
                self._raw[self._slot[e.code]] = e.value
            elif e.type == ec.EV_SYN:
                t = time.perf_counter()
                a = np.array(self._raw[:3], float) / self.acc_per_g * G
                w = np.radians(np.array(self._raw[3:], float)
                               / self.gyr_per_dps) * self.gyr_scale
                with self._lock:
                    self.acc, self.gyr, self.t = a, w - self.bias, t
                    self.n += 1
                    if self.keep_history:
                        self._hist.append((t, self.gyr.copy(), a))

    def latest(self):
        with self._lock:
            return self.t, self.gyr.copy(), self.acc.copy()

    def history(self, clear=True):
        with self._lock:
            h = self._hist
            if clear:
                self._hist = []
        return h

    def stop(self):
        self._halt.set()
        try:                                   # read_loop blocks; nudge it awake
            self.dev.close()
        except Exception:
            pass

    # ── calibration ──────────────────────────────────────────────────────────────
    def measure_bias(self, seconds=2.0):
        """Mean gyro output while the controller is STILL = the bias to subtract."""
        with self._lock:
            self.bias = np.zeros(3)
        t0, samples = time.perf_counter(), []
        while time.perf_counter() - t0 < seconds:
            t, w, _ = self.latest()
            if t > t0:
                samples.append(w)
            time.sleep(0.004)
        if len(samples) < 20:
            raise RuntimeError(f"only {len(samples)} samples — is the controller awake?")
        b = np.mean(samples, axis=0)
        with self._lock:
            self.bias = b
        return b, np.std(samples, axis=0)


# ── IMU -> tag-body rotation ──────────────────────────────────────────────────────

def solve_alignment(w_imu, w_body, weights=None):
    """Rotation X with v_body = X @ v_imu, from paired angular velocities.

    Orthogonal Procrustes. Angular velocity is a free vector, so pairing it needs no
    positions and no lever arm -- which is why this works from a hand-waving recording
    and needs no rig."""
    A = np.asarray(w_imu, float)
    B = np.asarray(w_body, float)
    if weights is not None:
        w = np.asarray(weights, float)[:, None]
        A, B = A * w, B * w
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def omega_from_rotations(R_list, t_list):
    """BODY-frame angular velocity between consecutive camera-measured rotations.

    Body frame, not camera frame: the gyro measures in its own moving frame, so the
    vision side must be expressed the same way or the fit is meaningless.
    R_k^T R_k+1 is the increment seen from the body; its log is omega*dt."""
    import cv2
    out_t, out_w = [], []
    for k in range(len(R_list) - 1):
        dt = t_list[k + 1] - t_list[k]
        if not (1e-4 < dt < 0.5):
            continue
        rv = cv2.Rodrigues(R_list[k].T @ R_list[k + 1])[0].ravel()
        out_t.append(0.5 * (t_list[k] + t_list[k + 1]))
        out_w.append(rv / dt)
    return np.array(out_t), np.array(out_w)


def best_time_shift(t_i, w_i, t_v, w_v, span=0.12, step=0.002):
    """Lag (s) to ADD to the vision timestamps to line them up with the IMU.

    Bluetooth delivery and the camera pipeline have different, unknown latencies, and a
    misalignment of even one vision frame rotates the fit. Found by correlating the two
    |omega| envelopes, which needs no knowledge of X."""
    n_i = np.linalg.norm(w_i, axis=1)
    n_v = np.linalg.norm(w_v, axis=1)
    if len(n_v) < 10 or n_v.std() < 1e-6:
        return 0.0, 0.0
    best, best_c = 0.0, -2.0
    for lag in np.arange(-span, span + 1e-9, step):
        s = np.interp(t_v + lag, t_i, n_i)
        if s.std() < 1e-9:
            continue
        c = float(np.corrcoef(s, n_v)[0, 1])
        if c > best_c:
            best, best_c = float(lag), c
    return best, best_c


def integrate_gyro(t_i, w_i):
    """Cumulative rotation of the IMU frame, one matrix per sample."""
    import cv2
    Q = np.empty((len(t_i), 3, 3))
    Q[0] = np.eye(3)
    for k in range(1, len(t_i)):
        Q[k] = Q[k - 1] @ cv2.Rodrigues(w_i[k - 1] * (t_i[k] - t_i[k - 1]))[0]
    return Q


def rotation_pairs(Rv, tv, t_i, Q, lag, window):
    """Matched rotation INCREMENTS over `window` seconds -> (r_vision, r_imu).

    Increments, not derivatives. omega from finite-differenced poses is hopeless here:
    the camera runs at ~20 Hz, so a 1 deg pose error becomes 25 deg/s of fake angular
    velocity against a real signal of ~28 deg/s (measured 2026-08-20). Over a 0.35 s
    baseline the same pose error is a few percent of a 14 deg rotation instead.

    The pairing is exact, not an approximation: the same physical rotation seen in the
    body frame and in the IMU frame differ by conjugation, dR_body = X dR_imu X^T, so
    their rotation VECTORS satisfy r_body = X r_imu -- identical angle, rotated axis."""
    import cv2
    rv, ri, at = [], [], []
    j = 0
    for k in range(len(tv)):
        while j < len(tv) and tv[j] - tv[k] < window:
            j += 1
        if j >= len(tv):
            break
        a = min(np.searchsorted(t_i, tv[k] + lag), len(Q) - 1)
        b = min(np.searchsorted(t_i, tv[j] + lag), len(Q) - 1)
        rv.append(cv2.Rodrigues(Rv[k].T @ Rv[j])[0].ravel())
        ri.append(cv2.Rodrigues(Q[a].T @ Q[b])[0].ravel())
        at.append(tv[k])
    return np.array(rv), np.array(ri), np.array(at)


def fit_alignment(t_i, w_i, Rv, tv, window=0.35, tol=0.20, verbose=True):
    """Solve X from a recording. Returns (X, lag, rms_deg, frac_deg, inliers, corr)."""
    tv = np.asarray(tv, float)
    t_o, w_o = omega_from_rotations(Rv, list(tv))
    lag, corr = best_time_shift(t_i, w_i, t_o, w_o)
    Q = integrate_gyro(t_i, w_i)
    rv, ri, _ = rotation_pairs(Rv, tv, t_i, Q, lag, window)
    if len(rv) < 20:
        return None, lag, float("inf"), float("inf"), np.zeros(0, bool), corr

    nv, ni = np.linalg.norm(rv, axis=1), np.linalg.norm(ri, axis=1)
    # X is a rotation, so it preserves length: |r_vision| must equal |r_imu| whatever X
    # is. Frames that disagree are bad poses, and this test needs no calibration -- which
    # is what lets it run BEFORE we have one.
    good = (ni > np.radians(8)) & (np.abs(nv - ni) < 0.25 * ni)
    if good.sum() < 15:
        return None, lag, float("inf"), float("inf"), good, corr

    # RANSAC on top: a lone visible tag is planar and two-fold ambiguous, and the
    # tracker can sit on the WRONG branch for a whole stretch -- self-consistent but
    # mirrored, so a per-frame test cannot see it. Consensus can.
    gi = np.flatnonzero(good)
    rng = np.random.default_rng(0)
    best_n, best_inl = 0, None
    for _ in range(3000):
        sel = rng.choice(gi, 3, replace=False)
        Xs = solve_alignment(ri[sel], rv[sel])
        err = np.linalg.norm((Xs @ ri.T).T - rv, axis=1)
        inl = good & (err < tol * ni)
        if inl.sum() > best_n:
            best_n, best_inl = int(inl.sum()), inl
    if best_inl is None or best_n < 15:
        return None, lag, float("inf"), float("inf"), good, corr

    X = solve_alignment(ri[best_inl], rv[best_inl])
    err = np.degrees(np.linalg.norm((X @ ri[best_inl].T).T - rv[best_inl], axis=1))
    mag = np.degrees(ni[best_inl]).mean()
    if verbose:
        print(f"  time offset {lag * 1e3:+.0f} ms (|omega| correlation {corr:.3f})")
        print(f"  {len(rv)} increment pairs over {window:.2f} s windows; "
              f"|omega| consistent: {good.sum()}; consensus inliers: {best_n}")
    return X, lag, float(err.mean()), float(mag), best_inl, corr


RAW = ROOT / "outputs/calib/pad_imu_raw.npz"
GYRO_RAW = ROOT / "outputs/calib/pad_imu_gyrocal.npz"
POSES = ROOT / "outputs/calib/pad_imu_poses.npz"


def pose_with_gravity(dets, K, model, a_imu, X, g_cam, guess=None):
    """Body pose, using the IMU to break the single-tag ambiguity.

    A lone tag is planar, so PnP returns TWO poses. They are not equally plausible once
    the accelerometer has a vote: each candidate predicts a different direction for
    gravity in the camera frame, and gravity is known and fixed. Pick the candidate whose
    prediction matches. Circular for the FIRST calibration -- but for checking or
    refining an existing one it removes the 'keep both tags visible' contortion."""
    import cv2
    from tag_body import body_pose, candidates, unit_corners
    if len(dets) >= 2 or X is None or g_cam is None:
        return body_pose(dets, K, model, guess)
    tid, c = dets[0]
    if tid not in model["tags"]:
        return None
    Rm, tm, side = model["tags"][tid]
    best = None
    for Ro, to, err in candidates(c, K, side):
        Rb = Ro @ Rm.T
        tb = to - Rb @ tm
        pred = Rb @ X @ a_imu
        n = np.linalg.norm(pred)
        if n < 1e-9:
            continue
        d = float(np.dot(pred / n, g_cam / np.linalg.norm(g_cam)))
        if best is None or d > best[0]:
            best = (d, Rb, tb, err * math.sqrt(2.0))
    if best is None:
        return None
    return best[1], best[2], best[3]


def save_raw(Rs, ts, hist):
    RAW.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RAW, R=np.array(Rs), t=np.array(ts),
                        t_imu=np.array([h[0] for h in hist]),
                        w_imu=np.array([h[1] for h in hist]),
                        a_imu=np.array([h[2] for h in hist]))
    print(f"  raw recording -> {RAW}  (re-solve offline with `align --replay`)")


def gravity_coverage(a_list):
    """How well the still poses span 3-D -> (smallest eigenvalue, max angle from mean).

    Gravity in ONE orientation pins only two of X's three degrees of freedom: nothing
    constrains rotation ABOUT gravity. So a set of poses that all face the same way can
    be fitted perfectly by a whole family of wrong X's, and its scatter looks wonderful
    while meaning nothing -- 47 frames spanning 3 deg gave every candidate an identical
    0.78 deg (measured 2026-08-21). Always read scatter next to this number."""
    U = np.asarray(a_list, float)
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    lo = float(np.linalg.svd(U.T @ U / len(U), compute_uv=False)[-1])
    m = U.mean(0)
    m /= np.linalg.norm(m)
    return lo, float(np.degrees(np.arccos(np.clip(U @ m, -1, 1))).max())


def gravity_scatter(R_list, a_list, X):
    """Spread (deg rms) of the reconstructed 'down' across still poses, for a FIXED X.

    This is the whole test in one number: gravity does not move, so a correct X gives the
    same camera-frame vector from every pose. Whatever spread remains is calibration
    error plus tag-pose noise."""
    A = np.asarray(a_list, float)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    pred = np.einsum("kij,jl,kl->ki", np.asarray(R_list, float), np.asarray(X, float), A)
    g = pred.mean(0)
    g /= np.linalg.norm(g)
    err = np.degrees(np.arccos(np.clip(pred @ g, -1, 1)))
    return float(np.sqrt((err ** 2).mean())), g


def solve_from_gravity(R_list, a_list, X0=None, iters=60, huber_deg=4.0):
    """Solve X from STATIC poses, using gravity as the reference. -> (X, g_cam, rms_deg)

    Independent of the gyro and of any timing: gravity never moves, so for every still
    pose k the SAME camera-frame vector must come back out:

        R_k @ X @ a_k = g_cam        (a_k = accelerometer while still = gravity, IMU frame)

    Alternate the two easy halves. With X fixed the best g is the (weighted) mean; with g
    fixed, X is a Procrustes fit of {a_k} onto {R_k^T g}. Both closed-form.

    ROBUST, because the input is AprilTag poses. A few degrees of pose error is NORMAL
    (sub-pixel corner noise, lighting, skew, small apparent size) and the occasional pose
    is far worse. Plain least squares lets one bad pose drag the answer, so this uses
    iteratively reweighted Huber to cap any single pose's influence. The rest averages
    down as 1/sqrt(N) -- which is why MORE POSES is the actual fix here, not a cleverer
    estimator. See gravity_uncertainty() for what N buys you."""
    A = np.asarray(a_list, float)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    R = np.asarray(R_list, float)
    X = np.eye(3) if X0 is None else np.asarray(X0, float)
    w = np.ones(len(A))
    g = None
    c = math.radians(huber_deg)
    for _ in range(iters):
        pred = np.einsum("kij,jl,kl->ki", R, X, A)      # R_k @ X @ a_k
        g = (pred * w[:, None]).sum(0) / max(w.sum(), 1e-9)
        g /= np.linalg.norm(g)
        err = np.arccos(np.clip(pred @ g, -1, 1))
        w = np.where(err <= c, 1.0, c / np.maximum(err, 1e-9))    # Huber
        X = solve_alignment(A, np.einsum("kji,j->ki", R, g), weights=w)
    pred = np.einsum("kij,jl,kl->ki", R, X, A)
    err = np.degrees(np.arccos(np.clip(pred @ g, -1, 1)))
    return X, g, float(np.sqrt((err ** 2).mean()))


def gravity_uncertainty(R_list, a_list, trials=150, seed=0):
    """Bootstrap: how well is X actually pinned down by THIS many poses? -> deg (1 sigma).

    The number that matters, and the one 'residual' cannot give you. Residual says how
    noisy each pose is; this says how much that noise still moves the ANSWER after
    averaging -- and it is what shrinks as you collect more samples."""
    rng = np.random.default_rng(seed)
    R, A = np.asarray(R_list, float), np.asarray(a_list, float)
    n = len(R)
    if n < 6:
        return float("nan")
    X0, _, _ = solve_from_gravity(R, A)
    d = [math.degrees(math.acos(np.clip(
        (np.trace(solve_from_gravity(R[i], A[i], X0=X0, iters=20)[0].T @ X0) - 1) / 2, -1, 1)))
        for i in (rng.integers(0, n, n) for _ in range(trials))]
    return float(np.sqrt(np.mean(np.square(d))))


def load_calib():
    if not CALIB.exists():
        return None
    d = json.loads(CALIB.read_text())
    if d.get("X") is None:                 # bias-only file: alignment not solved yet
        return None
    return dict(X=np.array(d["X"], float), bias=np.array(d["bias"], float),
                lag=float(d.get("lag", 0.0)), rms_dps=float(d.get("rms_dps", float("nan"))),
                gyr_scale=float(d.get("gyr_scale", GYRO_SCALE_DEFAULT)))


def save_calib(X, bias, lag, rms_dps, n):
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    d = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    d.update(X=X.tolist(), bias=list(map(float, bias)), lag=lag, rms_dps=rms_dps,
             samples=int(n), gyr_scale=_saved_gyr_scale())   # keep the gyro calibration
    CALIB.write_text(json.dumps(d, indent=2))
    print(f"  saved -> {CALIB}")


# ── CLI ───────────────────────────────────────────────────────────────────────────

def cmd_probe(args):
    imu = PadIMU()
    imu.start()
    print(f"{imu.name}: accel {imu.acc_per_g:.0f} units/g, gyro {imu.gyr_per_dps:.0f} units/dps")
    print("Ctrl-C to stop.\n")
    t0 = time.perf_counter()
    n0 = 0
    try:
        while True:
            time.sleep(0.25)
            t, w, a = imu.latest()
            rate = (imu.n - n0) / 0.25
            n0 = imu.n
            print(f"\r{rate:5.0f} Hz | gyro {np.round(np.degrees(w), 1)} deg/s "
                  f"| accel {np.round(a, 2)} m/s^2 |a|={np.linalg.norm(a):5.2f}   ",
                  end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        imu.stop()


def cmd_bias(args):
    imu = PadIMU()
    imu.start()
    time.sleep(0.4)
    print(f"Keep the controller PERFECTLY STILL for {args.seconds:.0f} s...")
    b, sd = imu.measure_bias(args.seconds)
    print(f"  bias  {np.round(np.degrees(b), 3)} deg/s")
    print(f"  noise {np.round(np.degrees(sd), 3)} deg/s rms")
    old = load_calib()
    save_calib(old["X"] if old else np.eye(3), b,
               old["lag"] if old else 0.0,
               old["rms_dps"] if old else float("nan"), 0)
    imu.stop()


def cmd_align(args):
    """Solve X by rotating the tagged controller in front of the camera."""
    import cv2
    from tag_pose import make_detector, detect, open_source
    from tag_body import load_model, body_pose

    imu = PadIMU()
    imu.start()
    time.sleep(0.4)
    print("Rest the controller ON THE TABLE and do not touch it (gyro bias, 3 s)...")
    bias, sd = imu.measure_bias(3.0)
    sd_dps = float(np.degrees(np.linalg.norm(sd)))
    print(f"  bias {np.round(np.degrees(bias), 2)} deg/s  (sd {sd_dps:.2f} deg/s)")
    if sd_dps > STILL_SD_DPS:
        imu.stop()
        sys.exit(f"\n  ABORT: the controller was MOVING during the bias measurement "
                 f"(sd {sd_dps:.2f} > {STILL_SD_DPS} deg/s).\n"
                 f"  A bias error of 10 deg/s rotates X by ~16 deg (measured), which is\n"
                 f"  most of the run-to-run spread we saw. Put it down on the table,\n"
                 f"  let go, and re-run.")
    print()

    model = load_model(args.model)
    det = make_detector("DICT_4X4_50")
    keep = set(model["tags"])
    src = open_source(args.source, args.fov)
    imu.keep_history = True
    imu.history()                                  # drop the still period

    print(f"Now ROTATE the controller in front of the camera for {args.seconds:.0f} s.")
    print("Twist it about ALL THREE axes -- roll, pitch, yaw.")
    print("KEEP 2+ TAGS IN VIEW: one tag alone is planar, so its pose is two-fold")
    print("ambiguous and the tracker can silently sit on the mirrored solution. Only")
    print("multi-tag frames are recorded here -- the counter below is what matters.")
    print("The 3-face box makes this easy: hold a CORNER toward the camera and two or")
    print("three faces stay visible through most of the rotation.")
    print("Slow, deliberate turns (~40 deg/s) beat fast ones.\n")
    rr = None
    if not args.no_rr:
        import rerun as rr
        rr.init("pad_imu_align", spawn=True)
        rr.log("axes", rr.ViewCoordinates.RUB, static=True)

    Rs, ts, guess = [], [], None
    axes, t0, last = [], time.perf_counter(), 0.0
    while (el := time.perf_counter() - t0) < args.seconds:
        frame, K = src.grab()
        if frame is None:
            continue
        dets = detect(det, frame, keep)
        out = body_pose(dets, K, model, guess) if len(dets) >= args.min_tags else None
        if out is not None and out[2] <= 4.0:
            R_b, t_b, _ = out
            guess = (R_b, t_b)
            Rs.append(R_b)
            ts.append(time.perf_counter())

        # Observability: X is solved from PAIRED angular velocities, so it is only
        # determined in the directions actually rotated about. Rotate about one axis
        # only and X is free about that axis -- invisible in the pose count, fatal in
        # the fit. The eigenvalues of sum(w_hat w_hat^T) measure exactly that span.
        _, w_now, _ = imu.latest()
        dps = math.degrees(np.linalg.norm(w_now))
        if dps > args.min_dps:
            axes.append(w_now / np.linalg.norm(w_now))
        cov = np.zeros(3)
        if len(axes) > 20:
            ev = np.linalg.eigvalsh(np.array(axes).T @ np.array(axes) / len(axes))
            cov = np.clip(ev[::-1], 0, None)          # descending, sums to 1

        if rr is not None:
            rr.set_time("time", duration=el)
            rr.log("rate/omega_dps", rr.Scalars(dps))
            rr.log("rate/n_tags", rr.Scalars(float(len(dets))))
            rr.log("rate/reproj_px", rr.Scalars(float(out[2]) if out else float("nan")))
            rr.log("coverage/worst_axis", rr.Scalars(float(cov[2])))
            for i, v in enumerate(cov):
                rr.log(f"coverage/eig{i}", rr.Scalars(float(v)))
            if len(axes) > 1 and len(Rs) % 5 == 0:
                A = np.array(axes[-1500:])
                rr.log("axes/omega", rr.Points3D(A, radii=0.008,
                                                 colors=[(90, 170, 255)]))
        if el - last >= 0.25:
            last = el
            bar = "".join("#" if c > 0.15 else ("-" if c > 0.05 else ".") for c in cov)
            print(f"\r  {el:4.1f}s  poses {len(Rs):4d}  tags {len(dets)}  "
                  f"|w| {dps:5.1f} dps  axis coverage [{bar}] "
                  f"worst {cov[2]:.3f}   ", end="", flush=True)
    print()
    src.stop()
    hist = imu.history()
    imu.stop()

    if len(Rs) < 40 or len(hist) < 100:
        sys.exit(f"not enough data (vision {len(Rs)}, imu {len(hist)}) — try again")

    save_raw(Rs, ts, hist)
    solve_and_report(Rs, ts, hist, bias, args)


def solve_and_report(Rs, ts, hist, bias, args, tags=None):
    t_i = np.array([h[0] for h in hist])
    w_i = np.array([h[1] for h in hist])
    X, lag, rms, mag, inl, corr = fit_alignment(t_i, w_i, Rs, ts,
                                                window=getattr(args, "window", 0.35))
    if X is None:
        print("\n  REJECTED — not enough consistent data to solve X. Not saving.")
        print("     Re-record with BOTH tags visible (see below).")
        return 1
    frac = 100 * rms / max(mag, 1e-9)
    ang = math.degrees(math.acos(np.clip((np.trace(X) - 1) / 2, -1, 1)))
    print(f"\n  X = rotation of {ang:.1f} deg")
    for r in np.round(X, 4):
        print("     ", r)
    print(f"  residual {rms:.2f} deg on {mag:.1f} deg rotations -> {frac:.0f}% of signal")

    n_in = int(inl.sum())
    if frac > 15.0 or n_in < 40:
        print("\n  REJECTED — this fit is not trustworthy. Not saving.")
        print(f"     {n_in} consensus inliers, residual {frac:.0f}% of signal "
              f"(want <15% and >=40)")
        print("     Cause, almost always: only ONE tag was visible. A lone tag is planar")
        print("     and two-fold ambiguous, so the tracker can sit on the MIRRORED")
        print("     solution for a whole stretch -- self-consistent, but wrong. Two tags")
        print("     make the point set non-planar and the pose unique.")
        print("     Re-record holding the controller so BOTH tags stay in view.")
        print("     Then re-solve offline (no waving) with:  pad_imu.py align --replay")
        return 1
    print(f"  {n_in} consensus inliers — accepted")
    save_calib(X, bias, lag, rms, n_in)
    return 0


def cmd_replay(args):
    """Re-solve X from the saved recording — no camera, no waving."""
    if not RAW.exists():
        sys.exit(f"no recording at {RAW} — run `align` first")
    d = np.load(RAW)
    hist = list(zip(d["t_imu"], d["w_imu"], d["a_imu"]))
    cal = load_calib()
    bias = cal["bias"] if cal else np.zeros(3)
    print(f"replaying {len(d['t'])} vision poses / {len(hist)} imu samples")
    return solve_and_report(list(d["R"]), list(d["t"]), hist, bias, args)


def cmd_check(args):
    """Is the tag<->IMU rotation right? Static test, gravity as the reference.

    Deliberately has nothing to do with timing, motion or delay. Hold the controller
    STILL in a series of different orientations. Gravity never moves, so if X is correct
    our reconstruction of 'down' must land in the same place from every pose. If X is
    wrong, it swings around as you turn the controller -- and the size of that swing is
    the size of the calibration error, in degrees."""
    import cv2
    import rerun as rr
    from tag_pose import make_detector, detect, open_source
    from tag_body import load_model, body_pose

    cal = load_calib()
    X_saved = cal["X"] if cal else None
    imu = PadIMU(bias=cal["bias"] if cal else None)
    imu.start()
    time.sleep(0.4)
    if cal is None:
        print("no saved calibration — this run can still SOLVE one from gravity")

    model = load_model(args.model)
    det = make_detector("DICT_4X4_50")
    keep = set(model["tags"])
    src = open_source(args.source, args.fov)
    rr.init("pad_imu_check", spawn=True)
    rr.log("check/error_deg", rr.SeriesLines(colors=[220, 60, 60], names="gravity error"),
           static=True)

    print("\nHOLD THE CONTROLLER STILL, then turn it to a NEW orientation and hold again.")
    print(f"Both tags in view. {args.poses} distinct poses; Ctrl-C when you have enough.\n")
    poses_R, poses_a, guess = [], [], None
    g_live = None                       # running gravity estimate, for the 1-tag tie-break
    still_R, still_a, t_still = [], [], None
    t0 = time.perf_counter()
    try:
        while len(poses_R) < args.poses:
            frame, K = src.grab()
            if frame is None:
                continue
            t = time.perf_counter()
            rr.set_time("time", duration=t - t0)
            _, w, a = imu.latest()
            spin = np.degrees(np.linalg.norm(w))
            dets = detect(det, frame, keep)
            out = (pose_with_gravity(dets, K, model, a, X_saved, g_live, guess)
                   if len(dets) >= args.min_tags else None)
            ok = out is not None and out[2] <= 4.0
            if ok:
                guess = (out[0], out[1])

            if ok and X_saved is not None:
                v = out[0] @ X_saved @ a
                v = v / np.linalg.norm(v)
                g_live = v if g_live is None else 0.98 * g_live + 0.02 * v
            if ok and spin < args.still_dps:
                if t_still is None:
                    t_still, still_R, still_a = t, [], []
                still_R.append(out[0])
                still_a.append(a)
            else:
                t_still = None

            live = float("nan")
            if X_saved is not None and ok:
                pred = out[0] @ X_saved @ a
                pred /= np.linalg.norm(pred)
                if poses_R:
                    ref = np.mean([R @ X_saved @ aa / np.linalg.norm(R @ X_saved @ aa)
                                   for R, aa in zip(poses_R, poses_a)], axis=0)
                    ref /= np.linalg.norm(ref)
                    live = math.degrees(math.acos(np.clip(float(pred @ ref), -1, 1)))
                    rr.log("check/error_deg", rr.Scalars(live))

            if t_still is not None and t - t_still > args.hold and len(still_R) > 5:
                R_m = still_R[len(still_R) // 2]
                a_m = np.mean(still_a, axis=0)
                new = all(math.degrees(math.acos(np.clip(
                    (np.trace(R_m.T @ P) - 1) / 2, -1, 1))) > args.min_sep for P in poses_R)
                if new:
                    poses_R.append(R_m)
                    poses_a.append(a_m)
                    print(f"\n  pose {len(poses_R)}/{args.poses} banked"
                          + (f"   (gravity error here: {live:.1f} deg)" if live == live else ""))
                t_still = None

            print(f"\r  tags {len(dets)} | spin {spin:5.1f} deg/s "
                  f"{'STILL' if t_still is not None else '     '} | poses {len(poses_R)}"
                  + (f" | error {live:5.1f} deg" if live == live else "           "),
                  end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        src.stop()
        imu.stop()

    if len(poses_R) < 4:
        sys.exit(f"\nonly {len(poses_R)} poses — need at least 4 in different orientations")
    POSES.parent.mkdir(parents=True, exist_ok=True)
    # APPEND across sessions. AprilTag pose error of a few degrees is irreducible per
    # pose, so the only lever is N -- and there is no reason a session has to start from
    # scratch. Bank a dozen poses whenever you pass the desk; --fresh starts over.
    if POSES.exists() and not args.fresh:
        prev = np.load(POSES)
        poses_R = list(prev["R"]) + poses_R
        poses_a = list(prev["a"]) + poses_a
        print(f"\n  + {len(prev['R'])} poses from previous sessions = {len(poses_R)} total")
    np.savez_compressed(POSES, R=np.array(poses_R), a=np.array(poses_a))
    print(f"  poses saved -> {POSES}  (re-solve/adopt later with `check --replay --save`)")

    print(f"\n\n=== {len(poses_R)} static poses ===")
    cov, spread = gravity_coverage(poses_a)
    print(f"  coverage {cov:.3f}, poses spread over {spread:.0f} deg  "
          + ("(good — X is pinned)" if cov > 0.05 else
             "(POOR — the poses face too similar a way; scatter below is meaningless, "
             "turn the controller through much bigger angles)"))
    if X_saved is not None:
        rms_saved, _ = gravity_scatter(poses_R, poses_a, X_saved)
        print(f"  SAVED X:   gravity scatter {rms_saved:5.2f} deg")
    X_g, g_cam, rms_g = solve_from_gravity(poses_R, poses_a)
    sig = gravity_uncertainty(poses_R, poses_a)
    print(f"  GRAVITY-SOLVED X: per-pose scatter {rms_g:5.2f} deg  <- AprilTag noise, "
          f"irreducible")
    print(f"                    X uncertainty   {sig:5.2f} deg  <- THIS is the calibration"
          f" quality")
    print(f"  {len(poses_R)} poses. Roughly halving that uncertainty needs ~4x the poses;"
          f" just run `check` again, it appends.")
    if X_saved is not None:
        disagree = math.degrees(math.acos(np.clip((np.trace(X_saved.T @ X_g) - 1) / 2, -1, 1)))
        print(f"\n  the two disagree by {disagree:.1f} deg")
        print("  -> " + ("the saved X is fine" if disagree < 5 else
                         "the saved X is WRONG; gravity says so from still poses alone"))
    print("\n  X from gravity:")
    for r in np.round(X_g, 4):
        print("     ", r)
    if args.save:
        save_calib(X_g, cal["bias"] if cal else np.zeros(3),
                   cal["lag"] if cal else 0.0, rms_g, len(poses_R))
    else:
        print("\n  (re-run with --save to adopt the gravity-solved X)")
    return 0


def cmd_check_replay(args):
    if not POSES.exists():
        sys.exit(f"no banked poses at {POSES} — run `check` first")
    d = np.load(POSES)
    Rs, As = list(d["R"]), list(d["a"])
    cal = load_calib()
    cov, spread = gravity_coverage(As)
    print(f"{len(Rs)} banked static poses; coverage {cov:.3f} over {spread:.0f} deg"
          + ("" if cov > 0.05 else "  (POOR — results below are not meaningful)"))
    if cal is not None:
        print(f"  SAVED X:          gravity scatter {gravity_scatter(Rs, As, cal['X'])[0]:5.2f} deg")
    X_g, _, rms = solve_from_gravity(Rs, As)
    print(f"  GRAVITY-SOLVED X: per-pose scatter {rms:5.2f} deg")
    print(f"                    X uncertainty   {gravity_uncertainty(Rs, As):5.2f} deg")
    if cal is not None:
        dis = math.degrees(math.acos(np.clip((np.trace(cal["X"].T @ X_g) - 1) / 2, -1, 1)))
        print(f"  disagreement: {dis:.1f} deg")
    if args.save:
        save_calib(X_g, cal["bias"] if cal else np.zeros(3),
                   cal["lag"] if cal else 0.0, rms, len(Rs))
    else:
        print("  (add --save to adopt the gravity-solved X)")
    return 0


def cmd_selftest(args):
    """Solver check on synthetic data — no hardware, no camera."""
    import cv2
    rng = np.random.default_rng(0)
    ok = True
    for trial, true_rv in enumerate([np.array([0.3, -1.2, 0.7]),
                                     np.array([np.pi, 0, 0]),
                                     np.array([0.05, 0.02, -0.01])]):
        X_true = cv2.Rodrigues(true_rv)[0]
        w_imu = rng.normal(0, 1.2, (600, 3))
        w_body = (X_true @ w_imu.T).T + rng.normal(0, 0.02, (600, 3))   # noisy vision
        X = solve_alignment(w_imu, w_body)
        err = math.degrees(math.acos(np.clip((np.trace(X.T @ X_true) - 1) / 2, -1, 1)))
        good = err < 1.0
        ok &= good
        print(f"  trial {trial}: recovered to {err:.3f} deg  {'OK' if good else 'FAIL'}")

    # the time-shift finder must recover a known lag
    t_i = np.arange(0, 12, 1 / 200)
    sig = np.sin(2 * np.pi * 0.7 * t_i) + 0.4 * np.sin(2 * np.pi * 1.9 * t_i)
    w_i = np.column_stack([sig, 0.3 * sig, -0.5 * sig])
    # Contract: the returned lag L is what you ADD to vision timestamps to sample the
    # IMU, i.e. w_v(t) should equal w_imu(t + L). Build the data that way round.
    true_lag = -0.045
    t_v = np.arange(0.3, 11.5, 1 / 13)
    w_v = np.column_stack([np.interp(t_v + true_lag, t_i, w_i[:, k]) for k in range(3)])
    lag, corr = best_time_shift(t_i, w_i, t_v, w_v)
    good = abs(lag - true_lag) < 0.006
    ok &= good
    print(f"  time shift: found {lag * 1e3:+.0f} ms, true {true_lag * 1e3:+.0f} ms "
          f"(corr {corr:.3f})  {'OK' if good else 'FAIL'}")
    # A single visible tag is planar and two-fold ambiguous, so vision poses DO flip.
    # Ungated, 5% flipped frames drag the fit 170 deg off (measured 2026-08-20, and the
    # cause of the first real align failing at 114.8 deg). Consensus must survive it.
    X_true = cv2.Rodrigues(np.array([0.4, -1.1, 0.6]))[0]
    t_i = np.arange(0, 40, 1 / 200)
    w_true = np.column_stack([1.4 * np.sin(2 * np.pi * .31 * t_i),
                              1.1 * np.sin(2 * np.pi * .23 * t_i + 1),
                              0.9 * np.sin(2 * np.pi * .17 * t_i + 2)])
    w_i = (X_true.T @ w_true.T).T + rng.normal(0, 0.004, w_true.shape)
    t_v = np.arange(0.5, 39.0, 1 / 20)
    R, Rs, tt = np.eye(3), [], 0.0
    for tv_ in t_v:
        while tt < tv_:
            R = R @ cv2.Rodrigues(np.array([np.interp(tt, t_i, w_true[:, k])
                                            for k in range(3)]) / 200)[0]
            tt += 1 / 200
        Rs.append(R.copy())
    # realistic pose noise: ~0.5 deg, which finite differencing would turn into 10 deg/s
    noisy = [r @ cv2.Rodrigues(rng.normal(0, np.radians(0.5), 3))[0] for r in Rs]
    flip = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
    for frac, lbl in ((0.0, "clean"), (0.10, "10% flips")):
        Rn = [r @ (flip if rng.random() < frac else np.eye(3)) for r in noisy]
        Xn, lag, rms, mag, inl, _ = fit_alignment(t_i, w_i, Rn, list(t_v), verbose=False)
        e = 999.0 if Xn is None else math.degrees(
            math.acos(np.clip((np.trace(Xn.T @ X_true) - 1) / 2, -1, 1)))
        good = e < 3.0
        ok &= good
        print(f"  {lbl:>10}: recovered to {e:6.2f} deg, residual "
              f"{100 * rms / max(mag, 1e-9):4.0f}% of signal, {int(inl.sum()):3d} inliers"
              f"  {'OK' if good else 'FAIL'}")

    print("\nselftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def _triad(path, R, p, length, colors, labels, radii=None):
    """Draw a rotation as three arrows from p. R's COLUMNS are the body axes in camera
    coordinates, so R.T's rows are what Arrows3D wants."""
    import rerun as rr
    rr.log(path, rr.Arrows3D(origins=np.tile(np.asarray(p, float), (3, 1)),
                             vectors=R.T * length, colors=colors, labels=labels,
                             radii=radii))


def cmd_view(args):
    """Live 3-D: the TAG-measured body frame vs the GYRO-propagated one, side by side.

    The question this answers is not "is X roughly right" (that is `check`) but "do the
    two sensors still agree as you MOVE, and does their disagreement stay put or grow?"

    Method: anchor the gyro orientation to the vision orientation once, then let the gyro
    run free -- R_imu <- R_imu @ exp([X w]x dt) -- and plot the residual rotation between
    the two. A CONSTANT offset means X is slightly off but stable, which is harmless and
    calibratable. A GROWING offset means gyro bias (or scale), and it is what silently
    poisons a tag dropout bridge. The distinction is the whole point, so the summary
    separates drift per second of TIME from drift per degree of ROTATION."""
    import cv2
    import rerun as rr
    import rerun.blueprint as rrb
    from tag_pose import make_detector, detect, open_source
    from tag_body import load_model, body_pose

    cal = load_calib()
    if cal is None:
        sys.exit("no saved calibration -- run `pad_imu.py align` (or `check --save`) first")
    X, bias_saved = cal["X"], cal["bias"]
    imu = PadIMU(bias=None)                 # bias applied below, after we decide which
    imu.start()
    time.sleep(0.4)

    # Gyro bias is NOT a one-time calibration -- it moves with temperature and across
    # power cycles, and the pad sleeps between sessions. A stale bias of even 1 deg/s
    # is 60 deg of drift per minute, which looks exactly like a bad X while X is fine.
    # So measure it fresh and SHOW the difference: that difference is the diagnosis.
    bias = bias_saved
    if args.bias_seconds > 0:
        print(f"\n  hold the controller DEAD STILL for {args.bias_seconds:.0f} s "
              "(measuring gyro bias)...")
        fresh, sd = imu.measure_bias(args.bias_seconds)
        d = np.degrees(fresh - bias_saved)
        moved = float(np.degrees(np.linalg.norm(sd)))
        print(f"  saved bias {np.round(np.degrees(bias_saved), 2)} deg/s")
        print(f"  fresh bias {np.round(np.degrees(fresh), 2)} deg/s   "
              f"(noise {moved:.2f} deg/s)")
        print(f"  DIFFERENCE {np.round(d, 2)} deg/s  (|d| = {np.linalg.norm(d):.2f})")
        if moved > 1.0:
            # A moving pad turns real rotation into "bias" and poisons everything after.
            print("  !! the pad was NOT still while measuring — that reading is not a "
                  "bias. Keeping the saved one; re-run and hold it down on the desk.")
            fresh = bias_saved
        elif np.linalg.norm(d) > 0.5:
            print("  -> the SAVED bias is stale. That alone would drift "
                  f"{np.linalg.norm(d) * 60:.0f} deg/min; using the fresh one.")
        bias = bias_saved if args.saved_bias else fresh
    with imu._lock:
        imu.bias = np.asarray(bias, float)

    run = LOG_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run.mkdir(parents=True, exist_ok=True)
    fcsv = open(run / "view.csv", "w", newline="")
    wcsv = csv.writer(fcsv)
    wcsv.writerow(["time_s", "tags", "reproj_px", "spin_dps", "rot_cum_deg",
                   "disagree_deg", "err_x", "err_y", "err_z",
                   "wx", "wy", "wz", "ax", "ay", "az"])
    print(f"  logging to {run}")

    model = load_model(args.model)
    det = make_detector("DICT_4X4_50")
    keep = set(model["tags"])
    src = open_source(args.source, args.fov)

    rr.init("pad_imu_view", spawn=True)
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(origin="world", name="tag (RGB) vs gyro (orange/cyan/magenta)"),
        rrb.Vertical(
            rrb.Spatial2DView(origin="cam", name="camera"),
            rrb.TimeSeriesView(origin="d", name="disagreement (deg)"),
            rrb.TimeSeriesView(origin="axis", name="which axis drifts (deg)"),
        ), column_shares=[3, 2])))
    for nm, col in (("d/total", [235, 90, 60]), ("d/spin_dps", [130, 130, 140]),
                    ("axis/x", [235, 90, 60]), ("axis/y", [60, 190, 100]),
                    ("axis/z", [80, 150, 250])):
        rr.log(nm, rr.SeriesLines(colors=col, names=nm.split("/")[-1]), static=True)

    # Both triads use the SAME colour per axis and the SAME length, so when the sensors
    # agree the arrows sit exactly on top of each other and you see THREE arrows, not
    # six. Any split you can see IS the disagreement. Vision is drawn thick, gyro thin.
    VIS_C = [[235, 60, 60], [60, 200, 90], [70, 140, 255]]
    IMU_C = [[255, 120, 120], [130, 230, 160], [140, 190, 255]]   # same hues, lighter

    print("\nMove the controller around, keeping the tag in view most of the time.")
    print("THICK arrows = camera, THIN arrows = gyro, same colour per axis.")
    print("If they agree you see 3 arrows. If they split, that gap IS the error.")
    print("Ctrl-C for the drift summary.\n")

    R_imu = None
    guess = None
    t_prev = t_anchor = None
    t0 = time.perf_counter()
    t_img = 0.0
    rot_cum = 0.0                       # total rotation travelled, deg
    hist = []                           # (elapsed, rot_cum, disagreement deg)
    n_fps, t_fps, hz = 0, time.perf_counter(), 0.0
    try:
        while True:
            frame, K = src.grab()
            if frame is None:
                continue
            t = time.perf_counter()
            rr.set_time("time", duration=t - t0)
            _, w, a = imu.latest()
            dt = 0.0 if t_prev is None else min(t - t_prev, 0.25)
            t_prev = t

            # propagate the gyro orientation in the BODY frame (right-multiply)
            w_body = X @ w
            if R_imu is not None and dt > 0:
                dR, _ = cv2.Rodrigues(w_body * dt)
                R_imu = R_imu @ dR
                rot_cum += math.degrees(np.linalg.norm(w_body) * dt)

            dets = detect(det, frame, keep)
            out = body_pose(dets, K, model, guess) if len(dets) >= args.min_tags else None
            ok = out is not None and out[2] <= 4.0
            if ok:
                guess = (out[0], out[1])
                R_vis, p_vis = out[0], out[1]
                if R_imu is None or (args.reanchor and t - t_anchor >= args.reanchor):
                    R_imu, t_anchor = R_vis.copy(), t
                    print(f"\n  anchored gyro to vision at t={t - t0:.1f}s")

            n_fps += 1
            if t - t_fps >= 1.0:
                hz, n_fps, t_fps = n_fps / (t - t_fps), 0, t
                rr.log("d/loop_hz", rr.Scalars(hz))
            spin = math.degrees(np.linalg.norm(w))
            rr.log("d/spin_dps", rr.Scalars(spin))
            # no tag-count series on purpose: the disagreement trace simply GAPS when
            # vision drops, which says the same thing without a second scale on the plot

            dis = float("nan")
            if ok and R_imu is not None:
                p = p_vis
                _triad("world/vision", R_vis, p, 0.060, VIS_C, ["x", "y", "z"],
                       radii=0.0018)
                _triad("world/gyro", R_imu, p, 0.060, IMU_C, [None, None, None],
                       radii=0.0007)
                # gravity as the IMU reconstructs it, in camera coords
                g = R_vis @ X @ a
                n = np.linalg.norm(g)
                if n > 1e-6:
                    rr.log("world/gravity", rr.Arrows3D(
                        origins=[p], vectors=[g / n * 0.05], colors=[[245, 225, 70]],
                        labels=["g (from IMU)"]))
                r, _ = cv2.Rodrigues(R_vis.T @ R_imu)       # residual, in the body frame
                r = np.degrees(r.ravel())
                dis = float(np.linalg.norm(r))
                rr.log("d/total", rr.Scalars(dis))
                for nm, v in zip("xyz", r):
                    rr.log(f"axis/{nm}", rr.Scalars(float(v)))
                hist.append((t - t0, rot_cum, dis))
            wcsv.writerow([f"{t - t0:.4f}", len(dets),
                           f"{out[2]:.2f}" if ok else "",
                           f"{spin:.3f}", f"{rot_cum:.3f}",
                           f"{dis:.4f}" if dis == dis else "",
                           *([f"{v:.4f}" for v in r] if dis == dis else ["", "", ""]),
                           *[f"{v:.6f}" for v in w], *[f"{v:.4f}" for v in a]])

            # Small + jpeg, sent often, beats big + raw sent rarely. Measured on this
            # machine: raw 1280x720 costs 10.1 ms to send and 83 MB/s of viewer input,
            # so it was throttled to 2 fps and LOOKED laggy even though detection ran at
            # 30 Hz. Half size + jpeg costs 1.76 ms, so it can run 10x more often.
            if not args.no_image and t - t_img >= 1.0 / args.image_hz:
                t_img = t
                small = frame if args.rr_scale >= 0.999 else cv2.resize(
                    frame, None, fx=args.rr_scale, fy=args.rr_scale)
                rr.log("cam/image", rr.Image(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
                                      .compress(jpeg_quality=70))

            print(f"\r  {hz:4.1f} Hz | tags {len(dets)} | spin {spin:5.1f} deg/s | "
                  f"rotated {rot_cum:6.0f} deg | disagreement "
                  + (f"{dis:5.2f} deg" if dis == dis else "  --  ") + "   ", end="")
    except KeyboardInterrupt:
        print("\n")
    finally:
        imu.stop()
        fcsv.close()
        try:
            src.close()
        except Exception:
            pass

    if len(hist) < 20:
        print("not enough paired samples for a drift summary")
        return
    H = np.array(hist)
    el, rot, dis = H[:, 0], H[:, 1], H[:, 2]
    # Separate the two failure modes: a constant misalignment vs an accumulating one.
    A_t = np.polyfit(el, dis, 1)
    A_r = np.polyfit(rot, dis, 1) if np.ptp(rot) > 30 else (float("nan"), float("nan"))
    print(f"  paired samples      : {len(hist)} over {el[-1]:.0f} s, "
          f"{rot[-1]:.0f} deg of rotation travelled")
    print(f"  disagreement        : mean {dis.mean():.2f} deg, "
          f"median {np.median(dis):.2f}, p90 {np.percentile(dis, 90):.2f}, "
          f"max {dis.max():.2f}")
    print(f"  vs TIME             : {A_t[0]:+.3f} deg/s   (offset {A_t[1]:.2f} deg)")
    if A_r[0] == A_r[0]:
        print(f"  vs ROTATION         : {A_r[0]:+.4f} deg per deg turned "
              f"({100 * A_r[0]:+.2f}% scale error)")
    drifting = abs(A_t[0]) > 0.05
    print("\n  " + ("DRIFTING -- the offset grows, so this is gyro bias/scale, not a "
                    "fixed misalignment. Re-run `bias` with the pad dead still."
                    if drifting else
                    "STABLE -- the offset stays put, so it is a fixed misalignment in X "
                    "(harmless for bridging; shrink it with more `check` poses)."))
    if np.median(dis) > 8:
        print("  NOTE: the offset itself is large; X is probably still coarse.")


SOLO_DEAD_DEG = 5.0                # tilt inside this is "neutral" -- hand tremor, not intent
SOLO_FULL_DEG = 30.0               # tilt at which the rate command saturates
SOLO_MAX_MMPS = 200.0              # mm/s at full tilt
SOLO_G_TAU = 1.0                   # s: how hard gravity pulls the attitude back to level


def cmd_solo(args):
    """IMU ONLY -- no camera, no tags. Compare the two ways to turn a pad into a command.

    Measured on this pad (2026-08-21, stationary so true displacement is zero), position
    by double-integrating acceleration drifts 8.6 mm @0.5 s, 35 mm @1 s, 145 mm @2 s,
    330 mm @3 s. It is quadratic because the error is 0.5*g*sin(theta)*t^2 -- gravity
    leaking through attitude error -- so no amount of clutching rescues a 2 s stroke.

    ORIENTATION has no such problem: 0.10 deg/s of gyro noise, and roll/pitch are pinned
    by gravity, which never drifts. So this view shows BOTH, honestly:

      * INTEGRATED position (blue trail) -- watch it run away, that is the 8.6 mm/35 mm/145 mm
      * TILT-TO-RATE (orange trail) -- tilt = velocity, no acceleration integration at all

    Auto-ZUPT stands in for the clutch: hold the pad still and both reset."""
    import cv2
    import rerun as rr
    import rerun.blueprint as rrb

    cal = load_calib()
    bias = cal["bias"] if cal else None
    imu = PadIMU(bias=bias)
    imu.start()
    time.sleep(0.4)
    print(f"\n  hold the controller DEAD STILL for {args.bias_seconds:.0f} s "
          "(measuring gyro bias)...")
    b, sd = imu.measure_bias(args.bias_seconds)
    print(f"  bias {np.round(np.degrees(b), 2)} deg/s   "
          f"noise {np.degrees(np.linalg.norm(sd)):.2f} deg/s")

    run = LOG_ROOT / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_solo")
    run.mkdir(parents=True, exist_ok=True)
    fcsv = open(run / "solo.csv", "w", newline="")
    wcsv = csv.writer(fcsv)
    wcsv.writerow(["time_s", "hz", "still", "pitch_deg", "roll_deg", "yaw_deg",
                   "vx", "vy", "vz", "rate_x", "rate_y", "rate_z",
                   "int_x", "int_y", "int_z", "tilt_x", "tilt_y", "tilt_z"])
    print(f"  logging to {run}")

    rr.init("pad_imu_solo", spawn=True)
    rr.send_blueprint(rrb.Blueprint(rrb.Horizontal(
        rrb.Spatial3DView(origin="world", name="orientation + the two trails"),
        rrb.Vertical(
            rrb.TimeSeriesView(origin="tilt", name="tilt (deg)"),
            rrb.TimeSeriesView(origin="cmd", name="tilt-to-rate command (mm/s)"),
            rrb.TimeSeriesView(origin="drift", name="integrated-position drift (mm)"),
        ), column_shares=[3, 2])))
    for nm, col in (("tilt/pitch", [235, 90, 60]), ("tilt/roll", [60, 190, 100]),
                    ("tilt/yaw", [80, 150, 250]),
                    ("cmd/vx", [235, 90, 60]), ("cmd/vy", [60, 190, 100]),
                    ("cmd/vz", [80, 150, 250]),
                    ("drift/integrated", [80, 150, 250]),
                    ("drift/spin_dps", [140, 140, 150])):
        rr.log(nm, rr.SeriesLines(colors=col, names=nm.split("/")[-1]), static=True)

    # attitude: body -> world, seeded level from the first gravity reading
    _, _, a0 = imu.latest()
    a0 = a0 / np.linalg.norm(a0)
    axis = np.cross(a0, [0.0, 0.0, 1.0])
    sn = np.linalg.norm(axis)
    R = (cv2.Rodrigues(axis / sn * math.atan2(sn, float(a0 @ [0.0, 0.0, 1.0])))[0]
         if sn > 1e-8 else np.eye(3))

    v = np.zeros(3)          # velocity from double integration (the doomed one)
    p_int = np.zeros(3)      # position from double integration
    p_tilt = np.zeros(3)     # position from tilt-to-rate (the honest one)
    trail_i, trail_t = [], []
    t0 = tprev = time.perf_counter()
    t_still = None
    n, t_fps, hz = 0, t0, 0.0
    print("\n  TILT the controller to drive the orange trail. Hold it STILL to reset both.")
    print("  Blue trail = double-integrated position. Watch it leave.\n")
    try:
        while True:
            t = time.perf_counter()
            dt = t - tprev
            if dt < 1.0 / args.rate:
                time.sleep(0.001)
                continue
            tprev = t
            n += 1
            if t - t_fps >= 0.5:
                hz = n / (t - t_fps); n, t_fps = 0, t
            rr.set_time("time", duration=t - t0)
            _, w, a = imu.latest()

            # ---- attitude: gyro integrates, gravity corrects roll/pitch only ----
            R = R @ cv2.Rodrigues(w * dt)[0]
            an = np.linalg.norm(a)
            if an > 1e-6:
                meas = a / an                       # gravity direction, body frame
                pred = R.T @ np.array([0.0, 0.0, 1.0])
                err = np.cross(meas, pred)          # rotation that levels the estimate
                R = R @ cv2.Rodrigues(err * (dt / SOLO_G_TAU))[0]

            still = (math.degrees(np.linalg.norm(w)) < 2.0 and abs(an - G) < 0.35)
            t_still = t_still if (still and t_still is not None) else (t if still else None)

            # ---- (1) the doomed one: double-integrate acceleration ----
            a_world = R @ a - np.array([0.0, 0.0, G])
            if still and t_still is not None and t - t_still > 0.4:
                v[:] = 0.0                          # ZUPT stands in for a clutch
                p_int[:] = 0.0
                p_tilt[:] = 0.0
                trail_i.clear(); trail_t.clear()
            else:
                v = v + a_world * dt
                p_int = p_int + v * dt

            # ---- (2) the honest one: tilt IS the velocity ----
            fwd, lft = R[:, 0], R[:, 1]             # controller axes, in world
            pitch = math.degrees(math.asin(float(np.clip(fwd[2], -1, 1))))
            roll = math.degrees(math.asin(float(np.clip(lft[2], -1, 1))))
            yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))

            def rate(deg):
                m = max(0.0, abs(deg) - SOLO_DEAD_DEG) / (SOLO_FULL_DEG - SOLO_DEAD_DEG)
                return math.copysign(min(m, 1.0) ** 2 * SOLO_MAX_MMPS, deg)

            cmd = np.array([rate(-pitch), rate(-roll), 0.0])   # z left for a stick/button
            p_tilt = p_tilt + cmd * dt * 1e-3

            trail_i.append(p_int.copy()); trail_t.append(p_tilt.copy())
            trail_i[:] = trail_i[-400:]; trail_t[:] = trail_t[-400:]

            rr.log("world/pad", rr.Arrows3D(
                origins=np.tile(p_tilt, (3, 1)), vectors=R.T * 0.05,
                colors=[[235, 60, 60], [60, 200, 90], [70, 140, 255]],
                labels=["fwd", "left", "up"]))
            if len(trail_i) > 1:
                rr.log("world/integrated", rr.LineStrips3D(
                    [np.array(trail_i)], colors=[[80, 150, 250]]))
                rr.log("world/tilt_rate", rr.LineStrips3D(
                    [np.array(trail_t)], colors=[[255, 165, 30]]))
            for nm, val in (("tilt/pitch", pitch), ("tilt/roll", roll), ("tilt/yaw", yaw),
                            ("cmd/vx", cmd[0]), ("cmd/vy", cmd[1]), ("cmd/vz", cmd[2]),
                            ("drift/integrated", float(np.linalg.norm(p_int)) * 1e3),
                            ("drift/spin_dps", math.degrees(np.linalg.norm(w)))):
                rr.log(nm, rr.Scalars(float(val)))

            wcsv.writerow([f"{t - t0:.4f}", f"{hz:.1f}", int(still),
                           f"{pitch:.2f}", f"{roll:.2f}", f"{yaw:.2f}",
                           *[f"{x:.4f}" for x in v], *[f"{x:.1f}" for x in cmd],
                           *[f"{x:.4f}" for x in p_int], *[f"{x:.4f}" for x in p_tilt]])
            print(f"\r  {hz:5.1f} Hz | {'STILL' if still else 'moving':6s} | "
                  f"pitch {pitch:+6.1f} roll {roll:+6.1f} | cmd [{cmd[0]:+6.0f}"
                  f"{cmd[1]:+6.0f}] mm/s | integrated {np.linalg.norm(p_int) * 1e3:7.1f} mm  ",
                  end="")
    except KeyboardInterrupt:
        print("\n")
    finally:
        imu.stop()
        fcsv.close()
    print(f"  log: {run}")


# ── gyro scale + bias from gravity alone (no camera, no tag model, no X) ──────────

def still_segments(t, w, a, max_dps=12.0, g_tol=0.35, min_s=0.25):
    """Index spans where the pad is quasi-static: slow AND |a| ~ g (so `a` IS gravity)."""
    ok = (np.linalg.norm(w, axis=1) < math.radians(max_dps)) & \
         (np.abs(np.linalg.norm(a, axis=1) - G) < g_tol)
    segs, i = [], 0
    while i < len(ok):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < len(ok) and ok[j]:
            j += 1
        if t[j - 1] - t[i] > min_s:
            segs.append((i, j))
        i = j
    return segs


def merge_segments(t, a, segs, same_deg=8.0):
    """Join rests that were split apart.

    One long rest can dip below the stillness test for a few samples (a knock on the
    table, sensor noise) and come back as two or three segments. Those pieces are the
    SAME pose, so the "rotation" between them is ~0 deg and they were being thrown away
    -- which is how a good 60 s recording produced 13 rests and 0 usable pairs."""
    out = []
    for s0, s1 in segs:
        g = a[s0:s1].mean(0); g /= np.linalg.norm(g)
        if out:
            p0, p1, pg = out[-1]
            if math.degrees(math.acos(np.clip(pg @ g, -1, 1))) < same_deg:
                out[-1] = (p0, s1, (pg + g) / np.linalg.norm(pg + g))
                continue
        out.append((s0, s1, g))
    return out


def gravity_transitions(t, w, a, segs, lo_deg=15.0, hi_deg=150.0, max_gap=8.0,
                        report=False):
    """Consecutive still poses far enough apart to measure a rotation between them."""
    merged = merge_segments(t, a, segs)
    out, why = [], Counter()
    for (a0, a1, g1), (b0, b1, g2) in zip(merged[:-1], merged[1:]):
        sep = math.degrees(math.acos(np.clip(g1 @ g2, -1, 1)))
        gap = t[b0] - t[a1]
        if sep <= lo_deg:
            why["turned too little (<15 deg)"] += 1
        elif sep >= hi_deg:
            why["turned too far (>150 deg)"] += 1
        elif gap >= max_gap:
            why[f"move took too long (>{max_gap:.0f} s)"] += 1
        else:
            out.append((a1, b0, g1, g2, sep))
    if report:
        print(f"  {len(segs)} rests -> {len(merged)} after joining split ones")
        for k, v in why.items():
            print(f"    dropped {v}: {k}")
    return out


def fit_gyro(t, w_raw, pairs):
    """Solve gyro scale + bias so integrating between still poses predicts gravity.

    Gravity is a known direction the accelerometer reads directly, so this needs no
    camera and no X. Scale and bias must be solved TOGETHER: a bias is a fixed rate and
    a scale is multiplicative, and fitting either alone absorbs part of the other
    (measured: scale alone 10.77 deg residual, both 1.72 deg)."""
    import cv2
    from scipy.optimize import least_squares

    def integ(lo, hi, scale, b):
        R = np.eye(3)
        for k in range(lo, hi):
            dt = t[k + 1] - t[k]
            if 0 < dt < 0.1:
                R = R @ cv2.Rodrigues((w_raw[k] - b) * scale * dt)[0]
        return R

    def resid(p):
        return np.concatenate([integ(lo, hi, p[0], p[1:4]).T @ g1 - g2
                               for lo, hi, g1, g2, _ in pairs])

    r = least_squares(resid, [1.0, 0.0, 0.0, 0.0], method="lm")
    per = [math.degrees(2 * math.asin(min(1.0, np.linalg.norm(v) / 2)))
           for v in resid(r.x).reshape(-1, 3)]
    return float(r.x[0]), r.x[1:4], float(np.median(per))


def cmd_gyrocal(args):
    """Measure the gyro's scale and bias against gravity. No camera needed."""
    imu = PadIMU(gyr_scale=1.0)             # solve the ABSOLUTE scale, not a residual one
    imu.start()
    time.sleep(0.4)
    print(f"Driver reports {imu.gyr_per_dps:.0f} gyro units per deg/s.\n")
    print(f"Put the controller ON THE TABLE. For the next {args.seconds:.0f} s, every few")
    print("seconds tip it onto a DIFFERENT face and let it rest ~1.5 s before moving")
    print("again. Resting is what makes the accelerometer read pure gravity; the rests")
    print("are the measurement, the motion between them is what is being calibrated.\n")
    imu.keep_history = True
    imu.history()
    t0 = time.perf_counter()
    while (el := time.perf_counter() - t0) < args.seconds:
        time.sleep(0.25)
        print(f"\r  {el:4.1f}s  imu {imu.n:6d}   ", end="", flush=True)
    hist = imu.history()
    imu.stop()
    print()

    t = np.array([h[0] for h in hist])
    w = np.array([h[1] for h in hist])
    a = np.array([h[2] for h in hist])
    np.savez(GYRO_RAW, t=t, w=w, a=a)          # so a failed run can be examined
    segs = still_segments(t, w, a)
    pairs = gravity_transitions(t, w, a, segs, report=True)
    print(f"  -> {len(pairs)} usable transitions "
          f"({np.mean([p[4] for p in pairs]) if pairs else 0:.0f} deg mean)")
    print(f"  raw saved -> {GYRO_RAW}")
    if len(pairs) < 5:
        sys.exit("  not enough still->move->still cycles. Rest it LONGER between moves.")

    scale, bias, res = fit_gyro(t, w, pairs)
    def resid_at(sc, b):
        import cv2
        out = []
        for lo, hi, g1, g2, _ in pairs:
            R = np.eye(3)
            for k in range(lo, hi):
                dt = t[k+1] - t[k]
                if 0 < dt < 0.1:
                    R = R @ cv2.Rodrigues((w[k] - b) * sc * dt)[0]
            out.append(math.degrees(math.acos(np.clip((R.T @ g1) @ g2, -1, 1))))
        return float(np.median(out))
    print(f"\n  scale     {scale:.4f}  -> {imu.gyr_per_dps/scale:.0f} units per deg/s")
    print(f"  bias      {np.degrees(bias).round(2)} deg/s "
          f"(|b| {np.degrees(np.linalg.norm(bias)):.2f})")
    print(f"  residual  {res:.2f} deg   (uncorrected: {resid_at(1.0, np.zeros(3)):.2f} deg)")
    if res > 5.0:
        print("\n  REJECTED - residual too high to trust. Not saving.")
        print("     Usually: it never really came to rest between moves.")
        return 1
    d = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    d["gyr_scale"] = scale
    d["bias"] = list(map(float, bias))
    d["gyro_residual_deg"] = res
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps(d, indent=2))
    print(f"\n  saved scale+bias -> {CALIB}")
    if "X" in d and d["X"] is not None:
        print("  NOTE: the stored X was solved with the OLD gyro scale -- re-run `align`.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe").set_defaults(fn=cmd_probe)
    b = sub.add_parser("bias"); b.add_argument("--seconds", type=float, default=2.0)
    b.set_defaults(fn=cmd_bias)
    a = sub.add_parser("align")
    a.add_argument("--source", default="9")
    a.add_argument("--fov", type=float, default=70.0)
    a.add_argument("--model", default=str(BODY_MODEL),
                   help="tag body model (default: the 3-face box)")
    a.add_argument("--seconds", type=float, default=30.0)
    a.add_argument("--min-dps", type=float, default=25.0)
    a.add_argument("--min-tags", type=int, default=2,
                   help="tags required per frame (2 = unambiguous; 1 only as a fallback)")
    a.add_argument("--window", type=float, default=0.35,
                   help="seconds per rotation-increment pair")
    a.add_argument("--replay", action="store_true",
                   help="re-solve from the last recording instead of recording anew")
    a.add_argument("--no-rr", action="store_true", help="no live Rerun view")
    a.set_defaults(fn=lambda ar: cmd_replay(ar) if ar.replay else cmd_align(ar))
    c = sub.add_parser("check")
    c.add_argument("--source", default="9")
    c.add_argument("--fov", type=float, default=70.0)
    c.add_argument("--model", default=str(BODY_MODEL),
                   help="tag body model (default: the 3-face box)")
    c.add_argument("--poses", type=int, default=25)
    c.add_argument("--hold", type=float, default=0.6, help="seconds of stillness per pose")
    c.add_argument("--still-dps", type=float, default=4.0)
    c.add_argument("--min-sep", type=float, default=25.0, help="deg between banked poses")
    c.add_argument("--min-tags", type=int, default=2)
    c.add_argument("--save", action="store_true", help="adopt the gravity-solved X")
    c.add_argument("--fresh", action="store_true",
                   help="discard previously banked poses instead of adding to them")
    c.add_argument("--replay", action="store_true",
                   help="re-solve from the last banked poses instead of recording anew")
    c.set_defaults(fn=lambda ar: cmd_check_replay(ar) if ar.replay else cmd_check(ar))
    v = sub.add_parser("view")
    v.add_argument("--source", default="9")
    v.add_argument("--fov", type=float, default=70.0)
    v.add_argument("--model", default=str(BODY_MODEL),
                   help="tag body model (default: the 3-face box)")
    v.add_argument("--min-tags", type=int, default=1)
    v.add_argument("--reanchor", type=float, default=0.0,
                   help="re-anchor the gyro to vision every N s (0 = free-run, shows drift)")
    v.add_argument("--no-image", action="store_true")
    v.add_argument("--image-hz", type=float, default=15.0,
                   help="how often the camera picture goes to Rerun")
    v.add_argument("--rr-scale", type=float, default=0.5,
                   help="size of that picture (1.0 = full; detection always uses full)")
    v.add_argument("--bias-seconds", type=float, default=3.0,
                   help="re-measure gyro bias at startup (0 = trust the saved one)")
    v.add_argument("--saved-bias", action="store_true",
                   help="use the saved bias even if the fresh measurement disagrees")
    v.set_defaults(fn=cmd_view)
    so = sub.add_parser("solo")
    so.add_argument("--rate", type=float, default=100.0, help="loop rate cap (Hz)")
    so.add_argument("--bias-seconds", type=float, default=3.0)
    so.set_defaults(fn=cmd_solo)
    g = sub.add_parser("gyrocal")
    g.add_argument("--seconds", type=float, default=60.0)
    g.set_defaults(fn=cmd_gyrocal)
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
