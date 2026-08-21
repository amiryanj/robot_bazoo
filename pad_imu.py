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
import glob
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vision"))

CALIB = ROOT / "outputs/calib/pad_imu.json"
G = 9.80665


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


class PadIMU(threading.Thread):
    """Background reader. Keeps the newest sample and integrates body rotation.

    Scale factors come from the driver's own `resolution` fields rather than a constant,
    because that is where the factory calibration lands: accel in units/g, gyro in units
    per deg/s (Linux input event-codes convention for INPUT_PROP_ACCELEROMETER)."""

    def __init__(self, path=None, bias=None):
        super().__init__(daemon=True)
        import evdev
        from evdev import ecodes
        self.ecodes = ecodes
        path = path or find_imu()
        if path is None:
            connected = Path("/proc/bus/input/devices").read_text().count("Pro Controller")
            raise RuntimeError(
                "the controller is connected but its IMU node is not readable — this "
                "shell predates your 'input' group membership. Either log out and back "
                "in, or prefix the command:\n"
                '    sg input -c "$CONDA_PREFIX/bin/python pad_imu.py ..."'
                if connected else
                "no 'Pro Controller (IMU)' node — the controller is not connected. "
                "Press Home on it to wake the Bluetooth link.")
        self.dev = evdev.InputDevice(path)
        self.name = self.dev.name

        absinfo = dict(self.dev.capabilities(absinfo=True)).get(ecodes.EV_ABS, [])
        res = {c: i.resolution for c, i in absinfo}
        self.acc_per_g = float(res.get(ecodes.ABS_X) or 4096)
        self.gyr_per_dps = float(res.get(ecodes.ABS_RX) or 14247)
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
                w = np.radians(np.array(self._raw[3:], float) / self.gyr_per_dps)
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
                lag=float(d.get("lag", 0.0)), rms_dps=float(d.get("rms_dps", float("nan"))))


def save_calib(X, bias, lag, rms_dps, n):
    CALIB.parent.mkdir(parents=True, exist_ok=True)
    CALIB.write_text(json.dumps(dict(X=X.tolist(), bias=list(map(float, bias)),
                                     lag=lag, rms_dps=rms_dps, samples=int(n)), indent=2))
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
    print("Hold the controller STILL for 2 s (gyro bias)...")
    bias, _ = imu.measure_bias(2.0)
    print(f"  bias {np.round(np.degrees(bias), 2)} deg/s\n")

    model = load_model()
    det = make_detector("DICT_4X4_50")
    keep = set(model["tags"])
    src = open_source(args.source, args.fov)
    imu.keep_history = True
    imu.history()                                  # drop the still period

    print(f"Now ROTATE the controller in front of the camera for {args.seconds:.0f} s.")
    print("Twist it about ALL THREE axes -- roll, pitch, yaw.")
    print("KEEP BOTH TAGS IN VIEW: one tag alone is planar, so its pose is two-fold")
    print("ambiguous and the tracker can silently sit on the mirrored solution. Only")
    print("2-tag frames are recorded here -- the counter below is what matters.")
    print("Slow, deliberate turns (~40 deg/s) beat fast ones.\n")
    Rs, ts, guess = [], [], None
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
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
        el = time.perf_counter() - t0
        print(f"\r  {el:4.1f}s  BOTH-TAG poses {len(Rs):4d}  (tags now: {len(dets)})  "
              f"imu {imu.n:5d}   ", end="", flush=True)
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

    model = load_model()
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
    a.add_argument("--seconds", type=float, default=30.0)
    a.add_argument("--min-dps", type=float, default=25.0)
    a.add_argument("--min-tags", type=int, default=2,
                   help="tags required per frame (2 = unambiguous; 1 only as a fallback)")
    a.add_argument("--window", type=float, default=0.35,
                   help="seconds per rotation-increment pair")
    a.add_argument("--replay", action="store_true",
                   help="re-solve from the last recording instead of recording anew")
    a.set_defaults(fn=lambda ar: cmd_replay(ar) if ar.replay else cmd_align(ar))
    c = sub.add_parser("check")
    c.add_argument("--source", default="9")
    c.add_argument("--fov", type=float, default=70.0)
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
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    args = ap.parse_args()
    sys.exit(args.fn(args) or 0)


if __name__ == "__main__":
    main()
