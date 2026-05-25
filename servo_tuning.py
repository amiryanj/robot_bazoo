"""
servo_tuning.py — toolbox for characterising and tuning the SO-101's STS3215 servos.

This is a *library* (no CLI, no side effects on import). The CLI lives in calibrate.py.

What it gives you
-----------------
1. Register access (Lock-aware)
   - get_pid / set_pid : read & write the servo's internal P/I/D coefficients.
     set_pid keeps torque ON by toggling only the EEPROM Lock bit, and verifies
     every write by reading it back.
   - set_acceleration  : the servo's on-board trapezoidal motion-profiling knob
     (RAM register 41) — this is the firmware-side answer to "point C" (limiting
     acceleration to reduce backlash/inertia ringing).

2. Reference trajectories (the "what to command")
   - step_ref      : a clean position step (for step-response / system ID).
   - trapezoid_ref : host-side accel/jerk-limited move (motion profiling, point C).
   - sine_ref      : a single-frequency sine.
   - chirp_ref     : a swept-frequency sine (point B) — this is exactly what your
     Bambu printer does to find mechanical resonances, minus the accelerometer.

3. Acquisition
   - run_trajectory : stream a reference into one joint and record present position
     (and optionally current/load) as fast as the serial bus allows.
   - capture_step / capture_chirp : convenience wrappers.

4. Analysis (the "what did it do")
   - step_metrics      : rise time, overshoot %, settling time, steady-state error.
   - fit_second_order  : fit a 2nd-order model -> natural frequency wn & damping zeta.
   - frequency_response: estimate the FRF (Bode) from a chirp -> find resonance peak.
   - step_cost         : scalar cost for the auto-tuner to minimise.

Units: body joints are in DEGREES, gripper in 0-100 (lerobot normalises with the
calibration file). Time is in seconds.
"""

from __future__ import annotations

import time
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

PID_KEYS = ("P", "I", "D")
PID_REGISTERS = {"P": "P_Coefficient", "I": "I_Coefficient", "D": "D_Coefficient"}
COEFF_MIN, COEFF_MAX = 0, 254          # STS3215 P/I/D are 1-byte registers
ACCEL_MIN, ACCEL_MAX = 0, 254          # Acceleration register (0 = profiling off)

# Factory / lerobot defaults, for reference and seeding the auto-tuner.
FACTORY_PID = {"P": 32, "I": 0, "D": 32}


# ── Raw-value decoders (STS3215 datasheet) ───────────────────────────────────

def decode_load(raw: int) -> float:
    """Present_Load -> signed percentage (-100..100)."""
    magnitude = (raw & 0x3FF) / 10.0
    direction = (raw >> 10) & 1
    return -magnitude if direction else magnitude


def decode_current_mA(raw: int) -> float:
    """Present_Current -> milliamps (1 unit = 6.5 mA)."""
    return raw * 6.5


def decode_voltage(raw: int) -> float:
    """Present_Voltage -> volts."""
    return raw / 10.0


# ── Register access ──────────────────────────────────────────────────────────

def get_pid(bus, motor: str) -> dict:
    """Read the servo's internal P/I/D coefficients."""
    return {k: int(bus.read(PID_REGISTERS[k], motor, normalize=False)) for k in PID_KEYS}


def set_pid(bus, motor: str, P=None, I=None, D=None,
            verify: bool = True, torque_off: bool = False) -> dict:
    """
    Write P/I/D coefficients to the servo.

    The coefficients live in EEPROM, which is write-protected by the `Lock`
    register. lerobot's disable_torque() clears Lock *and* drops torque (the arm
    goes limp). For an auto-tune loop that is unacceptable, so by default we clear
    only the Lock bit, leaving torque ON, then re-lock. Every write is verified by
    reading it back; if your firmware refuses EEPROM writes while torque is on,
    the readback won't match and we raise — pass torque_off=True to fall back to
    the guaranteed (but limp-arm) path.

    Returns the coefficients actually present after writing.
    """
    targets = {"P": P, "I": I, "D": D}
    targets = {k: (None if v is None else int(np.clip(round(v), COEFF_MIN, COEFF_MAX)))
               for k, v in targets.items()}

    if torque_off:
        bus.disable_torque(motor)              # Torque_Enable=0 + Lock=0
    else:
        bus.write("Lock", motor, 0)            # unlock EEPROM only; torque untouched
    try:
        for k, v in targets.items():
            if v is not None:
                bus.write(PID_REGISTERS[k], motor, v, normalize=False)
    finally:
        if torque_off:
            bus.enable_torque(motor)           # Torque_Enable=1 + Lock=1
        else:
            bus.write("Lock", motor, 1)

    got = get_pid(bus, motor)
    if verify:
        for k, v in targets.items():
            if v is not None and got[k] != v:
                raise RuntimeError(
                    f"PID write for '{motor}' failed: wanted {k}={v}, read back {got[k]}. "
                    f"This firmware likely needs torque disabled to write EEPROM — "
                    f"retry with torque_off=True."
                )
    return got


def get_acceleration(bus, motor: str) -> int:
    return int(bus.read("Acceleration", motor, normalize=False))


def set_acceleration(bus, motor: str, accel: int) -> None:
    """
    Set the servo's on-board acceleration limit (RAM register, takes effect live).
    Lower = gentler trapezoidal ramp into/out of moves = less inertia ringing.
    0 disables the profiler (servo slews as fast as the gains allow).
    """
    bus.write("Acceleration", motor, int(np.clip(accel, ACCEL_MIN, ACCEL_MAX)), normalize=False)


def read_pos(bus, motor: str) -> float:
    """Present position in degrees (gripper: 0-100)."""
    return float(bus.read("Present_Position", motor, normalize=True))


def command_pos(bus, motor: str, deg: float) -> None:
    """Command a goal position in degrees (fire-and-forget, fast)."""
    bus.sync_write("Goal_Position", {motor: float(deg)})


def read_voltage(bus, motor: str) -> float:
    return decode_voltage(bus.read("Present_Voltage", motor, normalize=False))


# ── Reference-trajectory generators ──────────────────────────────────────────
# Each returns (ref_fn, duration_s). ref_fn(t)->goal_deg is sampled every cycle.

def step_ref(target_deg: float):
    """Constant target — combined with a pre-positioned start this is a step at t=0."""
    return (lambda t: target_deg)


def sine_ref(center_deg: float, amp_deg: float, freq_hz: float):
    w = 2.0 * np.pi * freq_hz
    return (lambda t: center_deg + amp_deg * np.sin(w * t))


def chirp_ref(center_deg: float, amp_deg: float, f0_hz: float, f1_hz: float, duration_s: float):
    """
    Linear frequency sweep f0 -> f1 over duration_s (the resonance test).
    Instantaneous phase of a linear chirp: phi(t) = 2*pi*(f0*t + k/2 * t^2), k=(f1-f0)/T.
    """
    k = (f1_hz - f0_hz) / duration_s
    def ref(t):
        phi = 2.0 * np.pi * (f0_hz * t + 0.5 * k * t * t)
        return center_deg + amp_deg * np.sin(phi)
    return ref


def trapezoid_ref(start_deg: float, target_deg: float, vmax_dps: float, amax_dps2: float):
    """
    Host-side trapezoidal (accel-limited) position profile, start -> target.
    Returns (ref_fn, total_time_s). Once it reaches the target it holds.

    This is the software side of "point C". You can use it instead of (or together
    with) the servo's Acceleration register to feed the joint smooth, jerk-bounded
    setpoints so fast moves don't excite gearbox/inertia backlash.
    """
    dist = abs(target_deg - start_deg)
    direction = np.sign(target_deg - start_deg)
    if dist < 1e-9:
        return (lambda t: target_deg), 0.0

    t_acc = vmax_dps / amax_dps2
    d_acc = 0.5 * amax_dps2 * t_acc ** 2
    if 2 * d_acc >= dist:                       # triangular (never reaches vmax)
        t_acc = np.sqrt(dist / amax_dps2)
        d_acc = 0.5 * amax_dps2 * t_acc ** 2
        t_flat = 0.0
        vpeak = amax_dps2 * t_acc
    else:                                       # trapezoidal
        t_flat = (dist - 2 * d_acc) / vmax_dps
        vpeak = vmax_dps
    total = 2 * t_acc + t_flat

    def ref(t):
        if t <= 0:
            return start_deg
        if t >= total:
            return target_deg
        if t < t_acc:                           # accelerating
            d = 0.5 * amax_dps2 * t ** 2
        elif t < t_acc + t_flat:                # cruising
            d = d_acc + vpeak * (t - t_acc)
        else:                                   # decelerating
            td = t - t_acc - t_flat
            d = d_acc + vpeak * t_flat + vpeak * td - 0.5 * amax_dps2 * td ** 2
        return start_deg + direction * d
    return ref, total


# ── Acquisition ──────────────────────────────────────────────────────────────

def run_trajectory(bus, motor: str, ref_fn, duration_s: float,
                   extra: bool = False, max_rate_hz: float | None = None) -> dict:
    """
    Stream ref_fn(t) into `motor` and record Present_Position as fast as possible.

    Only re-commands the goal when it actually changes, so a step test spends almost
    all its serial bandwidth on position reads (highest possible sample rate). Set
    extra=True to also log current/load (adds round-trips -> lower rate).

    Returns numpy arrays: t, goal, pos [, current_mA, load_pct]. Also t includes the
    achieved mean sample rate in out["fs_hz"].
    """
    t_log, goal_log, pos_log = [], [], []
    cur_log, load_log = [], []
    last_goal = None
    min_dt = (1.0 / max_rate_hz) if max_rate_hz else 0.0

    t0 = time.perf_counter()
    while True:
        t = time.perf_counter() - t0
        if t > duration_s:
            break
        goal = float(ref_fn(t))
        if last_goal is None or abs(goal - last_goal) > 1e-4:
            bus.sync_write("Goal_Position", {motor: goal})
            last_goal = goal
        pos = float(bus.read("Present_Position", motor, normalize=True))
        t_log.append(t); goal_log.append(goal); pos_log.append(pos)
        if extra:
            cur_log.append(decode_current_mA(bus.read("Present_Current", motor, normalize=False)))
            load_log.append(decode_load(bus.read("Present_Load", motor, normalize=False)))
        if min_dt:
            time.sleep(max(min_dt - (time.perf_counter() - t0 - t), 0.0))

    out = {"t": np.asarray(t_log), "goal": np.asarray(goal_log), "pos": np.asarray(pos_log)}
    if len(out["t"]) > 1:
        out["fs_hz"] = float(1.0 / np.median(np.diff(out["t"])))
    else:
        out["fs_hz"] = 0.0
    if extra:
        out["current_mA"] = np.asarray(cur_log)
        out["load_pct"] = np.asarray(load_log)
    return out


def capture_step(bus, motor: str, start_deg: float, target_deg: float,
                 record_s: float = 1.5, pre_settle_s: float = 1.0, **kw) -> dict:
    """Move to start, let it settle, then step to target at t=0 and record."""
    command_pos(bus, motor, start_deg)
    time.sleep(pre_settle_s)
    data = run_trajectory(bus, motor, step_ref(target_deg), record_s, **kw)
    data["start_deg"] = start_deg
    data["target_deg"] = target_deg
    return data


def capture_chirp(bus, motor: str, center_deg: float, amp_deg: float,
                  f0_hz: float, f1_hz: float, duration_s: float = 8.0, **kw) -> dict:
    """Pre-position at center, then sweep f0->f1 and record."""
    command_pos(bus, motor, center_deg)
    time.sleep(0.8)
    ref = chirp_ref(center_deg, amp_deg, f0_hz, f1_hz, duration_s)
    data = run_trajectory(bus, motor, ref, duration_s, **kw)
    data["f0_hz"], data["f1_hz"] = f0_hz, f1_hz
    return data


# ── Analysis: step response ──────────────────────────────────────────────────

def step_metrics(t: np.ndarray, goal: np.ndarray, pos: np.ndarray,
                 settle_band_pct: float = 2.0) -> dict:
    """
    Classic step-response metrics computed directly from measured data.

    Handles steps in either direction. Steady-state value is the mean of the last
    10% of the record. Returns rise_time_s, overshoot_pct, settling_time_s,
    steady_state_error_deg, plus rms_error_deg over the whole record.
    """
    t = np.asarray(t, float); pos = np.asarray(pos, float); goal = np.asarray(goal, float)
    target = float(goal[-1])
    y0 = float(pos[0])
    step = target - y0
    n_tail = max(1, len(pos) // 10)
    y_final = float(np.mean(pos[-n_tail:]))

    out = {
        "step_deg": step,
        "y_final_deg": y_final,
        "steady_state_error_deg": target - y_final,
        "rms_error_deg": float(np.sqrt(np.mean((pos - goal) ** 2))),
    }

    if abs(step) < 1e-6:
        out.update(rise_time_s=float("nan"), overshoot_pct=float("nan"),
                   settling_time_s=float("nan"))
        return out

    # Normalise to a 0->1 rising response regardless of direction.
    norm = (pos - y0) / step                       # 0 at start, ~1 at settle

    # Rise time: 10% -> 90% of the achieved span.
    def first_cross(level):
        idx = np.where(norm >= level)[0]
        return t[idx[0]] if len(idx) else float("nan")
    t10, t90 = first_cross(0.10), first_cross(0.90)
    out["rise_time_s"] = (t90 - t10) if (t90 == t90 and t10 == t10) else float("nan")

    # Overshoot: peak excursion beyond the final value, as % of the step.
    peak_excursion = (np.max(pos) - y_final) if step > 0 else (y_final - np.min(pos))
    out["overshoot_pct"] = max(0.0, 100.0 * peak_excursion / abs(step))

    # Settling time: last time the response is outside the ±band around y_final.
    band = abs(step) * settle_band_pct / 100.0
    outside = np.where(np.abs(pos - y_final) > band)[0]
    out["settling_time_s"] = float(t[outside[-1]]) if len(outside) else 0.0
    return out


def fit_second_order(t: np.ndarray, pos: np.ndarray, target: float, y0: float | None = None):
    """
    Fit an under-damped 2nd-order step response and return (wn, zeta, model_fn).
    wn = natural frequency (rad/s), zeta = damping ratio. Useful for system ID and
    for choosing a chirp frequency band. Returns (None, None, None) if the fit fails.
    """
    from scipy.optimize import curve_fit
    t = np.asarray(t, float); pos = np.asarray(pos, float)
    if y0 is None:
        y0 = float(pos[0])
    A = target - y0

    def model(tt, wn, zeta):
        zeta = np.clip(zeta, 1e-3, 0.999)
        wd = wn * np.sqrt(1 - zeta ** 2)
        env = np.exp(-zeta * wn * tt)
        return y0 + A * (1 - env * (np.cos(wd * tt) + (zeta / np.sqrt(1 - zeta ** 2)) * np.sin(wd * tt)))

    try:
        popt, _ = curve_fit(model, t, pos, p0=[20.0, 0.3],
                            bounds=([1.0, 1e-3], [400.0, 0.999]), maxfev=8000)
        wn, zeta = float(popt[0]), float(popt[1])
        return wn, zeta, (lambda tt: model(np.asarray(tt, float), wn, zeta))
    except Exception:
        return None, None, None


# ── Analysis: frequency response (from a chirp) ──────────────────────────────

def frequency_response(t: np.ndarray, goal: np.ndarray, pos: np.ndarray,
                       fmax_hz: float | None = None):
    """
    Estimate the closed-loop frequency response H(f) = Pos/Goal from a chirp run,
    using Welch cross/auto spectra. Returns dict: freq_hz, mag_db, phase_deg,
    coherence, resonance_hz (peak of |H| above DC, where coherence is trustworthy).

    Resolution is capped by the achieved sample rate; with encoder-only feedback at
    a few hundred Hz this is reliable to ~20-30 Hz. An end-effector accelerometer
    (IMU) would extend this dramatically.
    """
    from scipy import signal
    t = np.asarray(t, float); goal = np.asarray(goal, float); pos = np.asarray(pos, float)
    if len(t) < 16:
        return None

    # Resample onto a uniform grid (serial timing jitters).
    fs = 1.0 / np.median(np.diff(t))
    t_u = np.arange(t[0], t[-1], 1.0 / fs)
    g = np.interp(t_u, t, goal) - np.mean(goal)
    y = np.interp(t_u, t, pos) - np.mean(pos)

    nper = min(len(t_u), 1024)
    f, Pxx = signal.welch(g, fs=fs, nperseg=nper)
    _, Pyy = signal.welch(y, fs=fs, nperseg=nper)
    _, Pxy = signal.csd(g, y, fs=fs, nperseg=nper)

    H = Pxy / np.where(Pxx == 0, np.nan, Pxx)
    mag_db = 20 * np.log10(np.abs(H) + 1e-12)
    phase_deg = np.degrees(np.angle(H))
    coh = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-18)

    fmax = fmax_hz or (fs / 2)
    band = (f > 0.5) & (f <= fmax) & (coh > 0.5)
    res_hz = float(f[band][np.argmax(mag_db[band])]) if np.any(band) else float("nan")

    return {"freq_hz": f, "mag_db": mag_db, "phase_deg": phase_deg,
            "coherence": coh, "resonance_hz": res_hz, "fs_hz": fs}


# ── Auto-tune cost ───────────────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "overshoot": 1.0,     # per %
    "settling": 20.0,     # per second
    "sse": 5.0,           # per degree
    "rise": 5.0,          # per second (encourage speed, but mildly)
    "rms": 2.0,           # per degree of tracking error
}


def step_cost(metrics: dict, weights: dict | None = None) -> float:
    """
    Scalar cost for the optimiser to MINIMISE, built from step metrics.
    Lower = crisper response (low overshoot, fast settle, small steady-state error).
    NaNs (e.g. a response that never rose) are penalised heavily.
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}

    def safe(x, big=1e3):
        return big if (x is None or x != x) else x

    return (
        w["overshoot"] * safe(metrics.get("overshoot_pct"))
        + w["settling"] * safe(metrics.get("settling_time_s"))
        + w["sse"] * abs(safe(metrics.get("steady_state_error_deg")))
        + w["rise"] * safe(metrics.get("rise_time_s"))
        + w["rms"] * safe(metrics.get("rms_error_deg"))
    )
