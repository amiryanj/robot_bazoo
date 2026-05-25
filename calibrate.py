#!/usr/bin/env python
"""
calibrate.py — STS3215 controller characterisation & auto-tuning for the SO-101.

Subcommands
-----------
  step      Run a clean position step, measure rise/overshoot/settling/steady-state
            error, fit a 2nd-order model, save a CSV + plot.
  chirp     Sweep a sine over a frequency band (the "resonance test", like a 3-D
            printer's input-shaping calibration) and plot the frequency response /
            find the resonance peak.
  profile   Compare a step with the servo's acceleration profiler OFF vs ON, to show
            how acceleration limiting tames inertia/backlash ringing (point C).
  autotune  Closed-loop gain search with Optuna: repeatedly step the joint, score the
            response, and converge on P/I/D that minimise overshoot + settling +
            tracking error. Writes the best gains and saves before/after plots.

Examples
--------
  conda activate lerobot
  python calibrate.py step     --joint shoulder_pan --size 25
  python calibrate.py chirp    --joint shoulder_pan --amp 8 --f0 0.5 --f1 15
  python calibrate.py profile  --joint shoulder_pan --size 25 --accel 20
  python calibrate.py autotune --joint shoulder_pan --size 25 --trials 40

Notes
-----
- Power: tune at the correct 7.4-7.5 V supply. At low voltage the servo trips its
  under-voltage / overload protection mid-test and the numbers are meaningless.
- EEPROM wear: P/I/D live in EEPROM. autotune writes them once per trial; a few
  hundred trials is harmless, but don't leave it looping for hours.
- Restore factory gains any time with:  python calibrate.py step --joint X --p 32 --i 0 --d 32 --size 0
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

import servo_tuning as st
from gamepad_utils import JOINT_LIMITS, graceful_shutdown

PORT = "/dev/ttyACM0"
ROBOT_ID = "so101"
OUT_ROOT = Path("/home/javad/workspace/lerobot_all/outputs/tuning")


# ── Setup helpers ────────────────────────────────────────────────────────────

def connect(port: str, robot_id: str) -> SOFollower:
    """Connect with no cameras (faster, not needed for tuning)."""
    robot = SOFollower(SOFollowerRobotConfig(port=port, id=robot_id))
    print("Connecting robot (no cameras)...")
    robot.connect()
    return robot


def hold_all(robot: SOFollower) -> dict:
    """Pin every joint at its current position so the arm doesn't drift during a test."""
    obs = robot.get_observation()
    robot.send_action({k: v for k, v in obs.items() if k.endswith(".pos")})
    return obs


def check_voltage(bus, motor: str) -> float:
    v = st.read_voltage(bus, motor)
    warn = "  ⚠ LOW — expect protection trips, results unreliable" if v < 6.5 else ""
    print(f"  Supply voltage: {v:.1f} V{warn}")
    return v


def clamp_to_limits(motor: str, deg: float) -> float:
    lo, hi = JOINT_LIMITS[motor]
    return float(np.clip(deg, lo, hi))


def apply_gains(bus, motor: str, args) -> dict:
    """Set P/I/D and/or Acceleration if the user passed them; report what's active."""
    if any(v is not None for v in (args.p, args.i, args.d)):
        st.set_pid(bus, motor, P=args.p, I=args.i, D=args.d, torque_off=args.torque_off)
    if args.accel is not None:
        st.set_acceleration(bus, motor, args.accel)
    pid = st.get_pid(bus, motor)
    accel = st.get_acceleration(bus, motor)
    print(f"  Active gains: P={pid['P']} I={pid['I']} D={pid['D']}  Acceleration={accel}")
    return pid


def out_dir(tag: str) -> Path:
    d = OUT_ROOT / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run_csv(path: Path, data: dict) -> None:
    keys = [k for k in ("t", "goal", "pos", "current_mA", "load_pct") if k in data]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for row in zip(*[data[k] for k in keys]):
            w.writerow([f"{x:.4f}" for x in row])


def print_metrics(m: dict, fit=None) -> None:
    print("  ── Step metrics ─────────────────────────")
    print(f"    rise time        : {m['rise_time_s']*1000:7.1f} ms")
    print(f"    overshoot        : {m['overshoot_pct']:7.1f} %")
    print(f"    settling time    : {m['settling_time_s']*1000:7.1f} ms")
    print(f"    steady-state err : {m['steady_state_error_deg']:7.2f} deg")
    print(f"    rms tracking err : {m['rms_error_deg']:7.2f} deg")
    if fit and fit[0] is not None:
        wn, zeta = fit[0], fit[1]
        print(f"    fitted model     : wn={wn:6.1f} rad/s ({wn/(2*np.pi):4.1f} Hz)  zeta={zeta:4.2f}")


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_step(data: dict, metrics: dict, fit, title: str, path: Path) -> None:
    t, goal, pos = data["t"], data["goal"], data["pos"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, goal, "--", color="gray", label="goal")
    ax.plot(t, pos, "-", color="C0", lw=1.6, label="position")
    y_final = metrics["y_final_deg"]
    band = abs(metrics["step_deg"]) * 0.02
    ax.axhspan(y_final - band, y_final + band, color="C2", alpha=0.12, label="±2% band")
    if fit and fit[2] is not None:
        ax.plot(t, fit[2](t), ":", color="C3", lw=1.2, label="2nd-order fit")
    ax.set_xlabel("time (s)"); ax.set_ylabel("position (deg)")
    ax.set_title(title); ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_overlay(runs: list, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(runs[0][1]["t"], runs[0][1]["goal"], "--", color="gray", label="goal")
    for i, (label, d) in enumerate(runs):
        ax.plot(d["t"], d["pos"], "-", lw=1.6, color=f"C{i}", label=label)
    ax.set_xlabel("time (s)"); ax.set_ylabel("position (deg)")
    ax.set_title(title); ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_bode(fr: dict, title: str, path: Path) -> None:
    f = fr["freq_hz"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].semilogx(f, fr["mag_db"], color="C0"); axes[0].set_ylabel("|H| (dB)")
    if fr["resonance_hz"] == fr["resonance_hz"]:
        axes[0].axvline(fr["resonance_hz"], color="C3", ls="--",
                        label=f"resonance ≈ {fr['resonance_hz']:.1f} Hz")
        axes[0].legend(loc="best")
    axes[1].semilogx(f, fr["phase_deg"], color="C1"); axes[1].set_ylabel("phase (deg)")
    axes[2].semilogx(f, fr["coherence"], color="C2"); axes[2].set_ylabel("coherence")
    axes[2].set_xlabel("frequency (Hz)"); axes[2].set_ylim(0, 1.05)
    for a in axes:
        a.grid(alpha=0.3, which="both")
    axes[0].set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ── Subcommands ──────────────────────────────────────────────────────────────

def resolve_step_targets(bus, motor, args):
    """Return (start, target) in joint units, clamped to limits."""
    cur = st.read_pos(bus, motor)
    start = args.start if args.start is not None else cur
    target = args.target if args.target is not None else cur + args.size
    return clamp_to_limits(motor, start), clamp_to_limits(motor, target)


def cmd_step(robot, args):
    bus = robot.bus
    motor = args.joint
    check_voltage(bus, motor)
    pid = apply_gains(bus, motor, args)
    start, target = resolve_step_targets(bus, motor, args)
    print(f"  Step: {start:.1f} -> {target:.1f} deg  (record {args.record}s)")

    data = st.capture_step(bus, motor, start, target,
                           record_s=args.record, pre_settle_s=args.settle)
    print(f"  Sample rate achieved: {data['fs_hz']:.0f} Hz ({len(data['t'])} samples)")
    metrics = st.step_metrics(data["t"], data["goal"], data["pos"])
    fit = st.fit_second_order(data["t"], data["pos"], target)
    print_metrics(metrics, fit)

    d = out_dir(f"step_{motor}")
    save_run_csv(d / "step.csv", data)
    title = f"{motor} step {start:.0f}->{target:.0f}°  P={pid['P']} I={pid['I']} D={pid['D']}"
    plot_step(data, metrics, fit, title, d / "step.png")
    print(f"  Saved: {d}")


def cmd_chirp(robot, args):
    bus = robot.bus
    motor = args.joint
    check_voltage(bus, motor)
    apply_gains(bus, motor, args)
    center = clamp_to_limits(motor, st.read_pos(bus, motor))
    lo, hi = JOINT_LIMITS[motor]
    amp = min(args.amp, (hi - lo) / 2 - 1)
    print(f"  Chirp: center={center:.1f}° amp={amp:.1f}° sweep {args.f0}->{args.f1} Hz over {args.duration}s")

    data = st.capture_chirp(bus, motor, center, amp, args.f0, args.f1, args.duration)
    print(f"  Sample rate achieved: {data['fs_hz']:.0f} Hz ({len(data['t'])} samples)")
    if data["fs_hz"] < 2 * args.f1:
        print(f"  ⚠ Sample rate < 2×f1 ({2*args.f1:.0f} Hz Nyquist) — raise f1 down or trust only low freqs.")

    fr = st.frequency_response(data["t"], data["goal"], data["pos"], fmax_hz=args.f1)
    if fr:
        print(f"  Resonance peak ≈ {fr['resonance_hz']:.1f} Hz")
    d = out_dir(f"chirp_{motor}")
    save_run_csv(d / "chirp.csv", data)
    if fr:
        plot_bode(fr, f"{motor} frequency response", d / "bode.png")
    print(f"  Saved: {d}")


def cmd_profile(robot, args):
    bus = robot.bus
    motor = args.joint
    check_voltage(bus, motor)
    # In profile, --accel is the "ON" value to A/B-test; don't pre-apply it as a gain.
    test_accel = args.accel if args.accel is not None else 20
    args.accel = None
    apply_gains(bus, motor, args)
    start, target = resolve_step_targets(bus, motor, args)

    runs = []
    for accel, label in [(0, "profiler OFF (accel=0)"), (test_accel, f"profiler ON (accel={test_accel})")]:
        st.set_acceleration(bus, motor, accel)
        time.sleep(0.2)
        data = st.capture_step(bus, motor, start, target, record_s=args.record, pre_settle_s=args.settle)
        m = st.step_metrics(data["t"], data["goal"], data["pos"])
        print(f"  {label}: overshoot {m['overshoot_pct']:.1f}%  settling {m['settling_time_s']*1000:.0f} ms")
        runs.append((label, data))

    d = out_dir(f"profile_{motor}")
    plot_overlay(runs, f"{motor} step {start:.0f}->{target:.0f}°  acceleration profiling", d / "profile.png")
    for label, data in runs:
        save_run_csv(d / f"{label.split()[1].lower()}.csv", data)
    print(f"  Saved: {d}")


def cmd_autotune(robot, args):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    bus = robot.bus
    motor = args.joint
    check_voltage(bus, motor)
    if args.accel is not None:
        st.set_acceleration(bus, motor, args.accel)

    start, target = resolve_step_targets(bus, motor, args)
    print(f"  Auto-tuning '{motor}'  step {start:.1f}->{target:.1f}°  trials={args.trials}")
    print(f"  Search: P∈[{args.p_min},{args.p_max}] I∈[{args.i_min},{args.i_max}] D∈[{args.d_min},{args.d_max}]")

    baseline_gains = st.get_pid(bus, motor)
    print(f"  Baseline gains: {baseline_gains}")

    def evaluate(P, I, D) -> tuple[float, dict, dict]:
        try:
            st.set_pid(bus, motor, P=P, I=I, D=D, torque_off=args.torque_off)
            data = st.capture_step(bus, motor, start, target,
                                   record_s=args.record, pre_settle_s=args.settle)
            m = st.step_metrics(data["t"], data["goal"], data["pos"])
            return st.step_cost(m), m, data
        except Exception as e:
            print(f"    trial error ({e}) — penalising")
            return 1e6, {}, {}

    # Baseline + factory as a first reference point.
    base_cost, base_m, base_data = evaluate(**baseline_gains)
    print(f"  Baseline cost: {base_cost:.1f}")

    def objective(trial):
        P = trial.suggest_int("P", args.p_min, args.p_max)
        I = trial.suggest_int("I", args.i_min, args.i_max)
        D = trial.suggest_int("D", args.d_min, args.d_max)
        cost, m, _ = evaluate(P, I, D)
        if m:
            trial.set_user_attr("overshoot_pct", round(m["overshoot_pct"], 1))
            trial.set_user_attr("settling_ms", round(m["settling_time_s"] * 1000))
        return cost

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    # Seed the search with known-reasonable points so it never starts blind.
    study.enqueue_trial(baseline_gains)
    study.enqueue_trial(st.FACTORY_PID)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)

    best = study.best_params
    best_cost, best_m, best_data = evaluate(**best)
    print("\n  ── Result ───────────────────────────────")
    print(f"    best gains : P={best['P']} I={best['I']} D={best['D']}")
    print(f"    best cost  : {best_cost:.1f}   (baseline {base_cost:.1f})")
    print_metrics(best_m)

    # Leave the best gains active on the servo (persists in EEPROM).
    st.set_pid(bus, motor, P=best["P"], I=best["I"], D=best["D"], torque_off=args.torque_off)
    print(f"  Best gains written to '{motor}'. Restore factory with --p 32 --i 0 --d 32.")

    d = out_dir(f"autotune_{motor}")
    # before/after overlay
    if base_data and best_data:
        plot_overlay([("baseline", base_data), ("tuned", best_data)],
                     f"{motor} step response: baseline vs tuned", d / "before_after.png")
    # trial history CSV
    with open(d / "trials.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "P", "I", "D", "cost", "overshoot_pct", "settling_ms"])
        for t in study.trials:
            w.writerow([t.number, t.params.get("P"), t.params.get("I"), t.params.get("D"),
                        f"{t.value:.2f}" if t.value is not None else "",
                        t.user_attrs.get("overshoot_pct", ""), t.user_attrs.get("settling_ms", "")])
    print(f"  Saved: {d}")


# ── Argparse ─────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default=PORT)
    p.add_argument("--id", default=ROBOT_ID, dest="robot_id")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--joint", required=True, choices=list(JOINT_LIMITS.keys()))
        sp.add_argument("--p", type=int, default=None, help="set P coefficient before test")
        sp.add_argument("--i", type=int, default=None, help="set I coefficient before test")
        sp.add_argument("--d", type=int, default=None, help="set D coefficient before test")
        sp.add_argument("--accel", type=int, default=None, help="set Acceleration register (0-254)")
        sp.add_argument("--torque-off", action="store_true", dest="torque_off",
                        help="disable torque while writing EEPROM (arm goes limp; use if PID writes fail to verify)")
        sp.add_argument("--record", type=float, default=1.5, help="record duration (s)")
        sp.add_argument("--settle", type=float, default=1.0, help="pre-step settle time (s)")

    sp = sub.add_parser("step", help="step response + metrics"); common(sp)
    sp.add_argument("--size", type=float, default=20.0, help="step magnitude from current pos (deg)")
    sp.add_argument("--start", type=float, default=None)
    sp.add_argument("--target", type=float, default=None)

    sp = sub.add_parser("chirp", help="frequency sweep + Bode/resonance"); common(sp)
    sp.add_argument("--amp", type=float, default=8.0, help="sweep amplitude (deg)")
    sp.add_argument("--f0", type=float, default=0.5, help="start frequency (Hz)")
    sp.add_argument("--f1", type=float, default=15.0, help="end frequency (Hz)")
    sp.add_argument("--duration", type=float, default=8.0, help="sweep duration (s)")

    sp = sub.add_parser("profile", help="acceleration-profiling A/B comparison"); common(sp)
    sp.add_argument("--size", type=float, default=20.0)
    sp.add_argument("--start", type=float, default=None)
    sp.add_argument("--target", type=float, default=None)
    # NOTE: --accel comes from common(); for profile it's the "ON" value to A/B (default 20 applied in handler)

    sp = sub.add_parser("autotune", help="Optuna closed-loop PID search"); common(sp)
    sp.add_argument("--size", type=float, default=20.0)
    sp.add_argument("--start", type=float, default=None)
    sp.add_argument("--target", type=float, default=None)
    sp.add_argument("--trials", type=int, default=40)
    sp.add_argument("--p-min", type=int, default=8, dest="p_min")
    sp.add_argument("--p-max", type=int, default=100, dest="p_max")
    sp.add_argument("--i-min", type=int, default=0, dest="i_min")
    sp.add_argument("--i-max", type=int, default=20, dest="i_max")
    sp.add_argument("--d-min", type=int, default=0, dest="d_min")
    sp.add_argument("--d-max", type=int, default=100, dest="d_max")
    return p


HANDLERS = {"step": cmd_step, "chirp": cmd_chirp, "profile": cmd_profile, "autotune": cmd_autotune}


def main():
    args = build_parser().parse_args()
    robot = connect(args.port, args.robot_id)
    try:
        hold_all(robot)
        HANDLERS[args.cmd](robot, args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        graceful_shutdown(robot)
        try:
            robot.disconnect()
        except Exception as e:
            print(f"  Disconnect warning: {e}")


if __name__ == "__main__":
    main()
