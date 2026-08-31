#!/usr/bin/env python
"""
Offline analysis of a station.py run: correlate motor motion with wrist-IMU
vibration, and localise the IMU frame on the arm.

Two questions:
  1. VIBRATION — strip gravity from the 800 Hz accel (high-pass), look at its
     spectrum (resonance peaks) and correlate its energy with joint speed /
     motor current. "Which motion shakes the wrist, and at what frequency?"
  2. LOCALIZATION — at static poses the accel IS gravity. MuJoCo FK gives the
     wrist_roll link's world orientation per pose, so gravity-in-link vs
     gravity-in-IMU over several poses solves the fixed mounting rotation
     R(link→imu) via Kabsch. "How is the sensor glued on?"

Usage:
    python analyze_run.py outputs/logs/2026-06-10_00-48-25
Writes <run>/analysis.png and prints a summary.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, find_peaks

SCALE = 0.038246                 # m/s² per LSB
G = 9.80665
from config import SCENE_XML as XML
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
IMU_LINK_BODY = "gripper"        # body that rotates with wrist_roll (carries the IMU)
HP_HZ = 3.0                      # high-pass cutoff: gravity/orientation < this, vibration above


# ── Load ────────────────────────────────────────────────────────────────────────

def load(run):
    m = pd.read_csv(f"{run}/station.csv")
    i = pd.read_csv(f"{run}/imu.csv")
    pos = m.pivot_table(index="time_s", columns="motor", values="position_deg")
    cur = m.pivot_table(index="time_s", columns="motor", values="current_mA")
    t = i.time_s.values
    fs = 1.0 / np.median(np.diff(t))
    acc = np.c_[i.x_raw, i.y_raw, i.z_raw] * SCALE       # m/s², IMU frame
    return pos, cur, t, fs, acc


# ── Vibration ─────────────────────────────────────────────────────────────────────

def vibration(t, fs, acc):
    b, a = butter(4, HP_HZ / (fs / 2), btype="high")
    ac = filtfilt(b, a, acc, axis=0)                     # gravity-removed
    vib = np.linalg.norm(ac, axis=1)                     # instantaneous |AC accel|
    # sliding RMS over ~40 ms
    w = max(1, int(0.04 * fs))
    rms = np.sqrt(np.convolve(vib**2, np.ones(w) / w, mode="same"))
    return ac, vib, rms


def spectrum(ac, fs):
    f, P = welch(np.linalg.norm(ac, axis=1), fs=fs, nperseg=4096)
    peaks, _ = find_peaks(P, prominence=P.max() * 0.05)
    peaks = peaks[np.argsort(P[peaks])[::-1]][:6]        # 6 strongest
    return f, P, sorted(f[p] for p in peaks)


# ── Static-pose detection ─────────────────────────────────────────────────────────

def static_poses(pos, t, rms):
    """Return list of (joint_angles_dict, mean_unit_gravity_imu) for distinct, quiet poses."""
    speed = pos.diff().abs().sum(axis=1) / pos.index.to_series().diff()
    quiet = speed < 1.0                                  # deg/s, whole-arm
    rms_at = np.interp(pos.index.values, t, rms)
    quiet &= rms_at < 0.5                                 # m/s², low shake
    # group consecutive quiet motor-frames into segments
    segs, cur = [], []
    qi = quiet.values
    for k, q in enumerate(qi):
        if q:
            cur.append(k)
        elif cur:
            segs.append(cur); cur = []
    if cur:
        segs.append(cur)

    poses = []
    for seg in segs:
        if len(seg) < 3:                                 # need a sustained hold
            continue
        t0, t1 = pos.index.values[seg[0]], pos.index.values[seg[-1]]
        ang = {j: float(pos[j].iloc[seg].mean()) for j in JOINTS}
        # mean accel over the same window → gravity direction in IMU frame
        sel = (t >= t0) & (t <= t1)
        if sel.sum() < 20:
            continue
        poses.append((ang, t1 - t0, sel))
    # keep distinct poses (joint vector differs > 5° from any kept one)
    kept = []
    for ang, dur, sel in sorted(poses, key=lambda p: -p[1]):
        v = np.array([ang[j] for j in JOINTS])
        if all(np.linalg.norm(v - np.array([k[0][j] for j in JOINTS])) > 5 for k in kept):
            kept.append((ang, dur, sel))
    return kept


# ── IMU mounting via MuJoCo FK + Kabsch ───────────────────────────────────────────

def fk_link_R(mm, md, mujoco, ang):
    import math
    for j in JOINTS:
        md.qpos[mm.jnt_qposadr[mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_JOINT, j)]] = math.radians(ang[j])
    mujoco.mj_forward(mm, md)
    bid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, IMU_LINK_BODY)
    return md.xmat[bid].reshape(3, 3).copy()             # R_world_link


def kabsch(g_link, g_imu):
    """Rotation R (link→imu) minimising ||g_imu - R g_link||, columns = unit vectors."""
    H = g_link @ g_imu.T
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def localize(run, poses, t, acc):
    import mujoco
    os.chdir(os.path.dirname(XML))
    mm = mujoco.MjModel.from_xml_path(XML)
    md = mujoco.MjData(mm)

    g_world = np.array([0, 0, -1.0])                     # gravity points down in world
    G_link, G_imu, info = [], [], []
    for ang, dur, sel in poses:
        g_i = acc[sel].mean(axis=0)
        g_i /= np.linalg.norm(g_i)                       # measured gravity dir, IMU frame
        Rwl = fk_link_R(mm, md, mujoco, ang)
        g_l = Rwl.T @ g_world                            # gravity dir, link frame
        G_link.append(g_l); G_imu.append(g_i)
        info.append((ang, dur, g_i, g_l))
    G_link = np.array(G_link).T
    G_imu = np.array(G_imu).T

    R = kabsch(G_link, G_imu) if G_link.shape[1] >= 2 else None
    resid = None
    if R is not None:
        pred = R @ G_link
        resid = np.degrees(np.arccos(np.clip((pred * G_imu).sum(0), -1, 1)))
    return R, resid, info


# ── Correlation on a common grid ──────────────────────────────────────────────────

def correlate(pos, cur, t, rms):
    grid = np.arange(t[0], t[-1], 0.02)                  # 50 Hz
    speed = {j: np.interp(grid, pos.index.values,
                          np.gradient(pos[j].values, pos.index.values)) for j in JOINTS}
    tot_speed = np.sum([np.abs(speed[j]) for j in JOINTS], axis=0)
    tot_cur = np.interp(grid, cur.index.values, cur.sum(axis=1).values)
    vib = np.interp(grid, t, rms)
    cc_speed = {j: np.corrcoef(np.abs(speed[j]), vib)[0, 1] for j in JOINTS}
    cc_tot = np.corrcoef(tot_speed, vib)[0, 1]
    cc_cur = np.corrcoef(tot_cur, vib)[0, 1]
    return grid, tot_speed, tot_cur, vib, cc_speed, cc_tot, cc_cur


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "outputs/logs/2026-06-10_00-48-25"
    run = os.path.abspath(run)                           # localize() chdir's away — keep absolute
    pos, cur, t, fs, acc = load(run)
    ac, vib, rms = vibration(t, fs, acc)
    f, P, peaks = spectrum(ac, fs)
    grid, tot_speed, tot_cur, vibg, cc_speed, cc_tot, cc_cur = correlate(pos, cur, t, rms)
    poses = static_poses(pos, t, rms)
    R, resid, info = localize(run, poses, t, acc)

    # ── report ────────────────────────────────────────────────────────────────────
    print(f"\n=== {run} ===")
    print(f"IMU fs ≈ {fs:.0f} Hz, {len(t)} samples, {t[-1]-t[0]:.1f}s")
    print(f"\nVIBRATION (gravity-removed, >{HP_HZ} Hz):")
    print(f"  |AC accel| RMS = {vib.std():.3f} m/s², peak = {vib.max():.2f} m/s²")
    print(f"  resonance/spectral peaks: {', '.join(f'{p:.1f} Hz' for p in peaks)}")
    print(f"\nCORRELATION of wrist vibration with:")
    print(f"  total joint speed : r = {cc_tot:+.2f}")
    print(f"  total motor current: r = {cc_cur:+.2f}")
    for j in sorted(cc_speed, key=lambda k: -abs(cc_speed[k])):
        if not np.isnan(cc_speed[j]):
            print(f"    {j:<14} r = {cc_speed[j]:+.2f}")
    print(f"\nLOCALIZATION — {len(info)} distinct static poses:")
    for ang, dur, g_i, g_l in info:
        a = " ".join(f"{j[:4]}={ang[j]:+.0f}" for j in JOINTS if abs(ang[j]) > 1)
        print(f"  hold {dur:.1f}s [{a}]  g_imu=({g_i[0]:+.2f},{g_i[1]:+.2f},{g_i[2]:+.2f})")
    if R is not None:
        print(f"\n  R(link→imu) =\n{np.array2string(R, precision=3, suppress_small=True, prefix='    ')}")
        print(f"  fit residual per pose: {np.array2string(resid, precision=1)}  (deg)")
        print(f"  mean residual = {resid.mean():.1f}°  "
              f"({'GOOD' if resid.mean() < 5 else 'rough — needs more/cleaner static poses'})")
        axes = "XYZ"
        for k in range(3):
            link_axis = R.T[:, k]                          # imu axis k expressed in link frame
            dom = np.argmax(np.abs(link_axis))
            print(f"  IMU +{axes[k]} ≈ {'+-'[int(link_axis[dom]<0)]}link-{axes[dom]} "
                  f"({link_axis[dom]:+.2f})")
    else:
        print("  not enough distinct static poses to solve mounting rotation.")

    # ── figure ────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(4, 1, figsize=(11, 12))
    for j in JOINTS:
        if (pos[j].max() - pos[j].min()) > 2:
            ax[0].plot(pos.index, pos[j], label=j, lw=1)
    ax[0].set_title("Joint positions"); ax[0].set_ylabel("deg"); ax[0].legend(fontsize=8, ncol=3)

    ax[1].plot(t, rms, color="crimson", lw=0.7)
    ax[1].set_title("Wrist vibration RMS (gravity-removed)"); ax[1].set_ylabel("m/s²")
    a1 = ax[1].twinx(); a1.plot(grid, tot_speed, color="steelblue", lw=0.7, alpha=0.6)
    a1.set_ylabel("Σ|joint speed| (deg/s)", color="steelblue")

    ax[2].loglog(f, P, color="k", lw=1)
    for p in peaks:
        ax[2].axvline(p, color="orange", ls="--", lw=0.8)
        ax[2].text(p, P.max(), f"{p:.0f}Hz", fontsize=7, rotation=90, va="top")
    ax[2].set_title("AC accel power spectrum (resonance peaks)")
    ax[2].set_xlabel("Hz"); ax[2].set_ylabel("PSD")

    ax[3].scatter(tot_speed, vibg, s=4, alpha=0.3)
    ax[3].set_title(f"Vibration vs joint speed  (r = {cc_tot:+.2f})")
    ax[3].set_xlabel("Σ|joint speed| (deg/s)"); ax[3].set_ylabel("vibration RMS (m/s²)")

    fig.tight_layout()
    out = f"{run}/analysis.png"
    fig.savefig(out, dpi=110)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
