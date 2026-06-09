"""Offline analysis of a logged IMU CSV to tune dead-reckoning params.

Usage:
    python3 analyze.py logs/imu_xxx.csv

What it does:
  1. Sanity-checks sample timing (dt jitter, dropped samples).
  2. Computes rolling std of acceleration magnitude (the ZUPT feature).
  3. Finds a rest/motion split automatically (Otsu in log space) and
     suggests ZUPT_THRESH.
  4. Reports per-axis noise floor and a bias vector from the longest rest
     segment.
  5. Simulates integration at several ZUPT thresholds and reports residual
     velocity drift, so you can see the effect of the knob.
  6. Plots the diagnostics.

Tunables it informs:  ZUPT_THRESH, ZUPT_WIN, BIAS_N
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

from imu_serial import SCALE, DT_US

ZUPT_WIN = 20   # samples in the rolling window (must match the live script)


def load(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    t = data["t_us"].astype(np.float64)
    acc = np.stack([data["x_raw"], data["y_raw"], data["z_raw"]], axis=1) * SCALE
    return t, acc


def rolling_std_mag(acc, win):
    """Rolling std of |acc| — bias-independent stationary feature."""
    mag = np.linalg.norm(acc, axis=1)
    n = len(mag)
    out = np.full(n, np.nan)
    if n < win:
        return mag, out
    # cumulative-sum trick for windowed mean / mean-of-squares
    c1 = np.cumsum(np.insert(mag, 0, 0.0))
    c2 = np.cumsum(np.insert(mag**2, 0, 0.0))
    m1 = (c1[win:] - c1[:-win]) / win
    m2 = (c2[win:] - c2[:-win]) / win
    var = np.maximum(m2 - m1**2, 0.0)
    out[win - 1:] = np.sqrt(var)
    return mag, out


def otsu(values, nbins=160):
    """Otsu threshold in log space (distribution spans orders of magnitude)."""
    v = values[np.isfinite(values)]
    v = v[v > 0]
    if len(v) < 10:
        return None
    lv = np.log10(v)
    hist, edges = np.histogram(lv, bins=nbins)
    hist = hist.astype(float)
    total = hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(hist)
    mu = np.cumsum(hist * centers)
    muT = mu[-1]
    wF = total - w
    with np.errstate(invalid="ignore", divide="ignore"):
        mB = mu / w
        mF = (muT - mu) / wF
        sigma_b = w * wF * (mB - mF) ** 2
    idx = int(np.nanargmax(sigma_b))
    return 10 ** centers[idx]


def longest_rest_segment(stationary):
    """Return (start, end) indices of the longest True run in a bool array."""
    best_len = best_start = 0
    cur_start = None
    for i, s in enumerate(stationary):
        if s and cur_start is None:
            cur_start = i
        elif not s and cur_start is not None:
            if i - cur_start > best_len:
                best_len, best_start = i - cur_start, cur_start
            cur_start = None
    if cur_start is not None and len(stationary) - cur_start > best_len:
        best_len, best_start = len(stationary) - cur_start, cur_start
    return best_start, best_start + best_len


def simulate(t, acc, bias, thresh, win):
    """Integrate with ZUPT at `thresh`; return v, p, and residual drift stats."""
    a = acc - bias
    _, rstd = rolling_std_mag(acc, win)
    v = np.zeros_like(a)
    p = np.zeros_like(a)
    vel = np.zeros(3)
    pos = np.zeros(3)
    last_t = t[0]
    residual_speeds = []
    for i in range(1, len(t)):
        dt = (t[i] - last_t) / 1e6
        last_t = t[i]
        if dt <= 0 or dt > 0.05:
            v[i] = vel
            p[i] = pos
            continue
        if np.isfinite(rstd[i]) and rstd[i] < thresh:
            # speed accumulated just before being zeroed = drift indicator
            residual_speeds.append(np.linalg.norm(vel))
            vel[:] = 0.0
        vel += a[i] * dt
        pos += vel * dt
        v[i] = vel
        p[i] = pos
    drift = np.array(residual_speeds) if residual_speeds else np.array([np.nan])
    return v, p, drift


def main():
    if len(sys.argv) < 2:
        print("usage: python3 analyze.py <log.csv>")
        sys.exit(1)
    path = sys.argv[1]
    t, acc = load(path)
    n = len(t)
    dur = (t[-1] - t[0]) / 1e6
    print(f"\n=== {path} ===")
    print(f"samples: {n}   duration: {dur:.2f}s   mean rate: {n/dur:.1f} Hz")

    # --- timing sanity ---
    dt = np.diff(t)
    print(f"\n[timing]  expected dt = {DT_US:.0f} µs")
    print(f"  dt  median={np.median(dt):.0f}  mean={dt.mean():.0f}  "
          f"std={dt.std():.0f}  min={dt.min():.0f}  max={dt.max():.0f} µs")
    gaps = np.sum(dt > 2 * DT_US)
    print(f"  batch-boundary gaps (>2x dt): {gaps}")

    # --- ZUPT feature + threshold suggestion ---
    mag, rstd = rolling_std_mag(acc, ZUPT_WIN)
    finite = rstd[np.isfinite(rstd)]
    noise_floor = float(np.median(finite))             # robust rest-noise level
    motion_peak = float(np.percentile(finite, 99.5))   # ignore tap outliers
    thr_otsu = otsu(rstd)
    thr = noise_floor * 4.0                             # recommended: 4x margin
    sep = motion_peak / noise_floor if noise_floor > 0 else 0
    print(f"\n[ZUPT feature]  rolling std of |acc|, window={ZUPT_WIN} samples")
    print(f"  rest-noise floor (median) = {noise_floor:.3f} m/s²")
    print(f"  motion level (99.5 pct)   = {motion_peak:.3f} m/s²")
    print(f"  separation ratio          = {sep:.1f}x  "
          f"({'good' if sep > 4 else 'WEAK — make bigger, clearer moves'})")
    print(f"  Otsu split (rest vs rare spikes) = {thr_otsu:.3f} m/s²")
    print(f"  --> recommended ZUPT_THRESH = {thr:.3f} m/s²  (4x noise floor)")

    stationary = np.isfinite(rstd) & (rstd < thr)
    rest_frac = stationary.mean()

    # --- bias + per-axis noise from TIGHT rest only (not the loose mask) ---
    tight = np.isfinite(rstd) & (rstd < noise_floor * 1.5)
    s0, s1 = longest_rest_segment(tight)
    seg = acc[s0:s1]
    if len(seg) > 5:
        bias = seg.mean(axis=0)
        noise = seg.std(axis=0)
        print(f"\n[bias from tightest rest]  idx {s0}..{s1}  "
              f"({(t[s1-1]-t[s0])/1e6:.2f}s, {len(seg)} samples)")
        print(f"  bias  (grav+offset) = [{bias[0]:+.3f} {bias[1]:+.3f} {bias[2]:+.3f}] m/s²")
        print(f"  per-axis noise (1 sigma) = [{noise[0]:.3f} {noise[1]:.3f} {noise[2]:.3f}] m/s²")
        print(f"  |bias| = {np.linalg.norm(bias):.3f} m/s²  (should be ~9.81 if truly still)")
    else:
        bias = acc.mean(axis=0)
        print("\n[bias] no tight rest found; using whole-log mean as bias")

    # --- threshold sweep: effect on drift ---
    print(f"\n[threshold sweep]  residual velocity right before each ZUPT reset")
    print(f"  (lower = ZUPT catching rest sooner / less accumulated drift)")
    sweep = [thr * f for f in (0.5, 0.75, 1.0, 1.5, 2.0)]
    sim_cache = {}
    for th in sweep:
        v, p, drift = simulate(t, acc, bias, th, ZUPT_WIN)
        sim_cache[th] = (v, p)
        frac = (np.isfinite(rstd) & (rstd < th)).mean()
        print(f"  thresh={th:5.3f}  rest={frac*100:3.0f}%  "
              f"drift mean={np.nanmean(drift):.3f}  max={np.nanmax(drift):.3f} m/s")

    # --- plots ---
    v_sel, p_sel = sim_cache[thr]
    t_s = (t - t[0]) / 1e6

    fig, ax = plt.subplots(4, 1, figsize=(12, 11), sharex=False)

    ax[0].plot(t_s, mag, lw=0.5, color="purple")
    ax[0].set_ylabel("|acc| m/s²")
    ax[0].set_title(f"{path}  —  acceleration magnitude")

    ax[1].plot(t_s, rstd, lw=0.6, color="teal")
    ax[1].axhline(thr, color="red", lw=1, ls="--", label=f"ZUPT_THRESH={thr:.3f}")
    ax[1].fill_between(t_s, 0, rstd.max() if np.isfinite(rstd).any() else 1,
                       where=stationary, color="gray", alpha=0.2, label="detected rest")
    ax[1].set_ylabel("rolling std")
    ax[1].set_yscale("log")
    ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_title("ZUPT feature + auto threshold")

    finite = rstd[np.isfinite(rstd) & (rstd > 0)]
    pos_finite = finite[finite > 0]
    ax[2].hist(np.log10(pos_finite), bins=120, color="teal", alpha=0.7)
    ax[2].axvline(np.log10(noise_floor), color="green", ls=":", label="noise floor")
    ax[2].axvline(np.log10(thr), color="red", ls="--", label="ZUPT_THRESH (4x)")
    ax[2].set_xlabel("log10(rolling std)")
    ax[2].set_ylabel("count")
    ax[2].legend(fontsize=8)
    ax[2].set_title("rest/motion separation (bimodal = good)")

    for i, c, lab in [(0, "r", "X"), (1, "g", "Y"), (2, "b", "Z")]:
        ax[3].plot(t_s, v_sel[:, i], c, lw=0.7, label=f"v{lab}")
    ax[3].fill_between(t_s, v_sel.min(), v_sel.max(),
                       where=stationary, color="gray", alpha=0.15)
    ax[3].set_ylabel("velocity m/s")
    ax[3].set_xlabel("seconds")
    ax[3].legend(loc="upper right", fontsize=8)
    ax[3].set_title(f"simulated velocity @ ZUPT_THRESH={thr:.3f} (gray = rest)")

    plt.tight_layout()
    out_png = path.rsplit(".", 1)[0] + "_analysis.png"
    plt.savefig(out_png, dpi=90)
    print(f"\nsaved plot -> {out_png}")
    print("\nApply the suggested values to plot_3d.py / plot_nav.py:")
    print(f"  ZUPT_THRESH = {thr:.3f}")
    print(f"  ZUPT_WIN    = {ZUPT_WIN}")
    plt.show()


if __name__ == "__main__":
    main()
