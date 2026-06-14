#!/usr/bin/env python
"""Analyze a ball-cal sweep (vision/ball_cal_teleop.py): fit the constant ball<->finger
geometry, reject detection outliers, and report the TRUE residual + a workspace map.

The ball is rigid to the gripper, so in the GRIPPER frame both
  detected_ball - tag_midpoint   and   detected_ball - FK_TCP
should be a constant; whatever is left after subtracting that constant is the residual
detection/calibration error. Outliers (gross misdetections) are rejected by distance from
the robust median.

    python vision/ball_cal_analyze.py [<run_dir or samples.json>]   # default: latest run
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "vision"))
CALIB = ROOT / "outputs/calib"


def load(arg):
    if arg:
        p = Path(arg)
        p = p / "samples.json" if p.is_dir() else p
    else:                                              # latest run folder, else flat json
        runs = sorted(CALIB.glob("ball_cal_*/samples.json"))
        flat = sorted(CALIB.glob("ball_cal_*.json"))
        p = runs[-1] if runs else (flat[-1] if flat else None)
    if not p or not p.exists():
        sys.exit("no ball-cal samples found.")
    print(f"loading {p}")
    return json.load(open(p))["records"]


def main():
    from pick_ball import Kin
    recs = load(sys.argv[1] if len(sys.argv) > 1 else None)
    kin = Kin()

    base, grip, tcp = [], [], []        # ball-tagmid (base), same rotated to gripper, TCP
    for r in recs:
        if not (r.get("ball") and len(r.get("tags", {})) == 2):
            continue
        ball = np.array(r["ball"]["base3d"])
        mid = (np.array(r["tags"]["1"]["base3d"]) + np.array(r["tags"]["2"]["base3d"])) / 2
        R, t = kin.fk(r["angles"])
        base.append(ball - mid)
        grip.append(R.T @ (ball - mid))                # gripper frame: orientation-invariant
        tcp.append(t)
    base, grip, tcp = np.array(base) * 1000, np.array(grip) * 1000, np.array(tcp)
    n = len(base)
    print(f"{n} complete samples (2 tags + ball)")

    # robust outlier rejection: distance from the per-axis median (gripper frame)
    med = np.median(grip, axis=0)
    dist = np.linalg.norm(grip - med, axis=1)
    mad = np.median(np.abs(dist - np.median(dist))) + 1e-6
    inl = dist < max(5 * 1.4826 * mad, 15.0)           # >~15mm from median = misdetection
    print(f"  inliers {inl.sum()}/{n}; rejected: "
          f"{[i for i, k in enumerate(inl) if not k]} (0-based, complete-sample order)")

    g = grip[inl]
    const = g.mean(axis=0)
    resid = g - const
    rms = np.sqrt((resid ** 2).mean(axis=0))
    print(f"\nconstant ball - tag_midpoint (gripper frame): {np.round(const, 1)} mm "
          f"(|{np.linalg.norm(const):.1f}|)")
    print(f"residual RMS (after removing the constant):    {np.round(rms, 2)} mm  "
          f"(3D {np.sqrt((resid**2).sum(1).mean()):.2f} mm)")
    print(f"residual range per axis: "
          f"{np.round(resid.min(0), 1)} .. {np.round(resid.max(0), 1)} mm")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rmag = np.linalg.norm(resid, axis=1)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        sc = ax[0].scatter(tcp[inl][:, 0] * 1000, tcp[inl][:, 1] * 1000, c=rmag,
                           cmap="viridis", s=70)
        ax[0].set_xlabel("TCP x (mm)"); ax[0].set_ylabel("TCP y (mm)")
        ax[0].set_title("residual magnitude vs workspace"); ax[0].axis("equal")
        fig.colorbar(sc, ax=ax[0], label="residual (mm)")
        for k, lbl in enumerate("xyz"):
            ax[1].plot(resid[:, k], "o-", label=f"{lbl} (RMS {rms[k]:.1f})", ms=4)
        ax[1].axhline(0, color="k", lw=0.5); ax[1].legend(); ax[1].set_xlabel("inlier #")
        ax[1].set_ylabel("residual (mm)"); ax[1].set_title("residual per axis (gripper frame)")
        out = CALIB / "ball_cal_residual.png"
        fig.tight_layout(); fig.savefig(out, dpi=110)
        print(f"\nplot -> {out}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
