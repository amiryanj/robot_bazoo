"""Interactive 3D viewer: SO-101 wrist_flex bracket + ADXL345 cradle + board.
Drag to rotate, scroll to zoom.  Run:  python3 view_assembly.py [--cradle-only]
"""
import sys, struct, numpy as np
import matplotlib
matplotlib.use("TkAgg")           # interactive window
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

A = "/home/javad/workspace/lerobot_all/SO-ARM100/Simulation/SO101/assets/"

def load(fn, scale=1.0):
    with open(fn, "rb") as f:
        f.read(80); n = struct.unpack("<I", f.read(4))[0]
        T = np.empty((n, 3, 3), np.float32)
        for k in range(n):
            f.read(12)
            for v in range(3):
                T[k, v] = struct.unpack("<3f", f.read(12))
            f.read(2)
    return T * scale

cradle_only = "--cradle-only" in sys.argv
cr = load("adxl_wrist_cradle.stl")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

if cradle_only:
    ax.add_collection3d(Poly3DCollection(cr, alpha=1, facecolor="#5a8f3c",
                                         edgecolor="k", linewidths=0.2))
    allp = cr.reshape(-1, 3)
    ax.set_title("ADXL345 cradle  (drag=rotate, scroll=zoom)")
else:
    br = load(A + "wrist_roll_pitch_so101_v2.stl", 1000.0)
    bmn = br.reshape(-1, 3).min(0); bmx = br.reshape(-1, 3).max(0)
    ywall = bmx[1]; xc = (bmn[0] + bmx[0]) / 2; zc = (bmn[2] + bmx[2]) / 2
    R = cr.copy()                          # rotate +Z(back normal) -> +Y wall
    R[..., 1] = cr[..., 2]; R[..., 2] = -cr[..., 1]
    R[..., 0] += xc; R[..., 1] += ywall; R[..., 2] += zc
    ax.add_collection3d(Poly3DCollection(br, alpha=0.85, facecolor="#b9b9c2",
                                         edgecolor="none"))
    ax.add_collection3d(Poly3DCollection(R, alpha=1, facecolor="#5a8f3c",
                                         edgecolor="k", linewidths=0.2))
    allp = np.vstack([br.reshape(-1, 3), R.reshape(-1, 3)])
    ax.set_title("wrist_flex bracket (grey) + ADXL cradle (green)  "
                 "[cradle placement illustrative]")

mn = allp.min(0); mx = allp.max(0); c = (mn + mx) / 2; r = (mx - mn).max() / 2
for a, cc in zip("xyz", c):
    getattr(ax, f"set_{a}lim")(cc - r, cc + r)
try:
    ax.set_box_aspect((1, 1, 1))
except Exception:
    pass
ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
plt.tight_layout()
plt.show()
