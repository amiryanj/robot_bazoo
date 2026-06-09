"""Live 3D dead-reckoning trajectory with a fading tail.

Hold still for ~0.5s at startup to calibrate bias, then move.
Press R to reset the trajectory to the origin.

Params at the top are what you tune from analyze.py output.
ZUPT here uses rolling std of |acc| (bias-independent), matching analyze.py.
"""
import threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from imu_serial import stream_samples, SCALE

# --- tunables (from analyze.py) ---
ZUPT_WIN    = 20      # samples in stationary-detection window
ZUPT_THRESH = 0.35    # m/s²; rolling std of |acc| below this => stationary
                      # (from analyze.py run1: 4x noise floor of 0.087)
BIAS_N      = 400     # samples (~0.5s) held still at startup to calibrate

# --- display ---
TAIL_SECONDS = 4.0
ODR          = 800
TAIL_LEN     = int(TAIL_SECONDS * ODR)
DISPLAY_PTS  = 500    # decimated points actually drawn (keeps 3D smooth)

lock = threading.Lock()
traj = deque(maxlen=DISPLAY_PTS)     # decimated (x,y,z) positions for drawing

state = dict(vel=np.zeros(3), pos=np.zeros(3), last_t=None,
             bias=None, bias_acc=[], mag_win=deque(maxlen=ZUPT_WIN),
             sample_i=0)
DECIM = max(1, ODR // 120)           # append ~120 display pts/sec


def integrate(ax, ay, az, t_us):
    s = state
    raw = np.array([ax, ay, az])

    if s['bias'] is None:
        s['bias_acc'].append(raw)
        if len(s['bias_acc']) >= BIAS_N:
            s['bias'] = np.mean(s['bias_acc'], axis=0)
            print(f"Bias calibrated: {s['bias']}  |bias|={np.linalg.norm(s['bias']):.3f}")
        return

    s['mag_win'].append(np.linalg.norm(raw))   # raw magnitude => bias-independent

    if s['last_t'] is None:
        s['last_t'] = t_us
        return
    dt = (t_us - s['last_t']) / 1e6
    s['last_t'] = t_us
    if dt <= 0 or dt > 0.05:
        return

    # ZUPT on rolling std of raw |acc|
    if len(s['mag_win']) == ZUPT_WIN and np.std(s['mag_win']) < ZUPT_THRESH:
        s['vel'][:] = 0.0

    a = raw - s['bias']
    s['vel'] += a * dt
    s['pos'] += s['vel'] * dt

    s['sample_i'] += 1
    if s['sample_i'] % DECIM == 0:
        with lock:
            traj.append(s['pos'].copy())


def reader():
    for (t, x, y, z) in stream_samples():
        integrate(x * SCALE, y * SCALE, z * SCALE, t)


threading.Thread(target=reader, daemon=True).start()

fig = plt.figure(figsize=(9, 8))
ax3d = fig.add_subplot(111, projection="3d")
ax3d.set_xlabel("X (m)")
ax3d.set_ylabel("Y (m)")
ax3d.set_zlabel("Z (m)")
ax3d.set_title("ADXL345 dead-reckoning trajectory  (hold still to calibrate; R=reset)")

tail = Line3DCollection([], linewidths=2)
ax3d.add_collection3d(tail)
head, = ax3d.plot([], [], [], "o", color="red", markersize=6)
origin, = ax3d.plot([0], [0], [0], "x", color="black", markersize=8)


def on_key(event):
    if event.key == "r":
        with lock:
            traj.clear()
        state['vel'][:] = 0.0
        state['pos'][:] = 0.0
        print("trajectory reset")


fig.canvas.mpl_connect("key_press_event", on_key)


def set_cube(pts):
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = (mn + mx) / 2
    half = max((mx - mn).max(), 0.1) / 2 * 1.2
    ax3d.set_xlim(center[0] - half, center[0] + half)
    ax3d.set_ylim(center[1] - half, center[1] + half)
    ax3d.set_zlim(center[2] - half, center[2] + half)
    ax3d.set_box_aspect((1, 1, 1))


def update(_):
    with lock:
        if len(traj) < 2:
            return
        pts = np.array(traj)

    # build fading tail: segment alpha ramps from faint (old) to solid (new)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    n = len(segs)
    alphas = np.linspace(0.05, 1.0, n)
    colors = np.zeros((n, 4))
    colors[:, 2] = 1.0          # blue
    colors[:, 0] = np.linspace(0.0, 0.2, n)
    colors[:, 3] = alphas
    tail.set_segments(segs)
    tail.set_color(colors)

    head.set_data([pts[-1, 0]], [pts[-1, 1]])
    head.set_3d_properties([pts[-1, 2]])

    set_cube(pts)


ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()
