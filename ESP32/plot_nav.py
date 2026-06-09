import serial
import struct
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

PORT      = "/dev/ttyUSB0"
BAUD      = 460800
ODR       = 800
SCALE     = 0.038246       # m/s²/LSB
DT_US     = 1_000_000 / ODR

WINDOW       = 800         # samples shown (~1 second)
ZUPT_WIN     = 20          # samples used for stationary detection
ZUPT_THRESH  = 0.15        # m/s² std → below this = stationary → reset velocity
BIAS_N       = 400         # first 0.5s used for bias calibration (hold still!)

lock = threading.Lock()

t_buf  = deque(maxlen=WINDOW)
ax_buf = deque(maxlen=WINDOW)
ay_buf = deque(maxlen=WINDOW)
az_buf = deque(maxlen=WINDOW)
vx_buf = deque(maxlen=WINDOW)
vy_buf = deque(maxlen=WINDOW)
vz_buf = deque(maxlen=WINDOW)
px_buf = deque(maxlen=WINDOW)
py_buf = deque(maxlen=WINDOW)
pz_buf = deque(maxlen=WINDOW)
zupt_flag = deque(maxlen=WINDOW)  # 1 where ZUPT fired

state = dict(
    vx=0.0, vy=0.0, vz=0.0,
    px=0.0, py=0.0, pz=0.0,
    last_t=None,
    bias=None,
    bias_acc=[],
)

def integrate(ax, ay, az, t_us):
    s = state

    # --- bias calibration ---
    if s['bias'] is None:
        s['bias_acc'].append((ax, ay, az))
        if len(s['bias_acc']) >= BIAS_N:
            # mean during stillness = gravity_in_sensor_frame + sensor_bias
            # subtract it all — works at any orientation, not just flat
            s['bias'] = np.mean(s['bias_acc'], axis=0)
            print(f"Bias calibrated: {s['bias']}")
        # zero-fill until calibrated
        t_buf.append(t_us);  ax_buf.append(ax); ay_buf.append(ay); az_buf.append(az)
        vx_buf.append(0); vy_buf.append(0); vz_buf.append(0)
        px_buf.append(0); py_buf.append(0); pz_buf.append(0)
        zupt_flag.append(0)
        return

    ax -= s['bias'][0]
    ay -= s['bias'][1]
    az -= s['bias'][2]

    # --- dt ---
    if s['last_t'] is None:
        s['last_t'] = t_us
        return
    dt = (t_us - s['last_t']) / 1e6
    s['last_t'] = t_us
    if dt <= 0 or dt > 0.05:       # skip bad gaps (micros overflow etc.)
        return

    # --- ZUPT ---
    fired = 0
    if len(ax_buf) >= ZUPT_WIN:
        mag = np.sqrt(
            np.array(list(ax_buf)[-ZUPT_WIN:])**2 +
            np.array(list(ay_buf)[-ZUPT_WIN:])**2 +
            np.array(list(az_buf)[-ZUPT_WIN:])**2
        )
        if np.std(mag) < ZUPT_THRESH:
            s['vx'] = s['vy'] = s['vz'] = 0.0
            fired = 1

    # --- integrate ---
    s['vx'] += ax * dt;  s['vy'] += ay * dt;  s['vz'] += az * dt
    s['px'] += s['vx'] * dt;  s['py'] += s['vy'] * dt;  s['pz'] += s['vz'] * dt

    t_buf.append(t_us)
    ax_buf.append(ax);  ay_buf.append(ay);  az_buf.append(az)
    vx_buf.append(s['vx']); vy_buf.append(s['vy']); vz_buf.append(s['vz'])
    px_buf.append(s['px']); py_buf.append(s['py']); pz_buf.append(s['pz'])
    zupt_flag.append(fired)


def reader(ser):
    while True:
        if ser.read(1) != b'\xaa': continue
        if ser.read(1) != b'\xbb': continue
        hdr = ser.read(5)
        if len(hdr) < 5: continue
        count = hdr[0]
        t0 = struct.unpack_from('<I', hdr, 1)[0]
        raw = ser.read(count * 6)
        if len(raw) < count * 6: continue
        with lock:
            for i in range(count):
                x, y, z = struct.unpack_from('<hhh', raw, i * 6)
                integrate(x * SCALE, y * SCALE, z * SCALE, t0 + i * DT_US)


ser = serial.Serial(PORT, BAUD, timeout=2)
threading.Thread(target=reader, args=(ser,), daemon=True).start()

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
fig.suptitle('ADXL345 Dead Reckoning  —  hold still 0.5s at ANY orientation to calibrate', fontsize=10)

a_ax, v_ax, p_ax = axes
for subplot, ylabel, title in [
    (a_ax, 'm/s²',  'Acceleration (bias removed)'),
    (v_ax, 'm/s',   'Velocity  (ZUPT applied — gray = ZUPT fired)'),
    (p_ax, 'm',     'Position  (drifts without stops — expected)'),
]:
    subplot.set_ylabel(ylabel)
    subplot.set_title(title, fontsize=9)
    subplot.axhline(0, color='gray', lw=0.4)

p_ax.set_xlabel('seconds')

la = [a_ax.plot([], [], c, lw=0.7, label=l)[0] for c, l in [('r','X'),('g','Y'),('b','Z')]]
lv = [v_ax.plot([], [], c, lw=0.7, label=l)[0] for c, l in [('r','X'),('g','Y'),('b','Z')]]
lp = [p_ax.plot([], [], c, lw=0.7, label=l)[0] for c, l in [('r','X'),('g','Y'),('b','Z')]]
zupt_line = v_ax.axvline(0, color='gray', lw=0.4, alpha=0.0)  # placeholder

for ax in axes:
    ax.legend(loc='upper right', fontsize=8)

def update(_):
    with lock:
        if len(t_buf) < 2:
            return
        t  = np.array(t_buf)
        ax = np.array(ax_buf); ay = np.array(ay_buf); az = np.array(az_buf)
        vx = np.array(vx_buf); vy = np.array(vy_buf); vz = np.array(vz_buf)
        px = np.array(px_buf); py = np.array(py_buf); pz = np.array(pz_buf)
        zf = np.array(zupt_flag)

    t_s = (t - t[0]) / 1e6
    xlim = (0, max(t_s[-1], 0.01))
    for subplot in axes:
        subplot.set_xlim(*xlim)

    la[0].set_data(t_s, ax); la[1].set_data(t_s, ay); la[2].set_data(t_s, az)
    lv[0].set_data(t_s, vx); lv[1].set_data(t_s, vy); lv[2].set_data(t_s, vz)
    lp[0].set_data(t_s, px); lp[1].set_data(t_s, py); lp[2].set_data(t_s, pz)

    # autoscale y with some padding
    for subplot, arrays in [(a_ax, [ax,ay,az]), (v_ax, [vx,vy,vz]), (p_ax, [px,py,pz])]:
        all_vals = np.concatenate(arrays)
        hi = max(abs(all_vals).max(), 0.01) * 1.3
        subplot.set_ylim(-hi, hi)

    # shade ZUPT regions on velocity plot
    for coll in list(v_ax.collections):
        coll.remove()
    if zf.any():
        v_ax.fill_between(t_s, -100, 100, where=zf.astype(bool),
                          color='gray', alpha=0.15, transform=v_ax.transData)

ani = animation.FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
plt.tight_layout()
plt.show()
ser.close()
