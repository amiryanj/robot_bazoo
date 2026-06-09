import serial
import struct
import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
from imu_serial import default_port

PORT      = default_port()  # $IMU_PORT, else ttyACM0 (C3), else ttyUSB0 (WROOM)
BAUD      = 460800
ODR       = 800           # Hz
SCALE     = 0.038246      # m/s²/LSB  (3.9mg × 9.80665)
WATERMARK = 25
DT_US     = 1_000_000.0 / ODR  # ~1250 µs between samples

# mode 1: 1-second rolling window  |  mode 2: 100ms zoom (fine detail)
WINDOW = {1: 800, 2: 80}
mode = [1]

lock  = threading.Lock()
t_buf = deque(maxlen=max(WINDOW.values()))
x_buf = deque(maxlen=max(WINDOW.values()))
y_buf = deque(maxlen=max(WINDOW.values()))
z_buf = deque(maxlen=max(WINDOW.values()))

def reader(ser):
    while True:
        if ser.read(1) != b'\xaa':
            continue
        if ser.read(1) != b'\xbb':
            continue
        hdr = ser.read(5)
        if len(hdr) < 5:
            continue
        count = hdr[0]
        t0 = struct.unpack_from('<I', hdr, 1)[0]
        raw = ser.read(count * 6)
        if len(raw) < count * 6:
            continue
        with lock:
            for i in range(count):
                x, y, z = struct.unpack_from('<hhh', raw, i * 6)
                t_buf.append(t0 + i * DT_US)
                x_buf.append(x * SCALE)
                y_buf.append(y * SCALE)
                z_buf.append(z * SCALE)

ser = serial.Serial(PORT, BAUD, timeout=2)
threading.Thread(target=reader, args=(ser,), daemon=True).start()

fig, ax = plt.subplots(figsize=(11, 4))
lx, = ax.plot([], [], 'r',  lw=0.8, label='X')
ly, = ax.plot([], [], 'g',  lw=0.8, label='Y')
lz, = ax.plot([], [], 'b',  lw=0.8, label='Z')
ax.set_ylim(-20, 20)
ax.set_ylabel('m/s²')
ax.axhline(0, color='gray', lw=0.5)
ax.legend(loc='upper right')

def set_title():
    labels = {1: 'Mode 1 — live (1s window)', 2: 'Mode 2 — high-res (100ms window)'}
    ax.set_title(f'ADXL345 @ 800Hz  |  {labels[mode[0]]}  |  press M to toggle')

set_title()

def on_key(event):
    if event.key == 'm':
        mode[0] = 2 if mode[0] == 1 else 1
        set_title()

fig.canvas.mpl_connect('key_press_event', on_key)

def update(_):
    w = WINDOW[mode[0]]
    with lock:
        if len(t_buf) < 2:
            return lx, ly, lz
        ta = np.array(list(t_buf)[-w:])
        xa = np.array(list(x_buf)[-w:])
        ya = np.array(list(y_buf)[-w:])
        za = np.array(list(z_buf)[-w:])

    t_s = (ta - ta[0]) / 1e6
    ax.set_xlim(0, t_s[-1] if t_s[-1] > 0 else 0.001)
    ax.set_xlabel('seconds' if mode[0] == 1 else 'seconds (last 100ms)')
    lx.set_data(t_s, xa)
    ly.set_data(t_s, ya)
    lz.set_data(t_s, za)
    return lx, ly, lz

ani = animation.FuncAnimation(fig, update, interval=40, blit=True)
plt.tight_layout()
plt.show()
ser.close()
