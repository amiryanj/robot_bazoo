"""Shared ADXL345 binary-protocol parser.

Packet format from firmware:
    [0xAA][0xBB][count:u8][t_micros:u32 LE][x,y,z : i16 LE  x count]

stream_samples() yields (t_us, x_raw, y_raw, z_raw) one sample at a time.
Raw counts are kept (not m/s²) so downstream code can re-scale/re-bias freely.
"""
import os
import struct
import serial

MAGIC0 = 0xAA
MAGIC1 = 0xBB

ODR    = 800                      # Hz, must match firmware BW_RATE
DT_US  = 1_000_000 / ODR          # ~1250 µs between samples
SCALE  = 0.038246                 # m/s² per LSB (3.9 mg * 9.80665)


def default_port():
    """Pick the serial port: $IMU_PORT, else find ESP32-C3 by USB vendor id, else ttyUSB0."""
    env = os.environ.get("IMU_PORT")
    if env:
        return env
    # Prefer identifying the C3 by its Espressif USB vendor id (303a:1001)
    # so the port survives enumeration order changes (e.g. arm also on ttyACM*).
    import glob
    import subprocess
    for p in sorted(glob.glob("/dev/ttyACM*")):
        try:
            out = subprocess.check_output(
                ["udevadm", "info", p], text=True, stderr=subprocess.DEVNULL)
            if "ID_VENDOR_ID=303a" in out:
                return p
        except Exception:
            pass
    # Fallback: WROOM on USB-UART bridge
    for p in ("/dev/ttyUSB0",):
        if os.path.exists(p):
            return p
    return "/dev/ttyUSB0"


def stream_samples(port=None, baud=460800):
    if port is None:
        port = default_port()
    ser = serial.Serial(port, baud, timeout=2)
    try:
        while True:
            b = ser.read(1)
            if not b or b[0] != MAGIC0:
                continue
            b = ser.read(1)
            if not b or b[0] != MAGIC1:
                continue
            hdr = ser.read(5)
            if len(hdr) < 5:
                continue
            count = hdr[0]
            t0 = struct.unpack_from('<I', hdr, 1)[0]
            raw = ser.read(count * 6)
            if len(raw) < count * 6:
                continue
            for i in range(count):
                x, y, z = struct.unpack_from('<hhh', raw, i * 6)
                yield (t0 + i * DT_US, x, y, z)
    finally:
        ser.close()
