"""Stream ADXL345 IMU data to Rerun alongside motor diagnostics.

Connects to the same Rerun viewer as diagnose_motors.py / command_log.py.
Logs under imu/wrist_roll/{accel_x,y,z,magnitude} in m/s².

Usage:
    # Terminal 1 — motor stream (or diagnose_motors.py)
    python command_log.py

    # Terminal 2 — IMU stream (this script)
    python ESP32/stream_imu_rerun.py

Both connect to the same Rerun viewer; all channels appear together.
"""
import sys
import os
import numpy as np
import rerun as rr

sys.path.insert(0, os.path.dirname(__file__))
from imu_serial import stream_samples, SCALE

ENTITY = "imu/wrist_roll"


def main():
    rr.init("so101_imu")
    rr.connect_grpc()
    print(f"Streaming IMU → Rerun  (entity: {ENTITY})  Ctrl+C to stop")

    t_start = None
    for (t_us, x_raw, y_raw, z_raw) in stream_samples():
        if t_start is None:
            t_start = t_us

        t_s = (t_us - t_start) / 1e6
        x = x_raw * SCALE
        y = y_raw * SCALE
        z = z_raw * SCALE
        mag = float(np.sqrt(x*x + y*y + z*z))

        rr.set_time("time", duration=t_s)
        rr.log(f"{ENTITY}/accel_x",   rr.Scalars(x))
        rr.log(f"{ENTITY}/accel_y",   rr.Scalars(y))
        rr.log(f"{ENTITY}/accel_z",   rr.Scalars(z))
        rr.log(f"{ENTITY}/magnitude", rr.Scalars(mag))


if __name__ == "__main__":
    main()
