"""Log raw ADXL345 samples to CSV.

Usage:
    python3 log_imu.py [output.csv] [duration_seconds]

If no output given, writes logs/imu_<timestamp>.csv
If no duration given, runs until Ctrl+C.

Suggested capture for tuning: do a sequence like
    [hold still 3s] -> [move +X 30cm, stop] -> [move back, stop] -> [hold still 3s]
so the analyzer can see clear rest/motion segments.
"""
import csv
import os
import sys
import time

from imu_serial import stream_samples

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "logs", f"imu_{int(time.time())}.csv")
    dur = float(sys.argv[2]) if len(sys.argv) > 2 else None

    os.makedirs(os.path.dirname(out), exist_ok=True)
    print(f"Logging to {out}" + (f" for {dur}s" if dur else " (Ctrl+C to stop)"))

    n = 0
    start = None
    f = open(out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t_us", "x_raw", "y_raw", "z_raw"])
    try:
        for (t, x, y, z) in stream_samples():
            w.writerow([f"{t:.0f}", x, y, z])
            n += 1
            if start is None:
                start = t
            if n % 800 == 0:
                print(f"\r{n} samples  ({(t-start)/1e6:5.1f}s)", end="", flush=True)
            if dur and (t - start) / 1e6 >= dur:
                break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        f.close()
        print(f"\nWrote {n} samples to {out}")


if __name__ == "__main__":
    main()
