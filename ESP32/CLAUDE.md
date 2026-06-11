# ESP32 Robot Project

## Hardware

| Component | Details |
|---|---|
| MCU (in use) | **ESP32-C3 Super Mini** — mounted on the arm's `wrist_roll`, streaming live |
| MCU (old dev) | ESP32-WROOM-32D (ESP32-D0WD-V3 rev 3.1) — bench prototype |
| IMU | ADXL345 (3-axis accelerometer, I2C) |
| USB-serial | C3: native USB-JTAG (`303a:1001`); WROOM: CH9102 (`1a86:55d3`) |

### ADXL345 Wiring (I2C)

**ESP32-WROOM-32D:**
```
ADXL345 VCC  → ESP32 3.3V
ADXL345 GND  → ESP32 GND
ADXL345 SDA  → ESP32 GPIO 21
ADXL345 SCL  → ESP32 GPIO 22
CS, INT1, INT2, SDO → leave unconnected for basic use
```

**ESP32-C3 Super Mini:**
```
ADXL345 VCC  → C3 3V3   (NOT 5V — ADXL is a 3.3V part)
ADXL345 GND  → C3 GND
ADXL345 SDA  → C3 GPIO 0
ADXL345 SCL  → C3 GPIO 1
```
C3 has a GPIO matrix — I2C maps to almost any pin via `Wire.begin(SDA, SCL)`.
Avoid GPIO 8 (onboard LED + strapping), 9 (BOOT/strapping), 2 (strapping),
11–17 (SPI flash), 18/19 (USB D-/D+).

### ESP32-C3 Super Mini notes
- Native USB-JTAG/serial (USB id `303a:1001`) — **no UART bridge chip**.
- `board = esp32-c3-devkitm-1`; needs `-DARDUINO_USB_MODE=1 -DARDUINO_USB_CDC_ON_BOOT=1`
  in `build_flags` for Serial over USB-CDC.
- Onboard LED on **GPIO 8, active LOW** (LOW=on).
- Serial port enumerates as **`/dev/ttyACM0`**.
- Working projects: `blink_c3/` (LED+serial), `imu_c3/` (ADXL345 800Hz FIFO stream).

---

## Toolchain

**PlatformIO** via pip in the `base` conda environment.

```bash
pip install platformio
```

### Standard platformio.ini for ESP32-WROOM-32D
```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino
monitor_speed = 460800
upload_speed = 115200
upload_port = /dev/ttyUSB0
monitor_port = /dev/ttyUSB0
```

---

## Serial Ports

- **`/dev/ttyACM0`** — ESP32-C3 Super Mini (native USB-JTAG, `303a:1001`) — the IMU
- **`/dev/ttyACM1`** — SO-101 arm (CH343, `1a86`). **Port collision:** the C3 and the
  arm both enumerate as `ttyACM*`; the C3 took `ttyACM0` so the arm moved to `ttyACM1`.
- **`/dev/ttyUSB0`** — old ESP32-WROOM-32D (CH9102 bridge), bench only

`imu_serial.default_port()` no longer assumes a number — it picks `$IMU_PORT` if set,
else scans `ttyACM*` for USB **vendor id `303a`** (the C3), else falls back to `ttyUSB0`.
So enumeration order doesn't matter. The arm side (`station.py`) defaults to `ttyACM1`
and takes `--port` to override.

To find which port is which: `for p in /dev/ttyACM*; do echo -n "$p "; udevadm info "$p" | grep ID_VENDOR_ID; done` — `303a` is the C3, `1a86` is the arm.

Make sure user is in `dialout` group:
```bash
sudo usermod -a -G dialout $USER
```

---

## Flashing

```bash
cd ~/workspace/ESP32/<project>
pio run --target upload
```

Auto-reset works on this board — no need to press BOOT manually.
If it fails once, just run again (board may have been mid-transmission).

**Do NOT use `upload_speed` above 115200** — higher speeds fail on this board.
**Serial monitor/plot must be closed** before flashing, or flash may fail.

---

## Serial Monitor

```bash
pio device monitor
```

Garbage on startup is normal — ESP32 bootloader runs at 74880 baud, switches to your baud rate once `setup()` runs.

---

## IMU — ADXL345

### Key registers
| Register | Address | Purpose |
|---|---|---|
| POWER_CTL | 0x2D | 0x00=standby, 0x08=measure |
| DATA_FORMAT | 0x31 | 0x0B = full-res ±16g (3.9mg/LSB) |
| BW_RATE | 0x2C | 0x0D=800Hz, 0x0A=100Hz |
| FIFO_CTL | 0x38 | 0x99 = stream mode, watermark=25 |
| FIFO_STATUS | 0x39 | bits[5:0] = samples in FIFO |
| DATAX0 | 0x32 | first of 6 bytes (x,y,z each 2 bytes) |

### Scale factor
```
FULL_RES mode: 3.9 mg/LSB → 0.038246 m/s²/LSB
```

### FIFO / 800Hz batch firmware
- I2C at 400kHz (`Wire.setClock(400000)`)
- Poll `FIFO_STATUS` in loop, flush when ≥ 25 samples
- Binary packet format: `[0xAA][0xBB][count:u8][t_micros:u32][x,y,z:i16 × count]`
- Serial at 460800 baud → ~5KB/s, well within capacity
- Batch arrives every ~31ms (25 samples @ 800Hz)

---

## Live Plot

```bash
python3 ~/workspace/ESP32/plot_imu.py
```

Requires: `pip install matplotlib pyserial` (already installed in base env)

- **M key** — toggle Mode 1 (1s rolling window) / Mode 2 (100ms zoom, fine detail)
- Close plot before flashing new firmware

---

## WiFi

```cpp
#include <WiFi.h>
WiFi.begin("Vive MJ", "<password>");
```

Board connects to local network and gets DHCP address. Tested — ping works.
Crystal is 26MHz (board reports 40MHz but 26MHz is correct per hardware).

---

## Host scripts

- `imu_serial.py` — shared binary-protocol parser (`stream_samples()`, `SCALE`,
  `default_port()`). Imported by everything below and by `../station.py`.
- `stream_imu_rerun.py` — IMU-only live stream to Rerun (no arm). For debugging the
  sensor by itself; `station.py` folds the same stream into the full cockpit.
- `log_imu.py` — raw IMU → CSV (`t_us, x_raw, y_raw, z_raw`).
- `plot_imu.py` — standalone matplotlib live plot.

The integrated path is `../station.py`: it reads this IMU in a background thread,
logs it to the shared Rerun timeline + `imu.csv`, time-synced with the motor stream.

## Future Work

- [x] Switch to ESP32-C3 Super Mini for final robot hardware (mounted on `wrist_roll`)
- [x] Integrate ADXL345 data alongside motor logging (`station.py`, synced CSVs)
- [x] Use the IMU to calibrate servo coefficients (`../calibrate.py` records it live
      per capture; tuned D=200 on the big joints 2026-06-11 — see root CLAUDE.md)
- [ ] Add gyroscope (ADXL345 is accelerometer only — consider ICM-42688 for full IMU)
- [ ] BLE remote control from phone (nRF Connect app for testing)
