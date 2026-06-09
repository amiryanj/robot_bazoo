#include <Arduino.h>
#include <Wire.h>

// ---- ADXL345 on ESP32-C3 Super Mini ----
// Wiring: VCC->3V3  GND->GND  SDA->GPIO0  SCL->GPIO1
static const int PIN_SDA = 0;
static const int PIN_SCL = 1;

static const uint8_t ADXL_ADDR = 0x53;   // ALT ADDRESS (SDO) low/unconnected

// Registers
static const uint8_t REG_DEVID       = 0x00;   // -> 0xE5
static const uint8_t REG_BW_RATE     = 0x2C;
static const uint8_t REG_POWER_CTL   = 0x2D;
static const uint8_t REG_DATA_FORMAT = 0x31;
static const uint8_t REG_DATAX0      = 0x32;
static const uint8_t REG_FIFO_CTL    = 0x38;
static const uint8_t REG_FIFO_STATUS = 0x39;

static const uint8_t WATERMARK = 25;          // flush when FIFO has >= this many

// Binary packet: [0xAA][0xBB][count:u8][t_micros:u32 LE][x,y,z : i16 LE x count]
static const uint8_t MAGIC0 = 0xAA;
static const uint8_t MAGIC1 = 0xBB;

static void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(ADXL_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

static uint8_t readReg(uint8_t reg) {
  Wire.beginTransmission(ADXL_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);            // repeated start
  Wire.requestFrom((int)ADXL_ADDR, 1);
  return Wire.available() ? Wire.read() : 0xFF;
}

// Read one 6-byte sample (x,y,z) from the FIFO output registers.
static void readSample(uint8_t *buf6) {
  Wire.beginTransmission(ADXL_ADDR);
  Wire.write(REG_DATAX0);
  Wire.endTransmission(false);
  Wire.requestFrom((int)ADXL_ADDR, 6);
  for (int i = 0; i < 6 && Wire.available(); i++) buf6[i] = Wire.read();
}

void setup() {
  Serial.begin(460800);
  Wire.begin(PIN_SDA, PIN_SCL);
  Wire.setClock(400000);

  // Wait for ADXL345 to answer with its known device id (0xE5).
  uint8_t devid = readReg(REG_DEVID);
  uint32_t t0 = millis();
  while (devid != 0xE5 && millis() - t0 < 3000) {
    delay(100);
    devid = readReg(REG_DEVID);
  }
  Serial.printf("# ADXL345 DEVID=0x%02X (expect 0xE5)\n", devid);

  // Configure: 800Hz, full-res +/-16g, stream FIFO with watermark, then measure.
  writeReg(REG_BW_RATE,     0x0D);   // 800 Hz output data rate
  writeReg(REG_DATA_FORMAT, 0x0B);   // full-res, +/-16g (3.9 mg/LSB)
  writeReg(REG_FIFO_CTL,    0x80 | (WATERMARK & 0x1F)); // stream mode, watermark
  writeReg(REG_POWER_CTL,   0x08);   // measure mode
  delay(10);
}

void loop() {
  uint8_t entries = readReg(REG_FIFO_STATUS) & 0x3F;
  if (entries < WATERMARK) return;

  if (entries > 32) entries = 32;            // FIFO is 32 deep
  uint8_t count = entries;

  static uint8_t pkt[7 + 32 * 6];
  pkt[0] = MAGIC0;
  pkt[1] = MAGIC1;
  pkt[2] = count;
  uint32_t t_us = micros();                  // timestamp of first sample
  memcpy(&pkt[3], &t_us, 4);

  uint8_t *p = &pkt[7];
  for (uint8_t i = 0; i < count; i++) {
    readSample(p);
    p += 6;
    delayMicroseconds(5);                    // ADXL345 needs >5us between FIFO reads
  }

  Serial.write(pkt, 7 + count * 6);
}
