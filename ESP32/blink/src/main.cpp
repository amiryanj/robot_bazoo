#include <Arduino.h>
#include <Wire.h>

#define ADXL345_ADDR    0x53
#define REG_POWER_CTL   0x2D
#define REG_DATA_FORMAT 0x31
#define REG_BW_RATE     0x2C
#define REG_FIFO_CTL    0x38
#define REG_FIFO_STATUS 0x39
#define REG_DATAX0      0x32

#define WATERMARK 25

static void writeReg(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

static uint8_t readReg(uint8_t reg) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom((uint8_t)ADXL345_ADDR, (uint8_t)1);
  return Wire.read();
}

static void sendBatch() {
  uint32_t t = micros();  // timestamp of first sample in batch
  uint8_t buf[6 * WATERMARK];
  for (int i = 0; i < WATERMARK; i++) {
    Wire.beginTransmission(ADXL345_ADDR);
    Wire.write(REG_DATAX0);
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)ADXL345_ADDR, (uint8_t)6);
    for (int j = 0; j < 6; j++) buf[i * 6 + j] = Wire.read();
  }
  // packet: [0xAA][0xBB][count:u8][t_micros:u32][x,y,z:i16 * count]
  Serial.write(0xAA);
  Serial.write(0xBB);
  Serial.write((uint8_t)WATERMARK);
  Serial.write((uint8_t*)&t, 4);
  Serial.write(buf, sizeof(buf));
}

void setup() {
  Serial.begin(460800);
  Wire.begin(21, 22);
  Wire.setClock(400000);  // 400kHz fast mode

  writeReg(REG_POWER_CTL,   0x00);  // standby before config
  writeReg(REG_DATA_FORMAT, 0x0B);  // full-res, ±16g, 3.9mg/LSB
  writeReg(REG_BW_RATE,     0x0D);  // 800Hz ODR
  writeReg(REG_FIFO_CTL,    0x99);  // stream mode, watermark=25
  writeReg(REG_POWER_CTL,   0x08);  // measure
}

void loop() {
  if ((readReg(REG_FIFO_STATUS) & 0x3F) >= WATERMARK)
    sendBatch();
}
