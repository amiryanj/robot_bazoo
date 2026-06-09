#include <Arduino.h>

// ESP32-C3 Super Mini onboard LED is on GPIO 8 and is ACTIVE LOW
// (LOW = on, HIGH = off).
const int LED_PIN = 8;

void setup() {
  Serial.begin(115200);
  pinMode(LED_PIN, OUTPUT);
}

void loop() {
  digitalWrite(LED_PIN, LOW);   // on
  Serial.println("blink: on");
  delay(500);

  digitalWrite(LED_PIN, HIGH);  // off
  Serial.println("blink: off");
  delay(500);
}
