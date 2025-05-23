#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(); // default I2C address is 0x40

#define SERVOMIN  150 // Minimum pulse length out of 4096
#define SERVOMAX  600 // Maximum pulse length out of 4096

void setup() {
  Serial.begin(115200);

  Wire.begin(21, 22); // SDA, SCL for ESP32

  pwm.begin();
  pwm.setPWMFreq(50);  // Analog servos run at ~50 Hz

  delay(10);
}

void loop() {
  // Sweep servos back and forth
  for (int pulse = SERVOMIN; pulse <= SERVOMAX; pulse++) {
    pwm.setPWM(0, 0, pulse);  // Servo 1 on channel 0
    pwm.setPWM(1, 0, pulse);  // Servo 2 on channel 1
    pwm.setPWM(2, 0, pulse);  // Servo 3 on channel 2
    delay(5);
  }

  for (int pulse = SERVOMAX; pulse >= SERVOMIN; pulse--) {
    pwm.setPWM(0, 0, pulse);
    pwm.setPWM(1, 0, pulse);
    pwm.setPWM(2, 0, pulse);
    delay(5);
  }
}
