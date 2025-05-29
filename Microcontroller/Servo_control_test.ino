#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(); // default I2C address is 0x40

#define SERVOMIN  150 // Minimum pulse length out of 4096
#define SERVOMAX  600 // Maximum pulse length out of 4096

// Number of servos
#define NUM_SERVOS 9

// Function to convert angle (0 to 180) to pulse length
uint16_t angleToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // SDA, SCL for ESP32
  pwm.begin();
  pwm.setPWMFreq(50);  // Analog servos run at ~50 Hz
  delay(10);
}

void loop() {
  // Example: Set angles for 9 servos
  int servoAngles[NUM_SERVOS] = {0, 30, 60, 90, 120, 150, 180, 90, 45};

  // Apply angles to each servo
  for (int i = 0; i < NUM_SERVOS; i++) {
    int pulse = angleToPulse(servoAngles[i]);
    pwm.setPWM(i, 0, pulse);
  }

  delay(2000); // Wait for 2 seconds

  // Example: Change angles again (just a test sweep)
  int servoAngles2[NUM_SERVOS] = {180, 150, 120, 90, 60, 30, 0, 90, 135};
  for (int i = 0; i < NUM_SERVOS; i++) {
    int pulse = angleToPulse(servoAngles2[i]);
    pwm.setPWM(i, 0, pulse);
  }

  delay(2000); // Wait again before repeating
}
