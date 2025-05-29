#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(); // default I2C address is 0x40

#define SERVOMIN  150 // Minimum pulse length out of 4096
#define SERVOMAX  600 // Maximum pulse length out of 4096
#define NUM_SERVOS 9

String inputString = "";
bool inputComplete = false;

// Convert angle (0 to 180) to PWM pulse value
uint16_t angleToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22); // SDA, SCL for ESP32
  pwm.begin();
  pwm.setPWMFreq(50);  // 50Hz for analog servos
  delay(10);

  Serial.println("Enter 9 angles (0-180) separated by commas, e.g.: 0,30,60,90,120,150,180,90,45");
}

void loop() {
  if (Serial.available()) {
    char inChar = (char)Serial.read();

    if (inChar == '\n') {
      inputComplete = true;
    } else {
      inputString += inChar;
    }
  }

  if (inputComplete) {
    int angles[NUM_SERVOS];
    int index = 0;
    char *token = strtok((char *)inputString.c_str(), ",");

    while (token != NULL && index < NUM_SERVOS) {
      angles[index] = constrain(atoi(token), 0, 180); // Parse and constrain each angle
      token = strtok(NULL, ",");
      index++;
    }

    if (index == NUM_SERVOS) {
      for (int i = 0; i < NUM_SERVOS; i++) {
        int pulse = angleToPulse(angles[i]);
        pwm.setPWM(i, 0, pulse);
      }
      Serial.println("Angles applied to servos.");
    } else {
      Serial.println("Invalid input. Please enter 9 comma-separated angles (0-180).");
    }

    inputString = "";
    inputComplete = false;
  }
}
