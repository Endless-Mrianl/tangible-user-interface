#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

#define MAX_SERVOS 112  // 16 servos per PCA9685, up to 7 modules

// List of I2C addresses for each PCA9685 module
uint8_t pcaAddresses[] = {0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46};
const int numDrivers = sizeof(pcaAddresses) / sizeof(pcaAddresses[0]);

Adafruit_PWMServoDriver pwmDrivers[7];  // Supports up to 7 modules

const int SERVOMIN = 150;
const int SERVOMAX = 600;

void setup() {
  Serial.begin(9600);
  Wire.begin();  // SDA: GPIO21, SCL: GPIO22 on ESP32

  // Initialize all PCA9685 modules
  for (int i = 0; i < numDrivers; i++) {
    pwmDrivers[i] = Adafruit_PWMServoDriver(pcaAddresses[i]);
    pwmDrivers[i].begin();
    pwmDrivers[i].setPWMFreq(50);  // Set frequency to 50Hz
    delay(5);
  }

  Serial.println("Multi-PCA9685 Servo Controller Ready.");
}

int angleToPulse(int angle) {
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

void loop() {
  static String inputString = "";

  while (Serial.available()) {
    char inChar = Serial.read();

    if (inChar == '\n') {
      inputString.trim();
      if (inputString.length() > 0) {
        Serial.print("Received: ");
        Serial.println(inputString);

        // Split the input into angles
        int servoIndex = 0;
        int startIdx = 0;

        while (true) {
          int commaIndex = inputString.indexOf(',', startIdx);
          String angleStr = (commaIndex == -1)
              ? inputString.substring(startIdx)
              : inputString.substring(startIdx, commaIndex);

          angleStr.trim();

          if (angleStr.length() > 0) {
            int angle = constrain(angleStr.toInt(), 0, 180);
            int pulse = angleToPulse(angle);

            int driverIndex = servoIndex / 16;
            int channel = servoIndex % 16;

            if (driverIndex < numDrivers) {
              pwmDrivers[driverIndex].setPWM(channel, 0, pulse);
              Serial.printf("Servo %d -> Angle %d° (Driver %d, Channel %d)\n",
                            servoIndex, angle, driverIndex, channel);
            }

            servoIndex++;
            if (servoIndex >= MAX_SERVOS) break;
          }

          if (commaIndex == -1) break;
          startIdx = commaIndex + 1;
        }
      }

      inputString = "";
    } else {
      inputString += inChar;
    }
  }
}