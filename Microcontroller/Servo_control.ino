#include <ESP32Servo.h>

#define NUM_SERVOS 9
Servo servos[NUM_SERVOS];

// Define GPIO pins for each servo (change as needed)
int servoPins[NUM_SERVOS] = {2, 4, 5, 13, 14, 15, 18, 19, 21};

String inputBuffer = "";

void setup() {
  Serial.begin(9600);
  // Attach each servo to its respective pin
  for (int i = 0; i < NUM_SERVOS; i++) {
    servos[i].setPeriodHertz(50); // 50 Hz standard
    servos[i].attach(servoPins[i], 500, 2400);  // Min/max pulse width in µs
  }
  Serial.println("Ready to receive angles");
}

void loop() {
  // Read serial data
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      processInput(inputBuffer);
      inputBuffer = "";  // Clear after processing
    } else {
      inputBuffer += c;
    }
  }
}

void processInput(String data) {
  int angles[NUM_SERVOS];
  int index = 0;

  char inputArray[data.length() + 1];
  data.toCharArray(inputArray, data.length() + 1);

  char *token = strtok(inputArray, ",");
  while (token != NULL && index < NUM_SERVOS) {
    angles[index++] = atoi(token);
    token = strtok(NULL, ",");
  }

  // Move servos
  for (int i = 0; i < index; i++) {
    servos[i].write(angles[i]);
  }

  Serial.println("Angles updated");
}
