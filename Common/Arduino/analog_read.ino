const int irPin = A0;       // Analog pin connected to IR receiver
const int sampleInterval = 50;  // Sampling interval in microseconds
const int bufferSize = 1000; // Size of the buffer to store raw data

int buffer[bufferSize];     // Array to store raw data
unsigned long startTime;
int index = 0;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);  // Set a short timeout for reading serial input
}

void loop() {
  // Wait for a signal from the Python script
  while (!Serial.available());
  char signal = Serial.read();
  if (signal == 'G') {
    startTime = micros();
    index = 0;

    // Sample the IR receiver output and fill the buffer
    while (index < bufferSize) {
      if (micros() - startTime >= sampleInterval) {
        buffer[index] = analogRead(irPin);
        index++;
        startTime += sampleInterval;
      }
    }

    // Print the buffer contents to the serial port
    for (int i = 0; i < bufferSize; i++) {
      Serial.print(buffer[i]);
      if (i < bufferSize - 1) {
        Serial.print(", ");
      }
    }
    Serial.println();
  }
}
