const int irPin = A0;             // Analog pin connected to IR receiver
const int sampleInterval = 10;    // Sampling interval in microseconds
const int preBufferSize = 100;    // Number of samples to keep before the trigger
const int postBufferSize = 900;   // Number of samples to capture after the trigger
const int bufferSize = preBufferSize + postBufferSize;  // Total size of the buffer
const int thresholdDelta = 15;    // The minimum rise required to detect a rising edge

int buffer[bufferSize];           // Array to store pre-trigger and post-trigger data (now int)
int preBuffer[preBufferSize];     // Circular buffer for pre-trigger values (now int)
unsigned long startTime;
int index = 0;
int preIndex = 0;
bool triggered = false;
bool readyToSample = false;
int previousValue = 0;

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);  // Set a short timeout for reading serial input
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'G') {
      readyToSample = true;  // Set flag to start monitoring for the trigger
      triggered = false;     // Ensure that we are ready to detect a new trigger
      previousValue = analogRead(irPin);  // Initialize the previous value
    }
  }

  if (readyToSample) {
    int irValue = analogRead(irPin);

    // Store the current value in the pre-buffer (without scaling, storing full 0-1024 range)
    preBuffer[preIndex] = irValue;
    preIndex = (preIndex + 1) % preBufferSize;  // Move to the next position in the circular buffer

    // Check for a rising edge
    if (!triggered && (irValue - previousValue) > thresholdDelta) {
      triggered = true;  // Set the triggered flag
      readyToSample = false;  // Stop monitoring once the trigger is detected
      startSampling();  // Start sampling
    }

    // Update previous value
    previousValue = irValue;
  }
}

void startSampling() {
  // Clear previous data
  memset(buffer, 0, sizeof(buffer));

  // Copy pre-buffer contents to the main buffer
  for (int i = 0; i < preBufferSize; i++) {
    buffer[i] = preBuffer[(preIndex + i) % preBufferSize];
  }

  // Start sampling after the pre-buffer
  startTime = micros();
  index = preBufferSize;

  while (index < bufferSize) {
    if (micros() - startTime >= sampleInterval) {
      buffer[index] = analogRead(irPin);  // No scaling, keeping full 0-1024 range
      index++;
      startTime = micros();
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
  Serial.flush();  // Ensure all data is sent before continuing
}
