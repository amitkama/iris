#include <IRremote.h>

// Define the IR LED pin
#define IR_LED_PIN 3  // Change this to the pin where you connect the IR LED

// Create an IRsend object with the IR LED pin
IRsend irSender(IR_LED_PIN);

void setup() {
  Serial.begin(9600);
  delay(2000);  // Allow time to open the serial monitor

  Serial.println("IR Sender for Digit 1 on Samsung TV");
}

void loop() {
  // Send the Samsung TV Digit 1 IR code continuously
  sendSamsungTVDigit1Code();
}

void sendSamsungTVDigit1Code() {
  // Replace the following value with the code obtained from the Samsung TV remote for the "1" button
  unsigned long samsungDigit1Code = 0xFB040707;  // Placeholder for the actual IR code

  Serial.println("Sending Samsung TV Digit 1 Code");
  
  // Send the NEC protocol with a 38 kHz carrier frequency
  irSender.sendNEC(samsungDigit1Code, 32, 38);
  
  // Allow the repeat feature to work
  delay(100);
}
