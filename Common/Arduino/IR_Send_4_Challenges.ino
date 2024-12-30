#include <IRremote.h>
#define IR_LED_PIN 3

IRsend irSender(IR_LED_PIN);

void setup() {
  Serial.begin(9600);
  delay(2000);  // Time to open the serial monitor
  Serial.println("IR Sender for Digit 1 on Samsung TV");
}

void loop() {
  // Send the Samsung TV Digit 1 IR code continuously
  sendSamsungTVDigit1Code();
}

void sendSamsungTVDigit1Code() {
  unsigned long samsungDigit1Code = 0xFB040707;  // Digit 1 IR code
  // unsigned long samsungDigit1Code = 0xFA050707;  // Digit 2 IR code
  // unsigned long samsungDigit1Code = 0xF9060707;  // Digit 3 IR code
  // unsigned long samsungDigit1Code = 0xF7080707;  // Digit 4 IR code
  Serial.println("Sending Samsung TV Digit 1 Code");
  irSender.sendNEC(samsungDigit1Code, 32, 38);
  delay(100);
}
