const int pumpPin = 3; 

void setup() {
  pinMode(pumpPin, OUTPUT);
  Serial.begin(9600);
  Serial.println("Adafruit MOSFET Pump Control Ready.");
}

void loop() {

  Serial.println("Dropping to 25% flow rate...");
  analogWrite(pumpPin, 50); // Run continuously at a lower speed
  delay(5000);

}