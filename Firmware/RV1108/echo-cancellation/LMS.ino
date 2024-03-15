#include "AudioStream.h"
#include "Audio.h"
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <SerialFlash.h>
#include "AudioFilterLMS.h"

AudioInputI2S            input_signal;
AudioFilterLMS           LMS;
AudioOutputI2S           output_signal;

AudioConnection          patchCord1(input_signal, 0, LMS, 0);
AudioConnection          patchCord2(input_signal, 1, LMS, 1);
AudioConnection          patchCord3(LMS, 0, output_signal, 0);
AudioControlSGTL5000     sgtl5000_1;

float mu = 0.005;
int vec_offset = 35;

void setup() {
  Serial.begin(9600);
  AudioMemory(24);

  pinMode(1,INPUT_PULLUP);
  
  sgtl5000_1.enable();
  sgtl5000_1.volume(0.8);
  sgtl5000_1.inputSelect(AUDIO_INPUT_LINEIN);
  sgtl5000_1.lineInLevel(15,15);
  sgtl5000_1.lineOutLevel(13);

  LMS.begin(&mu,vec_offset);
  Serial.print("Going!");
  Serial.print(AUDIO_BLOCK_SAMPLES);
}

void loop() {
  // Set learning rate to zero if the button is pressed
  if(digitalRead(1)) {
    mu = 0.0;
  }
}
