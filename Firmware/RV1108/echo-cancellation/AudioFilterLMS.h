//Akash

#ifndef AudioFilterLMS_h
#define AudioFilterLMS_h

#include "Arduino.h"
#include "AudioStream.h"

class AudioFilterLMS : public AudioStream {

private:
  audio_block_t* inputQueueArray[2];
  int n_coeffs;
  int n_hist;
  float coeffs[256];
  float hist[512];
  float *lr;
  float err_sample;
  int latency;

public:
  AudioFilterLMS(void) : AudioStream(2, inputQueueArray) {}
  void begin(float* mu, int lat);
  void shuffle_hist(float sample);
  virtual void update(void);

};

#endif
