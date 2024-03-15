//Akash

#include "Audio.h"
#include "AudioFilterLMS.h"

// Initialize the attributes
void AudioFilterLMS::begin(float* mu, int lat) {
  lr = mu;
  n_coeffs = 256;
  n_hist = 512;
  latency = lat;
  for (int i = 0; i < n_coeffs; i++) {
    coeffs[i] = 0.0;
  }
  for (int i = 0; i < n_hist; i++) {
    hist[i] = 0.0;
  }
}

// Shuffle a new sample into the extended input vector
void AudioFilterLMS::shuffle_hist(float sample) {
  for (int i = n_hist - 1; i > 0; i--) {
    hist[i] = hist[i-1];
  }
  hist[0] = sample;
}

// "update" is automatically called by audio library every block cycle
void AudioFilterLMS::update(void) {
  // Define the audio blocks that are recieved and transmitted, then recieve the source and error blocks
  audio_block_t *src_block, *err_block, *out_block;
  src_block = receiveReadOnly(0);
  err_block = receiveReadOnly(1);
  if (!src_block || !err_block) {
    Serial.print("Did not recieve block.");
    return;
  }

  // Update the weight vector using the first error sample from the current block and the previous state of the input vector
  float err_sample = ((float)err_block->data[0]) / (float)32767;
  for (int i = 0; i < n_coeffs; i++) {
    coeffs[i] -= (*lr) * err_sample * hist[i+latency];
  }

  // Pass the current source block through the LMS filter to produce the output block
  out_block = allocate();
  float src_sample;
  float out_sample;
  for (int i = 0; i < AUDIO_BLOCK_SAMPLES; i++) {
    src_sample = ((float)src_block->data[i]) / ((float)32767);
    shuffle_hist(src_sample);
    out_sample = 0.0;
    for (int j = 0; j < n_coeffs; j++) {
      out_sample += coeffs[j] * hist[j];
    }
    out_block->data[i] = (int16_t)(out_sample * 32767);
  }

  // Transmit and release audio blocks
  transmit(out_block);
  release(out_block);
  release(src_block);
  release(err_block);
}
