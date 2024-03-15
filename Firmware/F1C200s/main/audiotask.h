#ifndef AUDIOTASK_H
#define AUDIOTASK_H
#include "thread.h"
#include "mutex.h"
#include "settings.h"

#define NONE        0
#define KOTI_AEC    1
#define QUAL_COMM   2
#define SPEEX_DSP   3
#define WEB_RTC     4
#define NNOM        5

#define DENOISE_METHOD WEB_RTC

#define SAVE_RAW_FILE 0
#define ENABLE_VAD  1
#define REC_PLAY_IN_A_THREAD    0
#define AUDIO_COMM  1

void    StartAudioTask();
void    StopAudioTask();
void    ConfigMix();
#endif // AUDIOTASK_H
