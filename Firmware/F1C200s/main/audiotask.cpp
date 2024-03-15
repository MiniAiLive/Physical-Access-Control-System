#include "audiotask.h"
#include "msg.h"
#include "drv_gpio.h"
#include "shared.h"
#include "appdef.h"
#include "i2cbase.h"
#include "pcm_base.h"
#include "uartbase.h"
#include "uartcomm.h"
#include "termios.h"
#include "g711_table.h"

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>

#define DELAY_SAMPLE_COUNT 32
#define BATCH_SAMPLE_COUNT 4

#define SHRT_MIN    (-32767)        /* minimum (signed) short value */
#define SHRT_MAX      32767         /* maximum (signed) short value */

UARTTask*   g_pUartTask = NULL;

#if (DENOISE_METHOD == KOTI_AEC)
    #include "../_aec/KotiAEC.h"
#elif (DENOISE_METHOD == QUAL_COMM)
    #include "stdlib.h"
    #include "stdio.h"
    #include "math.h"
    #include <sys/time.h>
    #include "NCS_NES_DEMO_Library.h"
#elif (DENOISE_METHOD == SPEEX_DSP)
    #include <speex_preprocess.h>
#elif (DENOISE_METHOD == WEB_RTC)
    #include "noise_suppression.h"
    enum nsLevel {
        kLow,
        kModerate,
        kHigh,
        kVeryHigh
    };
    NsHandle *nsHandle = NULL;
    int16_t *frameBuffer;
    int sampleRate = 8000;
#elif (DENOISE_METHOD == NNOM)
    #include "nnom.h"
    #include "weights.h"
    #include "mfcc.h"
    #include "wav.h"
    #include "equalizer_coeff.h"

    #define NUM_FEATURES NUM_FILTER

    #define _MAX(x, y) (((x) > (y)) ? (x) : (y))
    #define _MIN(x, y) (((x) < (y)) ? (x) : (y))

    #define NUM_CHANNELS 	1
    #define SAMPLE_RATE 	16000
    #define AUDIO_FRAME_LEN 512

    nnom_model_t *model = NULL;
    mfcc_t * mfcc = NULL;

    // audio buffer for input
    float audio_buffer[AUDIO_FRAME_LEN] = {0};
    int16_t audio_buffer_16bit[AUDIO_FRAME_LEN] = {0};

    // buffer for output
    int16_t audio_buffer_filtered[AUDIO_FRAME_LEN] = { 0 };

    // mfcc features and their derivatives
    float mfcc_feature[NUM_FEATURES] = { 0 };
    float mfcc_feature_prev[NUM_FEATURES] = { 0 };
    float mfcc_feature_diff[NUM_FEATURES] = { 0 };
    float mfcc_feature_diff_prev[NUM_FEATURES] = { 0 };
    float mfcc_feature_diff1[NUM_FEATURES] = { 0 };
    // features for NN
    float nn_features[64] = {0};
    int8_t nn_features_q7[64] = {0};

    // NN results, which is the gains for each frequency band
    float band_gains[NUM_FILTER] = {0};
    float band_gains_prev[NUM_FILTER] = {0};

    // 0db gains coefficient
    float coeff_b[NUM_FILTER][NUM_COEFF_PAIR] = FILTER_COEFF_B;
    float coeff_a[NUM_FILTER][NUM_COEFF_PAIR] = FILTER_COEFF_A;
    // dynamic gains coefficient
    float b_[NUM_FILTER][NUM_COEFF_PAIR] = {0};
    float delay_b_[DELAY_SAMPLE_COUNT][NUM_FILTER * NUM_COEFF_PAIR] = {0};
#endif

#if (ENABLE_VAD == 1)
    #include "vad/include/vad.h"
//    VadInst *vadInst_rec;
    VadInst *vadInst_play;
//    int sampleRate = 16000;
#endif

#if (DENOISE_METHOD == NNOM)
    #define SAMPLE_NUMBER (AUDIO_FRAME_LEN / 2)//320
#else
    #define SAMPLE_NUMBER (160)//320
#endif

int capturing;
int closed;

#if (AUDIO_COMM == 1)
char law_buffer[SAMPLE_NUMBER] = {0};
char law_play_buffer[SAMPLE_NUMBER] = {0};
#endif

char buffer[SAMPLE_NUMBER * 2] = {0};
char aec_buffer[SAMPLE_NUMBER * 2] = {0};
char out_buffer[SAMPLE_NUMBER * 2] = {0};

#if (DENOISE_METHOD == NNOM || 1)
int16_t play_buffer[SAMPLE_NUMBER * BATCH_SAMPLE_COUNT] = {0};
float delay_buffer[SAMPLE_NUMBER * 2 * DELAY_SAMPLE_COUNT] = {0};
#endif

char delay_buffer_8[SAMPLE_NUMBER * 2 * DELAY_SAMPLE_COUNT] = {0};
int idx = 0;
pthread_t recthread;
pthread_t playthread;
unsigned int size;

#if (DENOISE_METHOD == KOTI_AEC)
    void aecInit();
    void aecFree();
    void aecFarend(short *farendBuf, int length);
    void aecProcess(short *nearendBuf, short *dstBuf, int length);
#elif (DENOISE_METHOD == QUAL_COMM)
    NCSNES_ConfigPara_T ptNCSNESConfigPara;
    void qualcommInit();
#elif(DENOISE_METHOD == SPEEX_DSP)
    SpeexPreprocessState *st = NULL;
    void initspeexdsp();
#elif(DENOISE_METHOD == WEB_RTC)
    void denoise_create();
    void denoise(int16_t *input, int samplesCount);
    void denoise_free();
#elif(DENOISE_METHOD == NNOM)
    int  nnom();
    void create_nnom();
    void free_nnom();
    void process_nnom(int16_t* src_buffer, float* dst_buffer, float* dst_b);

    void y_h_update(float *y_h, uint32_t len);
    void equalizer(float* x, float* y, uint32_t signal_len, float *b, float *a, uint32_t num_band, uint32_t num_order);
    void set_gains(float *b_in, float *b_out,  float* gains, uint32_t num_band, uint32_t num_order);
    void quantize_data(float*din, int8_t *dout, uint32_t size, uint32_t int_bit);
    void log_values(float* value, uint32_t size, FILE* f);
#endif

#if (ENABLE_VAD == 1)
    void create_vad();
    void free_vad();
#endif

unsigned int capture_sample();
void play_sample();

void* play_thread(void*)
{
    sleep(1);
    printf("play thread\n");
#if (AUDIO_COMM == 1)
    if(g_pUartTask)
        g_pUartTask->setPlayFlag(1);
#endif
    play_sample();
}

void ConfigMix()
{
#if 0
    system("tinymix set 0 0"); //INT	1	dac digital volume                      0 (range 0->63)
    system("tinymix set 1 0"); //BOOL	1	dac: right chanel en                    Off
    system("tinymix set 2 0"); //BOOL	1	dac: left chanle en                     Off
    system("tinymix set 3 0"); //BOOL	1	dac: right analog output mixer en       Off
    system("tinymix set 4 0"); //BOOL	1	dac: left analog output mixer en        Off
    system("tinymix set 5 0"); //BOOL	1	dac: right mute                         Off
    system("tinymix set 6 0"); //BOOL	1	dac: left mute                          Off
    system("tinymix set 7 0"); //BOOL	1	hp right source select: 0-dac, 1-mixer  Off
    system("tinymix set 8 0"); //BOOL	1	hp left source select: 0-dac, 1-mixer   Off
    system("tinymix set 9 0"); //BOOL	1	dac: right mixer mute: mic              Off
    system("tinymix set 10 0"); //BOOL	1	dac: right mixer mute: linein           Off
    system("tinymix set 11 0"); //BOOL	1	dac: right mixer mute: FM               Off
    system("tinymix set 12 0"); //BOOL	1	dac: right mixer mute: right dac        Off
    system("tinymix set 13 0"); //BOOL	1	dac: right mixer mute: left dac         Off
    system("tinymix set 14 1"); //BOOL	1	head phone power                        On
    system("tinymix set 15 0"); //BOOL	1	dac: left mixer mute: mic               Off
    system("tinymix set 16 0"); //BOOL	1	dac: left mixer mute: linein            Off
    system("tinymix set 17 0"); //BOOL	1	dac: left mixer mute: FM                Off
    system("tinymix set 18 0"); //BOOL	1	dac: left mixer mute: right dac         Off
    system("tinymix set 19 0"); //BOOL	1	dac: left mixer mute: left dac          Off
    system("tinymix set 20 0"); //BOOL	1	dac: left hpout to right hpout          Off
    system("tinymix set 21 0"); //BOOL	1	dac: right hpout to left hpout          Off
    system("tinymix set 23 0"); //INT	1	MICIN GAIN control                      3 (range 0->7)
    system("tinymix set 24 0"); //INT	1	LINEIN GAIN control                     0 (range 0->7)
    system("tinymix set 26 1"); //INT	1	COS slop time control for Anti-pop      1 (range 0->3)
    system("tinymix set 28 0"); //BOOL	1	ADC mixer mute for FML                  Off
    system("tinymix set 29 0"); //BOOL	1	ADC mixer mute for FMR                  Off
    system("tinymix set 30 0"); //BOOL	1	ADC mixer mute for linein               Off
    system("tinymix set 31 0"); //BOOL	1	ADC mixer mute for left ouput           Off ----------------------------------
    system("tinymix set 32 0"); //BOOL	1	ADC mixer mute for right ouput          Off ----------------------------------
    system("tinymix set 33 0"); //BOOL	1	ADC PA speed select                     Off
    system("tinymix set 34 0"); //INT	1	ADC FM volume                           0 (range 0->7)
    system("tinymix set 37 0"); //BOOL	1	SPK_L Mux Left Mixer en                 Off
    system("tinymix set 38 0"); //BOOL	1	SPK_R Mux Right Mixer en                Off
    system("tinymix set 39 0"); //BOOL	1	External Speaker Switch                 Off
#endif
    system("tinymix set 22 60"); //INT	1	head phone volume                       63 (range 0->63) ---------------------
    system("tinymix set 25 3"); //INT	1	ADC INPUT GAIN control                  3 (range 0->7)
    system("tinymix set 27 1"); //BOOL	1	ADC mixer mute for mic                  On -----------------------------------
    system("tinymix set 35 1"); //BOOL	1	ADC MIC Boost AMP en                    On -----------------------------------
    system("tinymix set 36 4"); //INT	1	ADC MIC Boost AMP gain control          4 (range 0->7)
}

void StartAudioTask()
{
#if (DENOISE_METHOD == KOTI_AEC)
    aecInit();
#elif(DENOISE_METHOD == QUAL_COMM)
    qualcommInit();
#elif (DENOISE_METHOD == SPEEX_DSP)
    initspeexdsp();
#elif (DENOISE_METHOD == WEB_RTC)
    denoise_create();
#elif (DENOISE_METHOD == NNOM)
    create_nnom();
#endif

#if (ENABLE_VAD == 1)
    create_vad();
#endif

#if (AUDIO_COMM == 1)
    pcm16_alaw_tableinit();
    alaw_pcm16_tableinit();

    UART_SetBaudrate(B460800);
    g_pUartTask = new UARTTask;
    g_pUartTask->Start();
#endif
    capturing = 1;
    closed = 0;

#if (REC_PLAY_IN_A_THREAD == 0)
    if((pthread_create(&playthread, NULL, play_thread, NULL)) == 0)
    {
        printf("Create pthread ok!\n");
    }
    else
    {
        printf("Create pthread failed!\n");
    }
#endif
    if((pthread_create(&recthread, NULL, record_thread, NULL)) == 0)
    {
        printf("Create pthread ok!\n");
    }
    else
    {
        printf("Create pthread failed!\n");
    }
}

void StopAudioTask()
{
    capturing = 0;
    closed = 1;
    idx = 0;
    memset(delay_buffer, 0, SAMPLE_NUMBER * 2 * DELAY_SAMPLE_COUNT);

    if(playthread)
    {
        pthread_join(playthread, NULL);
        playthread = 0;
    }

    if(recthread)
    {
        pthread_join(recthread, NULL);
        recthread = 0;
    }
#if (AUDIO_COMM == 1)
    if(g_pUartTask != NULL)
    {
        g_pUartTask->Stop();
        delete g_pUartTask;
        g_pUartTask = NULL;
    }
#endif    
#if (DENOISE_METHOD == KOTI_AEC)
    aecFree();
#elif (DENOISE_METHOD == WEB_RTC)
    denoise_free();
#elif (DENOISE_METHOD == NNOM)
    free_nnom();
#endif

#if (ENABLE_VAD == 1)
    free_vad();
#endif
}

unsigned int capture_sample()
{
    int i;
    struct pcm *h_cap;
    if (pcm_open_stream(&h_cap, PCM_IN) == EXIT_FAILURE)
    {
        printf("[ARecThread] open capture failed.\n");
        return 0;
    }
#if (REC_PLAY_IN_A_THREAD == 1)
    struct pcm *h_play;

    if (pcm_open_stream(&h_play, PCM_OUT) == EXIT_FAILURE)
        return 0 ;// err;
#endif
//    size = pcm_frames_to_bytes(h_cap, pcm_get_buffer_size(h_cap));
    size = SAMPLE_NUMBER * 2;

    while (capturing) {
        float r = Now();
        pcm_read(h_cap, buffer, size);
#if (AUDIO_COMM == 1)
        pcm16_to_alaw(size, (const char*)buffer, law_buffer);
        g_pUartTask->pushSendBuffer(law_buffer);
#endif
#if (SAVE_RAW_FILE == 1)
        FILE* _fp;
        _fp = fopen("/tmp/rec_out.raw", "a+");
        if (_fp)
        {
            fwrite(buffer, size, 1, _fp);
            fclose(_fp);
        }
#endif
#if 0
        int nVadRet = WebRtcVad_Process(vadInst_rec, sampleRate, (short*)buffer, 160);
        if (nVadRet == -1) {
            printf("failed in WebRtcVad_Process\n");
//            capturing = 0;
//            closed = 1;
        } else if (nVadRet == 0){
            memset(buffer, 0, size);
            // output result
        }
#if (SAVE_RAW_FILE == 1)
        _fp = fopen("/tmp/rec_mid_out.raw", "a+");
        if (_fp)
        {
            fwrite(buffer, size, 1, _fp);
            fclose(_fp);
        }
#endif
#endif
        float r1 = Now();
#if (DENOISE_METHOD == KOTI_AEC)
        aecProcess((short*)buffer, (short*)aec_buffer, size / 2);
#elif (DENOISE_METHOD == QUAL_COMM)
        NCS_NES_Algorithm_Proc((short*)buffer, (short*)aec_buffer, (short*)out_buffer, &ptNCSNESConfigPara, SAMPLE_NUMBER, sampleRate);
//        printf("(read)engine   time  %f\n", Now() - r1);
        memcpy(aec_buffer, out_buffer, size);
        memcpy(delay_buffer_8 + idx * size, out_buffer, size);
        idx++;
        idx = idx % DELAY_SAMPLE_COUNT;
    #if (REC_PLAY_IN_A_THREAD == 1)
//            pcm_write(h_play, aec_buffer, size);
            _fp = fopen("/tmp/rec_aec_out.raw", "a+");
            if (_fp)
            {
                fwrite(aec_buffer, size, 1, _fp);
                fclose(_fp);
            }
    #endif
#elif (DENOISE_METHOD == SPEEX_DSP)
        speex_preprocess_run(st, (short*)buffer);
        int nVadRet = WebRtcVad_Process(vadInst, sampleRate, (short*)buffer, SAMPLE_NUMBER);
        printf("vad ret 1:  %d \n", nVadRet);
        if (nVadRet == -1) {
            printf("failed in WebRtcVad_Process\n");
            WebRtcVad_Free(vadInst);
            return -1;
        } else if (nVadRet == 0){
            memset(buffer, 0, size);
            // output result
        }

        nVadRet = WebRtcVad_Process(vadInst, sampleRate, (short*)buffer, SAMPLE_NUMBER);
        printf("vad ret 2:  %d \n", nVadRet);
        if (nVadRet == -1) {
            printf("failed in WebRtcVad_Process\n");
            WebRtcVad_Free(vadInst);
            return -1;
        } else if (nVadRet == 0){
            memset(buffer, 0, size);
            // output result
        }
        pcm_write(h_play, buffer, size);
#elif (DENOISE_METHOD == WEB_RTC)
//        memcpy(out_buffer, buffer, size);
#if (ENABLE_VAD == 1 && 0)
        int nVadRet = WebRtcVad_Process(vadInst, sampleRate, (short*)buffer, SAMPLE_NUMBER);
        printf("vad ret 1:  %d \n", nVadRet);
        if (nVadRet == -1) {
            printf("failed in WebRtcVad_Process\n");
            WebRtcVad_Free(vadInst);
            return -1;
        } else if (nVadRet == 0){
            memset(buffer, 0, size);
            // output result
        }

        nVadRet = WebRtcVad_Process(vadInst, sampleRate, (short*)buffer, SAMPLE_NUMBER);
        printf("vad ret 2:  %d \n", nVadRet);
        if (nVadRet == -1) {
            printf("failed in WebRtcVad_Process\n");
            WebRtcVad_Free(vadInst);
            return -1;
        } else if (nVadRet == 0){
            memset(buffer, 0, size);
            // output result
        }
#endif
#elif (DENOISE_METHOD == NNOM)
        process_nnom((int16_t*)buffer, &delay_buffer[idx * size], delay_b_[idx]);
//        memcpy(delay_buffer_8 + idx * size, buffer, size);
//        printf("####################################   idx   %d\n", idx);
        idx++;
        idx = idx % DELAY_SAMPLE_COUNT;
#if (REC_PLAY_IN_A_THREAD == 1)
        pcm_write(h_play, (char*)buffer, size);
        _fp = fopen("/tmp/rec_aec_out.raw", "a+");
        if (_fp)
        {
            fwrite(buffer, size, 1, _fp);
            fclose(_fp);
        }
#endif
#endif
//        printf("(read)engine   time  %f\n", Now() - r1);
//        printf("######################  capture   %f (%f)\n", Now() - r, Now());
    }

    pcm_close_stream(h_cap);
    return 1;
}

void play_sample()
{
    struct pcm *h_play;
    int i;

    if (pcm_open_stream(&h_play, PCM_OUT) == EXIT_FAILURE)
        return;// err;
#if (DENOISE_METHOD == QUAL_COMM || DENOISE_METHOD == RNNOISE || DENOISE_METHOD == NNOM)
//    while(idx < BATCH_SAMPLE_COUNT && idx >= DELAY_SAMPLE_COUNT - 4)
//        usleep(10 * 1000);
#endif
    int tmp = (idx + DELAY_SAMPLE_COUNT - BATCH_SAMPLE_COUNT * 2) % DELAY_SAMPLE_COUNT;
    tmp = tmp / BATCH_SAMPLE_COUNT * BATCH_SAMPLE_COUNT;
    do {
        float r = Now();
#if (AUDIO_COMM == 1)
        if(g_pUartTask->popRecvBuffer(law_play_buffer) == 0)
            continue;
        alaw_to_pcm16(SAMPLE_NUMBER, (const char*)law_play_buffer, (char*)out_buffer);
//        alaw_to_pcm16(SAMPLE_NUMBER, (const char*)law_buffer, (char*)out_buffer);

        short* tmp_buf = (short*)out_buffer;
        for(int i = 0; i<SAMPLE_NUMBER; i++)
        {
            tmp_buf[i] = tmp_buf[i] * 2;
            if(tmp_buf[i] > SHRT_MAX)
                tmp_buf[i] = SHRT_MAX;
            else if(tmp_buf[i] < SHRT_MIN)
                tmp_buf[i] = SHRT_MIN;
        }
#endif

#if (DENOISE_METHOD == NONE)
//        usleep(10*1000);
        pcm_write(h_play, out_buffer, size);
#elif(DENOISE_METHOD == KOTI_AEC)
        pcm_write(h_play, aec_buffer, size);
#elif(DENOISE_METHOD == QUAL_COMM)
//        printf("&&&&&&&&&&&&&&&&&&&r   write  tmp   %d\n", tmp);
        for(int i = 0; i < BATCH_SAMPLE_COUNT; i ++)
        {
            int nVadRet = WebRtcVad_Process(vadInst, sampleRate, (short*)delay_buffer_8 + (tmp + i) * size / 2, size / 2);
//            printf("vad ret 1:  %d \n", nVadRet);
            if (nVadRet == -1) {
                printf("failed in WebRtcVad_Process\n");
                capturing = 0;
                closed = 1;
            } else if (nVadRet == 0){
                memset(delay_buffer_8 + (tmp + i) * size, 0, size);
                // output result
            }
        }

        pcm_write(h_play, delay_buffer_8 + tmp * size, size * BATCH_SAMPLE_COUNT);
#if (SAVE_RAW_FILE == 1)
        FILE* _fp;
        _fp = fopen("/tmp/rec_aec_out.raw", "a+");
        if (_fp)
        {
            fwrite(delay_buffer_8 + tmp * size, size*BATCH_SAMPLE_COUNT, 1, _fp);
            fclose(_fp);
        }
#endif
        tmp += BATCH_SAMPLE_COUNT;
        tmp = tmp % DELAY_SAMPLE_COUNT;
#elif(DENOISE_METHOD == WEB_RTC)
        denoise((short*)out_buffer, SAMPLE_NUMBER);

        pcm_write(h_play, out_buffer, size);
#elif(DENOISE_METHOD == NNOM)
        float r1 = Now();
#if 1
        int nNoVadCnt = 0;
        for(int k = 0; k< BATCH_SAMPLE_COUNT; k++)
        {
            float* tmp_f = &delay_buffer[(tmp + k) * size];

            // finally, we apply the equalizer to this audio frame to denoise
            equalizer(tmp_f, &tmp_f[AUDIO_FRAME_LEN / 2], AUDIO_FRAME_LEN/2, delay_b_[tmp + k],(float*)coeff_a, NUM_FILTER, NUM_ORDER);
            // convert the filtered signal back to int16
            for (int i = 0; i < AUDIO_FRAME_LEN / 2; i++)
                play_buffer[k * AUDIO_FRAME_LEN / 2 + i] = tmp_f[i + AUDIO_FRAME_LEN / 2] * 32768.f * 0.6f;

//            int nVadRet = WebRtcVad_Process(vadInst_play, sampleRate, &play_buffer[(k) * AUDIO_FRAME_LEN / 2], 160);
//            if (nVadRet == -1) {
//                printf("failed in WebRtcVad_Process\n");
////                nNoVadCnt++;
////                break;
////                    capturing = 0;
////                    closed = 1;
//            } else if (nVadRet == 0){
//                memset(&play_buffer[(k) * AUDIO_FRAME_LEN / 2], 0, AUDIO_FRAME_LEN);
////                nNoVadCnt++;
////                break;
//                // output result
//            }
#if 1
            if(k > 0)
            {
//                nVadRet = WebRtcVad_Process(vadInst_play, sampleRate, &play_buffer[(k - 1) * AUDIO_FRAME_LEN / 2], 320);
//                if (nVadRet == -1) {
//                    printf("failed in WebRtcVad_Process\n");
//                    nNoVadCnt++;
//                    break;
////                    capturing = 0;
////                    closed = 1;
//                } else if (nVadRet == 0){
////                    memset(&play_buffer[(k - 1) * AUDIO_FRAME_LEN / 2], 0, AUDIO_FRAME_LEN * 2);
//                    nNoVadCnt++;
//                    break;
//                    // output result
//                }

//                nVadRet = WebRtcVad_Process(vadInst_play, sampleRate, &play_buffer[(k + 1) * AUDIO_FRAME_LEN / 2 - 320], 320);
//                if (nVadRet == -1) {
//                    printf("failed in WebRtcVad_Process\n");
//                    nNoVadCnt++;
//                    break;
////                    capturing = 0;
////                    closed = 1;
//                } else if (nVadRet == 0){
////                    memset(&play_buffer[(k - 1) * AUDIO_FRAME_LEN / 2], 0, AUDIO_FRAME_LEN * 2);
//                    nNoVadCnt++;
//                    break;
//                    // output result
//                }

                int nVadRet = WebRtcVad_Process(vadInst_play, sampleRate, &play_buffer[(k - 1) * AUDIO_FRAME_LEN / 2], 480);
                if (nVadRet == -1) {
                    printf("failed in WebRtcVad_Process\n");
                    nNoVadCnt++;
//                    break;
//                    capturing = 0;
//                    closed = 1;
                } else if (nVadRet == 0){
                    memset(&play_buffer[(k - 1) * AUDIO_FRAME_LEN / 2], 0, AUDIO_FRAME_LEN * 2);
                    nNoVadCnt++;
//                    break;
                    // output result
                }
            }
#endif
        }
#endif

//        printf("@@@@@@@@@@@@@@@@222 no vad cnt  %d\n", nNoVadCnt);
//        if(nNoVadCnt != 0)
//            memset(play_buffer, 0, size * BATCH_SAMPLE_COUNT);

        pcm_write(h_play, (char*)play_buffer, size * BATCH_SAMPLE_COUNT);
//        printf("@@@@@@@@@@@@@@@@@@2  tmp   %d :  equalize   %f\n", tmp, Now() - r1);
#if (SAVE_RAW_FILE == 1)
        FILE* _fp;
        _fp = fopen("/tmp/rec_aec_out.raw", "a+");
        if (_fp)
        {
            fwrite((char*)play_buffer, size*BATCH_SAMPLE_COUNT, 1, _fp);
            fclose(_fp);
        }
#endif
        tmp += BATCH_SAMPLE_COUNT;
        tmp = tmp % DELAY_SAMPLE_COUNT;
#endif
#if (SAVE_RAW_FILE == 1)
        FILE* _fp;
        _fp = fopen("/tmp/rec_aec_out.raw", "a+");
        if (_fp)
        {
            fwrite(out_buffer, size, 1, _fp);
            fclose(_fp);
        }
#endif

//        printf("$$$$$$$$$$$$$$$     play   %f (%f)\n", Now() - r, Now());
#if (DENOISE_METHOD == KOTI_AEC)
//        aecFarend((short*)aec_buffer, size / 2);
#endif
    } while (!closed);

#if (AUDIO_COMM == 1)
    if(g_pUartTask)
        g_pUartTask->setPlayFlag(0);
#endif        
    pcm_close_stream(h_play);
}

#if (DENOISE_METHOD == KOTI_AEC)
void aecInit()
{
    KotiAEC_init();
}

void aecFree()
{
    KotiAEC_destory();
}

#define ECHO_PROC_FRAME_UNIT 160

void aecFarend(short *farendBuf, int length)
{
    for(int i = 0; i < length; i += ECHO_PROC_FRAME_UNIT)
    {
        speex_aec_playback_for_async(farendBuf + i);
    }
}

void aecProcess(short* nearendBuf, short *dstBuf, int length)
{
//    FILE* _fp;
//    _fp = fopen("/tmp/play_out.raw", "a+");
//    if (_fp)
//    {
//        fwrite(nearendBuf, length, 2, _fp);
//        fclose(_fp);
//    }

    for(int i = 0; i < length; i += ECHO_PROC_FRAME_UNIT)
    {
        KotiAEC_process(NULL, nearendBuf + i, (int16_t*)dstBuf + i);
    }

//    _fp = fopen("/tmp/aec_out.raw", "a+");
//    if (_fp)
//    {
//        fwrite(dstBuf, length, 2, _fp);
//        fclose(_fp);
//    }
}
#elif (DENOISE_METHOD == QUAL_COMM)
void qualcommInit()
{
    ptNCSNESConfigPara.NCS_Enable = 2;
    ptNCSNESConfigPara.FixedDelay = 1;
    ptNCSNESConfigPara.AttenuationDB = -30;
    ptNCSNESConfigPara.Tilt = 9;
    ptNCSNESConfigPara.EchoTail = 300;
    ptNCSNESConfigPara.NLPLevel = 5;
    ptNCSNESConfigPara.OutputNoiseLevel = 1;
    ptNCSNESConfigPara.InputFixedGain = 6;
    ptNCSNESConfigPara.PreAGC_Gain = 4;
    ptNCSNESConfigPara.PostAGC_Gain = 2;
    ptNCSNESConfigPara.Mode = 1;
    ptNCSNESConfigPara.FidelityLevel = 2;


    ptNCSNESConfigPara.AES_Enable = 0;
    ptNCSNESConfigPara.AES_Filter_L = 240;//400
    ptNCSNESConfigPara.AES_M_factor = 0.09*32768;
    ptNCSNESConfigPara.AES_i_Delay = 32;
    ptNCSNESConfigPara.AES_Tilt = -0.85*32768;
    ptNCSNESConfigPara.AES_FixedDelaySample = 0;
    ptNCSNESConfigPara.AES_LevelNLP = 0.8*32768;

    ptNCSNESConfigPara.ul_in_fixed_gain = 0;
    ptNCSNESConfigPara.dl_in_fixed_gain = 0;
    ptNCSNESConfigPara.ul_out_fixed_gain = 0;
    ptNCSNESConfigPara.Test_mode = 0;
    ptNCSNESConfigPara.Mips_test = 0;

    NCS_NES_Algorithm_Init(&ptNCSNESConfigPara, sampleRate);
}

#elif (DENOISE_METHOD == SPEEX_DSP)
void initspeexdsp()
{
    int i;
    float f;

    st = speex_preprocess_state_init(SAMPLE_NUMBER, 8000);

    i=1;
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_DENOISE, &i);
    i=0;
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_AGC, &i);
    i=8000;
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_AGC_LEVEL, &i);
    i=-90;
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_NOISE_SUPPRESS, &i);
    i=0;
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_DEREVERB, &i);
    f=.0;
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_DEREVERB_DECAY, &f);
    f=.0;
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_SET_DEREVERB_LEVEL, &f);



    int32_t tmp = -1; float tmp1 = 0.0f;
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_DENOISE, &tmp);
    printf("SPEEX_PREPROCESS_GET_DENOISE: %d\n", tmp);
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_NOISE_SUPPRESS, &tmp);
    printf("SPEEX_PREPROCESS_GET_NOISE_SUPPRESS: %d\n", tmp);

    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_AGC, &tmp);
    printf("SPEEX_PREPROCESS_GET_AGC: %d\n", tmp);
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_VAD, &tmp);
    printf("SPEEX_PREPROCESS_GET_VAD: %d\n", tmp);
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_PROB_START, &tmp);
    printf("SPEEX_PREPROCESS_GET_PROB_START: %d\n", tmp);
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_PROB_CONTINUE, &tmp);
    printf("SPEEX_PREPROCESS_GET_PROB_CONTINUE: %d\n", tmp);

    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_AGC_LEVEL, &tmp1);
    printf("SPEEX_PREPROCESS_GET_AGC_LEVEL: %f\n", tmp1);
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_AGC_DECREMENT, &tmp);
    printf("SPEEX_PREPROCESS_GET_AGC_DECREMENT: %d\n", tmp);
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_AGC_INCREMENT, &tmp);
    printf("SPEEX_PREPROCESS_GET_AGC_INCREMENT: %d\n", tmp);
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_AGC_MAX_GAIN, &tmp);
    printf("SPEEX_PREPROCESS_GET_AGC_MAX_GAIN: %d\n", tmp);
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_AGC_GAIN, &tmp);
    printf("SPEEX_PREPROCESS_GET_AGC_GAIN: %d\n", tmp);
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_AGC_TARGET, &tmp);
    printf("SPEEX_PREPROCESS_GET_AGC_TARGET: %d\n", tmp);
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_AGC_LOUDNESS, &tmp);
    printf("SPEEX_PREPROCESS_GET_AGC_LOUDNESS: %d\n", tmp);

    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_ECHO_SUPPRESS, &tmp);
    printf("SPEEX_PREPROCESS_GET_ECHO_SUPPRESS: %d\n", tmp);
    speex_preprocess_ctl(st, SPEEX_PREPROCESS_GET_ECHO_SUPPRESS_ACTIVE, &tmp);
    printf("SPEEX_PREPROCESS_GET_ECHO_SUPPRESS_ACTIVE: %d\n", tmp);
}
#elif (DENOISE_METHOD == WEB_RTC)

void denoise_create() {
    enum nsLevel level = kVeryHigh;
    size_t samples = MIN(160, sampleRate / 100);
    frameBuffer = (int16_t *)malloc(sizeof(*frameBuffer) * samples);

    nsHandle = WebRtcNs_Create();
    int status = WebRtcNs_Init(nsHandle, sampleRate);
    status = WebRtcNs_set_policy(nsHandle, level);
}

void denoise(int16_t *input, int samplesCount)
{
    size_t samples = MIN(160, sampleRate / 100);
    size_t frames = (samplesCount / samples);

    for (int i = 0; i < frames; i++)
    {
        for (int k = 0; k < samples; k++)
            frameBuffer[k] = input[k];
        memcpy(frameBuffer, input, samples * sizeof(int16_t));

        int16_t *nsIn[1] = { frameBuffer };   //ns input[band][data]
        int16_t *nsOut[1] = { frameBuffer };  //ns output[band][data]
        WebRtcNs_Analyze(nsHandle, nsIn[0]);
        WebRtcNs_Process(nsHandle, (const int16_t *const *)nsIn, 1, nsOut);

        memcpy(input, frameBuffer, samples * sizeof(int16_t));
        input += samples;
    }
}

void denoise_free() {
    if (nsHandle)
    {
        WebRtcNs_Free(nsHandle);
        nsHandle = NULL;
        free(frameBuffer);
    }
}
#elif (DENOISE_METHOD == NNOM)
// update the history
void y_h_update(float *y_h, uint32_t len)
{
    for (uint32_t i = len-1; i >0 ;i--)
        y_h[i] = y_h[i-1];
}
//  equalizer by multiple n order iir band pass filter.
// y[i] = b[0] * x[i] + b[1] * x[i - 1] + b[2] * x[i - 2] - a[1] * y[i - 1] - a[2] * y[i - 2]...
void equalizer(float* x, float* y, uint32_t signal_len, float *b, float *a, uint32_t num_band, uint32_t num_order)
{
    // the y history for each band
    static float y_h[NUM_FILTER][NUM_COEFF_PAIR] = { 0 };
    static float x_h[NUM_COEFF_PAIR * 2] = { 0 };
    uint32_t num_coeff = num_order * 2 + 1;

    // i <= num_coeff (where historical x is involved in the first few points)
    // combine state and new data to get a continual x input.
    memcpy(x_h + num_coeff, x, num_coeff * sizeof(float));
    for (uint32_t i = 0; i < num_coeff; i++)
    {
        y[i] = 0;
        for (uint32_t n = 0; n < num_band; n++)
        {
            y_h_update(y_h[n], num_coeff);
            y_h[n][0] = b[n * num_coeff] * x_h[i+ num_coeff];
            for (uint32_t c = 1; c < num_coeff; c++)
                y_h[n][0] += b[n * num_coeff + c] * x_h[num_coeff + i - c] - a[n * num_coeff + c] * y_h[n][c];
            y[i] += y_h[n][0];
        }
    }
    // store the x for the state of next round
    memcpy(x_h, &x[signal_len - num_coeff], num_coeff * sizeof(float));

    // i > num_coeff; the rest data not involed the x history
    for (uint32_t i = num_coeff; i < signal_len; i++)
    {
        y[i] = 0;
        for (uint32_t n = 0; n < num_band; n++)
        {
            y_h_update(y_h[n], num_coeff);
            y_h[n][0] = b[n * num_coeff] * x[i];
            for (uint32_t c = 1; c < num_coeff; c++)
                y_h[n][0] += b[n * num_coeff + c] * x[i - c] - a[n * num_coeff + c] * y_h[n][c];
            y[i] += y_h[n][0];
        }
    }
}

// set dynamic gains. Multiple gains x b_coeff
void set_gains(float *b_in, float *b_out,  float* gains, uint32_t num_band, uint32_t num_order)
{
    uint32_t num_coeff = num_order * 2 + 1;
    for (uint32_t i = 0; i < num_band; i++)
        for (uint32_t c = 0; c < num_coeff; c++)
            b_out[num_coeff *i + c] = b_in[num_coeff * i + c] * gains[i]; // only need to set b.
}

void quantize_data(float*din, int8_t *dout, uint32_t size, uint32_t int_bit)
{
    float limit = (1 << int_bit);
    for(uint32_t i=0; i<size; i++)
        dout[i] = (int8_t)(_MAX(_MIN(din[i], limit), -limit) / limit * 127);
}

void log_values(float* value, uint32_t size, FILE* f)
{
    char line[16];
    for (uint32_t i = 0; i < size; i++) {
        snprintf(line, 16, "%f,", value[i]);
        fwrite(line, strlen(line), 1, f);
    }
    fwrite("\n", 2, 1, f);
}

void create_nnom()
{
    // NNoM model
    model = nnom_model_create();

    // 26 features, 0 offset, 26 bands, 512fft, 0 preempha, attached_energy_to_band0
    mfcc = mfcc_create(NUM_FEATURES, 0, NUM_FEATURES, 512, 0, true);
}

void free_nnom()
{
    // print some model info
    model_io_format(model);
    model_stat(model);
    model_delete(model);
}

void process_nnom(int16_t* src_buffer, float* dst_buffer, float* dst_b)
{
    if(src_buffer == NULL)
        return;

    memcpy(audio_buffer_16bit, &audio_buffer_16bit[AUDIO_FRAME_LEN/2], AUDIO_FRAME_LEN/2*sizeof(int16_t));

    // now read the new data
    memcpy(&audio_buffer_16bit[AUDIO_FRAME_LEN / 2], src_buffer, AUDIO_FRAME_LEN / 2 * sizeof(int16_t));

    float r = Now();
    // get mfcc
    mfcc_compute(mfcc, audio_buffer_16bit, mfcc_feature);
//    printf("=============  mfcc %f\n", Now() - r);

//log_values(mfcc_feature, NUM_FEATURES, flog);

    // get the first and second derivative of mfcc
    for(uint32_t i=0; i< NUM_FEATURES; i++)
    {
        mfcc_feature_diff[i] = mfcc_feature[i] - mfcc_feature_prev[i];
        mfcc_feature_diff1[i] = mfcc_feature_diff[i] - mfcc_feature_diff_prev[i];
    }
    memcpy(mfcc_feature_prev, mfcc_feature, NUM_FEATURES * sizeof(float));
    memcpy(mfcc_feature_diff_prev, mfcc_feature_diff, NUM_FEATURES * sizeof(float));

    // combine MFCC with derivatives
    memcpy(nn_features, mfcc_feature, NUM_FEATURES*sizeof(float));
    memcpy(&nn_features[NUM_FEATURES], mfcc_feature_diff, 10*sizeof(float));
    memcpy(&nn_features[NUM_FEATURES+10], mfcc_feature_diff1, 10*sizeof(float));

//log_values(nn_features, NUM_FEATURES+20, flog);

    // quantise them using the same scale as training data (in keras), by 2^n.
    quantize_data(nn_features, nn_features_q7, NUM_FEATURES+20, 3);

    // run the mode with the new input
    memcpy(nnom_input_data, nn_features_q7, sizeof(nnom_input_data));
    model_run(model);

    // read the result, convert it back to float (q0.7 to float)
    for(int i=0; i< NUM_FEATURES; i++)
        band_gains[i] = (float)(nnom_output_data[i]) / 127.f;

//log_values(band_gains, NUM_FILTER, flog);

    // one more step, limit the change of gians, to smooth the speech, per RNNoise paper
    for(int i=0; i< NUM_FEATURES; i++)
        band_gains[i] = _MAX(band_gains_prev[i]*0.8f, band_gains[i]);
    memcpy(band_gains_prev, band_gains, NUM_FEATURES *sizeof(float));

    // apply the dynamic gains to each frequency band.
    set_gains((float*)coeff_b, (float*)b_, band_gains, NUM_FILTER, NUM_ORDER);
    memcpy(dst_b, b_, sizeof(b_));

    // convert 16bit to float for equalizer
    for (int i = 0; i < AUDIO_FRAME_LEN/2; i++)
        audio_buffer[i] = audio_buffer_16bit[i + AUDIO_FRAME_LEN / 2] / 32768.f;

    memcpy(dst_buffer, audio_buffer, AUDIO_FRAME_LEN * 4);
#if 0
    // finally, we apply the equalizer to this audio frame to denoise
    r = Now();
    equalizer(audio_buffer, &audio_buffer[AUDIO_FRAME_LEN / 2], AUDIO_FRAME_LEN/2, (float*)b_,(float*)coeff_a, NUM_FILTER, NUM_ORDER);
    printf("@@@@@@@@@@@@@@@22  nnome   %f\n", Now() - r);
    // convert the filtered signal back to int16
    for (int i = 0; i < AUDIO_FRAME_LEN / 2; i++)
        dst_buffer[i] = audio_buffer[i + AUDIO_FRAME_LEN / 2] * 32768.f *0.6f;
#endif
//    printf("+===============  process time   %f\n", Now() - r);
}

nnom_status_t layer_callback(nnom_model_t *m, nnom_layer_t *layer)
{
    static int outputIndex[NNOM_TYPE_MAX] = { 0 , } ;
    char name[32];
    FILE* fp;

    outputIndex[layer->type]++;
    snprintf(name, sizeof(name),"%s%d.raw",
            default_layer_names[layer->type],
            outputIndex[layer->type]);
    fp = fopen(name,"w");

    if(fp != NULL)
    {
//        fwrite(layer->out->tensor->p_data, 1, tensor_size(layer->out->tensor), fp);
        for(int i = 0; i< tensor_size(layer->out->tensor); i++)
        {
            fprintf(fp, "%d\n", *(uint8_t*)(layer->out->tensor->p_data + i));
        }
        fclose(fp);
    }
    else
    {
        printf("failed to save %s\n",name);
    }
    return NN_SUCCESS;
}


int nnom()
{
    wav_header_t wav_header;
    size_t size;

    //char* input_file = "../../_noisy_sample.wav";
    //char* output_file = "../../_nn_fixed_filtered_sample.wav";
    char* input_file = "sample.wav";
    char* output_file = "filtered_sample.wav";
    FILE* src_file;
    FILE* des_file;

    char* log_file = "log.csv";
    FILE* flog = fopen(log_file, "wb");

    // if user has specify input and output files.

    src_file = fopen(input_file, "rb");
    des_file = fopen(output_file, "wb");
    if (src_file == NULL)
    {
        printf("Cannot open wav files, default input:'%s'\n", input_file);
        printf("Or use command to specify input file: xxx.exe [input.wav] [output.wav]\n");
        return -1;
    }
    if (des_file == NULL)
    {
        fclose(src_file);
        return -1;
    }

    // read wav file header, copy it to the output file.
    fread(&wav_header, sizeof(wav_header), 1, src_file);
    fwrite(&wav_header, sizeof(wav_header), 1, des_file);

    // lets jump to the "data" chunk of the WAV file.
    if (strncmp(wav_header.datachunk_id, "data", 4)){
        wav_chunk_t chunk;
        chunk.size = wav_header.datachunk_size;
        // find the 'data' chunk
        do {
            char* buf = (char*)malloc(chunk.size);
            fread(buf, chunk.size, 1, src_file);
            fwrite(buf, chunk.size, 1, des_file);
            free(buf);
            fread(&chunk, sizeof(wav_chunk_t), 1, src_file);
            fwrite(&chunk, sizeof(wav_chunk_t), 1, des_file);
        } while (strncmp(chunk.id, "data", 4));
    }
    // NNoM model
    nnom_model_t *model = model = nnom_model_create();
    printf("############### sizeof nnom_tensor_t  %d\n", sizeof(nnom_tensor_t));
//    model_set_callback(model, layer_callback);

    // 26 features, 0 offset, 26 bands, 512fft, 0 preempha, attached_energy_to_band0
    mfcc_t * mfcc = mfcc_create(NUM_FEATURES, 0, NUM_FEATURES, 512, 0, true);

    printf("\nProcessing file: %s\n", input_file);
    while(1) {
        // move buffer (50%) overlapping, move later 50% to the first 50, then fill
        memcpy(audio_buffer_16bit, &audio_buffer_16bit[AUDIO_FRAME_LEN/2], AUDIO_FRAME_LEN/2*sizeof(int16_t));

        // now read the new data
        size = fread(&audio_buffer_16bit[AUDIO_FRAME_LEN / 2], AUDIO_FRAME_LEN / 2 * sizeof(int16_t), 1, src_file);
        if(size == 0)
            break;

        float r = Now();
        // get mfcc
        mfcc_compute(mfcc, audio_buffer_16bit, mfcc_feature);

log_values(mfcc_feature, NUM_FEATURES, flog);

        // get the first and second derivative of mfcc
        for(uint32_t i=0; i< NUM_FEATURES; i++)
        {
            mfcc_feature_diff[i] = mfcc_feature[i] - mfcc_feature_prev[i];
            mfcc_feature_diff1[i] = mfcc_feature_diff[i] - mfcc_feature_diff_prev[i];
        }
        memcpy(mfcc_feature_prev, mfcc_feature, NUM_FEATURES * sizeof(float));
        memcpy(mfcc_feature_diff_prev, mfcc_feature_diff, NUM_FEATURES * sizeof(float));

        // combine MFCC with derivatives
        memcpy(nn_features, mfcc_feature, NUM_FEATURES*sizeof(float));
        memcpy(&nn_features[NUM_FEATURES], mfcc_feature_diff, 10*sizeof(float));
        memcpy(&nn_features[NUM_FEATURES+10], mfcc_feature_diff1, 10*sizeof(float));

log_values(nn_features, NUM_FEATURES+20, flog);

        // quantise them using the same scale as training data (in keras), by 2^n.
        quantize_data(nn_features, nn_features_q7, NUM_FEATURES+20, 3);

        // run the mode with the new input
        memcpy(nnom_input_data, nn_features_q7, sizeof(nnom_input_data));
        model_run(model);

        // read the result, convert it back to float (q0.7 to float)
        for(int i=0; i< NUM_FEATURES; i++)
            band_gains[i] = (float)(nnom_output_data[i]) / 127.f;

log_values(band_gains, NUM_FILTER, flog);

        // one more step, limit the change of gians, to smooth the speech, per RNNoise paper
        for(int i=0; i< NUM_FEATURES; i++)
            band_gains[i] = _MAX(band_gains_prev[i]*0.8f, band_gains[i]);
        memcpy(band_gains_prev, band_gains, NUM_FEATURES *sizeof(float));

        // apply the dynamic gains to each frequency band.
        set_gains((float*)coeff_b, (float*)b_, band_gains, NUM_FILTER, NUM_ORDER);

        // convert 16bit to float for equalizer
        for (int i = 0; i < AUDIO_FRAME_LEN/2; i++)
            audio_buffer[i] = audio_buffer_16bit[i + AUDIO_FRAME_LEN / 2] / 32768.f;

        // finally, we apply the equalizer to this audio frame to denoise
        equalizer(audio_buffer, &audio_buffer[AUDIO_FRAME_LEN / 2], AUDIO_FRAME_LEN/2, (float*)b_,(float*)coeff_a, NUM_FILTER, NUM_ORDER);

        // convert the filtered signal back to int16
        for (int i = 0; i < AUDIO_FRAME_LEN / 2; i++)
            audio_buffer_filtered[i] = audio_buffer[i + AUDIO_FRAME_LEN / 2] * 32768.f *0.6f;

//        printf("+===============  process time   %f\n", Now() - r);
        // write the filtered frame to WAV file.
        fwrite(audio_buffer_filtered, 256*sizeof(int16_t), 1, des_file);
    }

    // print some model info
    model_io_format(model);
    model_stat(model);
    model_delete(model);

    fclose(flog);
    fclose(src_file);
    fclose(des_file);

    printf("\nNoisy signal '%s' has been de-noised by NNoM.\nThe output is saved to '%s'.\n", input_file, output_file);
    return 0;
}
#endif

#if (ENABLE_VAD == 1)
void create_vad()
{
    vadInst_play = WebRtcVad_Create();
    if (vadInst_play == NULL)
        return;
    int status = WebRtcVad_Init(vadInst_play);
    if (status != 0) {
        printf("WebRtcVad_Init fail\n");
        WebRtcVad_Free(vadInst_play);
        return;
    }

    short vad_mode = 3;
    status = WebRtcVad_set_mode(vadInst_play, vad_mode);
    if (status != 0) {
        printf("WebRtcVad_set_mode fail\n");
        WebRtcVad_Free(vadInst_play);
        return;
    }

//    vadInst_rec = WebRtcVad_Create();
//    if (vadInst_rec == NULL)
//        return;
//    status = WebRtcVad_Init(vadInst_rec);
//    if (status != 0) {
//        printf("WebRtcVad_Init fail\n");
//        WebRtcVad_Free(vadInst_rec);
//        return;
//    }

//    status = WebRtcVad_set_mode(vadInst_rec, 3);
//    if (status != 0) {
//        printf("WebRtcVad_set_mode fail\n");
//        WebRtcVad_Free(vadInst_rec);
//        return;
//    }
}

void free_vad()
{
    WebRtcVad_Free(vadInst_play);
//    WebRtcVad_Free(vadInst_rec);
}
#endif
