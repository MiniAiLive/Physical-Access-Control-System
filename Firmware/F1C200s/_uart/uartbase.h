#ifndef UART_BASE_H
#define UART_BASE_H

#define PACKET_UNIT_SIZE    (128)

#include "thread.h"
#include "mutex.h"
#include <pthread.h>
using namespace std;
#include <vector>
#include <string>

#define AUDIO_PACKET_PREFIX         0x00
#define AUDIO_PACKET_PREFIX_LEN     7
#define AUDIO_PACKET_HEADER         0x5A
#define AUDIO_PACKET_LEN            160
#define AUDIO_QUEUE_SIZE                  8

extern pthread_mutex_t g_captureLock;
extern pthread_cond_t  g_captureCond;

#pragma pack(push, 1)
typedef struct _tagAUDIO_MSG
{
    unsigned char   bHeader;
    unsigned char   bUnmute;
    char   abBuffer[AUDIO_PACKET_LEN];
    unsigned char   bCheckSum;
} AUDIO_MSG;
#pragma pack(pop)

class UARTTask : public Thread
{
public:
    UARTTask();
    ~UARTTask();

    void    Start();
    void    Stop();

    int     pushSendBuffer(const char* buf);
    int     popRecvBuffer(char* buf);
    void    setPlayFlag(int f){m_iPlaying = f;}
protected:
    void    run();
    int     RecvCmd(AUDIO_MSG* pxCmd);
    int     SendCmd(AUDIO_MSG* pxCmd);
    unsigned char CalcCheckSum(AUDIO_MSG* pxCmd);

    int     m_iRunning;
    int     m_iPlaying;
    vector<void*> m_send_buffers;
    vector<void*> m_recv_buffers;

    Mutex m_bLocker;
};

extern Mutex g_xUartMutex;

#endif // UART_BASE_H
