
#include "uartbase.h"
#include "uartcomm.h"
#include "mutex.h"
#include "settings.h"
#include "msg.h"

#include <stdio.h>
#include <memory.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <sys/mman.h>


Mutex g_xUartMutex;

UARTTask::UARTTask()
{
    m_iRunning = 0;
    m_iPlaying = 0;
}

UARTTask::~UARTTask()
{


}

void UARTTask::Start()
{
    m_iRunning = 1;
    Thread::Start();
}

void UARTTask::Stop()
{
    m_iRunning = 0;
    for (unsigned int i = 0; i < m_send_buffers.size(); i++)
    {
        void* b = m_send_buffers.at(i);
        if (b != NULL)
            free(b);
    }
    for (unsigned int i = 0; i < m_recv_buffers.size(); i++)
    {
        void* b = m_recv_buffers.at(i);
        if (b != NULL)
            free(b);
    }
    m_send_buffers.clear();
    m_recv_buffers.clear();
    Thread::Wait();
}

int UARTTask::pushSendBuffer(const char* buf)
{
    if(buf == NULL || m_iPlaying == 0)
        return 0;

    void* buffer = malloc(AUDIO_PACKET_LEN);
    memcpy(buffer, buf, AUDIO_PACKET_LEN);
    m_bLocker.Lock();
    m_send_buffers.push_back(buffer);
    m_bLocker.Unlock();
    return 1;
}

int UARTTask::popRecvBuffer(char* buffer)
{
    if(buffer == NULL)
        return 0;

    if(m_recv_buffers.size() == 0)
        return 0;

    vector<void*>::iterator itr;

    do {
        m_bLocker.Lock();
        itr = m_recv_buffers.begin();
        void* buf = *itr;
        memcpy(buffer, buf, AUDIO_PACKET_LEN);
        m_recv_buffers.erase(itr);
        m_bLocker.Unlock();
        free(buf);
    } while(m_recv_buffers.size() > AUDIO_QUEUE_SIZE);

    return 1;
}

void UARTTask::run()
{
    int a = 0;
    AUDIO_MSG xSendMsg;
    vector<void*>::iterator itr;
    int iRet = 0;
    while(m_iRunning)
    {
        AUDIO_MSG xRecvMsg;
        float r = Now();
#if 1
        if(m_iPlaying)
        {
            iRet = RecvCmd(&xRecvMsg);
            if(iRet == 1)
            {
                void* buffer = malloc(AUDIO_PACKET_LEN);
                if (CalcCheckSum(&xRecvMsg) == xRecvMsg.bCheckSum)
                {
                    memcpy(buffer, xRecvMsg.abBuffer, AUDIO_PACKET_LEN);
                }
                else
                {
                    memset(buffer, 0, AUDIO_PACKET_LEN);
                    printf("__________  checksum error\n");
                }
                m_bLocker.Lock();
                m_recv_buffers.push_back(buffer);
                m_bLocker.Unlock();
#if 0
                FILE* _fp;
                _fp = fopen("/tmp/rec_recv_out.raw", "a+");
                if (_fp)
                {
                    fwrite(buffer, AUDIO_PACKET_LEN, 1, _fp);
                    fclose(_fp);
                }
#endif
            }
            else
                printf("@@@@@@@@@@@@@@@  Failed comm   %d\n", iRet);
//            printf("$$$$$$$ (%d) recv  time  %f(%d)- %d\n", m_iPlaying, Now() - r, iRet, m_recv_buffers.size());
        }
#endif
        int iDelayTime = 20 - (Now() - r);

        if(m_send_buffers.size() != 0)
        {
            do {
                m_bLocker.Lock();
                itr = m_send_buffers.begin();
                void* buf = *itr;
                memcpy(xSendMsg.abBuffer, buf, AUDIO_PACKET_LEN);
                m_send_buffers.erase(itr);
                m_bLocker.Unlock();
                free(buf);
            } while(m_send_buffers.size() > AUDIO_QUEUE_SIZE);

            xSendMsg.bHeader = AUDIO_PACKET_HEADER;
            xSendMsg.bUnmute = g_xSS.iVoiceCallFlag;
            xSendMsg.bCheckSum = CalcCheckSum(&xSendMsg);
            iRet = SendCmd(&xSendMsg);
#if 0
            FILE* _fp;
            _fp = fopen("/tmp/rec_send_out.raw", "a+");
            if (_fp)
            {
                fwrite(xSendMsg.abBuffer, AUDIO_PACKET_LEN, 1, _fp);
                fclose(_fp);
            }
#endif
            if(iDelayTime > 0)
                usleep(iDelayTime * 1000);

//            printf("##################  send time  %f (%d)\n", Now() - r, m_send_buffers.size() + 1);
        }
//        printf("$$$$$$$  one cycle  time  %f   (%d)\n", Now() - r, iDelayTime);
    }
}

int UARTTask::RecvCmd(AUDIO_MSG* pxCmd)
{
    int iRet = 0;
    int iFlag = 0;
    if(pxCmd == NULL)
        return 0;

    float r = Now();

    while(m_iRunning)
    {
        if(Now() - r > 30)
            break;

        iRet = UART_Recv((unsigned char*)pxCmd, 1);
        if(iRet <= 0)
        {
            usleep(1000);
            continue;
        }

        if(*(unsigned char*)pxCmd == 0)
            iFlag++;
        else if(iFlag >= 4 && *(unsigned char*)pxCmd == AUDIO_PACKET_HEADER)
        {
            iFlag = 100;
            break;
        }
        else
            iFlag = 0;

        usleep(500);
    }

    if(iFlag == 100)
    {
        unsigned int iRecvLen = UART_Recv((unsigned char*)pxCmd + 1, sizeof(AUDIO_MSG) - 1);
        iRecvLen += 1;
        if(iRecvLen < sizeof(AUDIO_MSG))
        {
            int len;
            int i;
            for (i = 0; i < 20 && m_iRunning; i ++)
            {
                len = UART_Recv((unsigned char*)pxCmd + iRecvLen, sizeof(AUDIO_MSG) - iRecvLen);
                if (len > 0)
                {
                    iRecvLen += len;
                    if (iRecvLen >= sizeof(AUDIO_MSG))
                        break;
                }
                if(i == 19)
                {
                    printf("$$$$$$$$$$$$$$$$$$$  receive failed  %d\n", iRecvLen);
                    return 0;
                }

                usleep(1000);
            }
        }

//        printf("#################  recv data  %d(%f): %x-%x\n", iRecvLen, Now() - r,pxCmd->bHeader, pxCmd->bCheckSum);
        return 1;
    }
    else{
        printf("##################3   recv failed  no flag 100\n");
    }
//    printf("##################  failed receive data\n");
    return 0;
}

int UARTTask::SendCmd(AUDIO_MSG* pxCmd)
{
    if(pxCmd == NULL)
        return 0;
    unsigned char abBuf[sizeof(AUDIO_MSG) + AUDIO_PACKET_PREFIX_LEN] = {0};
    memcpy(abBuf + AUDIO_PACKET_PREFIX_LEN, pxCmd, sizeof(AUDIO_MSG));
    g_xUartMutex.Lock();
    UART_Send(abBuf, sizeof(AUDIO_MSG) + AUDIO_PACKET_PREFIX_LEN);
    g_xUartMutex.Unlock();

    return 1;
}

unsigned char UARTTask::CalcCheckSum(AUDIO_MSG* pxCmd)
{
    unsigned char* pbData = (unsigned char*)pxCmd;
    int iCheckSum = 0;
    int iCheckLen = sizeof(AUDIO_MSG);

    for(int i = 0; i < iCheckLen - 1; i ++)
        iCheckSum += pbData[i];

    iCheckSum = 0xFF - (iCheckSum & 0xFF);
    return (unsigned char)iCheckSum;
}
