#include "watchtask.h"
#include "unistd.h"
#include "drv_gpio.h"
#include "appdef.h"
#include "shared.h"
#include "msg.h"
#include "i2cbase.h"
#include "lcdtask.h"
#include "settings.h"

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>

WatchTask::WatchTask()
{
    m_iIDCounter = 0;
    m_iRunning = 0;
    m_iTimerCount = 0;
}

WatchTask::~WatchTask()
{

}

void WatchTask::Start()
{
    m_iRunning = 1;
    m_iTimerCount = 0;

    Thread::Start();
}

void WatchTask::Stop()
{
    m_iRunning = 0;
    Thread::Wait();
}

int WatchTask::AddTimer(float iMsec)
{
    if(m_iTimerCount > MAX_TIMER_COUNT)
        return -1;

    m_xTimerMutex.Lock();
    m_iIDCounter ++;
    m_aiTimerIDs[m_iTimerCount] = m_iIDCounter;
    m_aiTimerCounter[m_iTimerCount] = 0;
    m_aiTimerMsec[m_iTimerCount] = iMsec;
    m_arTimerTick[m_iTimerCount] = Now();
    m_xTimerMutex.Unlock();

    return m_iIDCounter;
}

void WatchTask::RemoveTimer(int iTimerID)
{
    m_xTimerMutex.Lock();

    int iExist = -1;
    for(int i = 0; i < m_iTimerCount; i ++)
    {
        if(m_aiTimerIDs[i] == iTimerID)
        {
            iExist = i;
            break;
        }
    }

    if(iExist < 0)
    {
        m_xTimerMutex.Unlock();
        return;
    }

    for(int i = iExist; i < m_iTimerCount - 1; i ++)
    {
        m_aiTimerIDs[i] = m_aiTimerIDs[i + 1];
        m_aiTimerCounter[i] = m_aiTimerCounter[i + 1];
        m_aiTimerMsec[i] = m_aiTimerMsec[i + 1];
        m_arTimerTick[i] = m_arTimerTick[i + 1];
    }
    m_iTimerCount --;
    m_xTimerMutex.Unlock();
}

void WatchTask::ResetTimer(int iTimerID)
{
    m_xTimerMutex.Lock();
    int iExist = -1;
    for(int i = 0; i < m_iTimerCount; i ++)
    {
        if(m_aiTimerIDs[i] == iTimerID)
        {
            iExist = i;
            break;
        }
    }

    m_aiTimerCounter[iExist] ++;
    m_arTimerTick[iExist] = Now();
    m_xTimerMutex.Unlock();
}

int WatchTask::GetCounter(int iTimerID)
{
    m_xTimerMutex.Lock();
    int iExist = -1;
    for(int i = 0; i < m_iTimerCount; i ++)
    {
        if(m_aiTimerIDs[i] == iTimerID)
        {
            iExist = i;
            break;
        }
    }

    if(iExist < 0)
    {
        m_xTimerMutex.Unlock();
        return -1;
    }

    int iRet = 0;
    iRet = m_aiTimerCounter[iExist];
    m_xTimerMutex.Unlock();

    return iRet;
}

void WatchTask::run()
{
    int iROKCounter = 0;
    int iOldState = -1;
    int iFirstMount = 0;
    while(m_iRunning)
    {
        float rNow = Now();
        m_xTimerMutex.Lock();
        for(int i = 0; i < m_iTimerCount; i ++)
        {
            if(rNow - m_arTimerTick[i] > m_aiTimerMsec[i] * 1000)
            {
                m_arTimerTick[i] = rNow;
                SendGlobalMsg(MSG_WATCH, WATCH_TYPE_TIMER, m_aiTimerIDs[i], m_aiTimerCounter[i]);
            }
        }
        m_xTimerMutex.Unlock();

        if((iROKCounter % 10) == 0)
        {
            
            int iRet = MainSTM_Command(MAIN_STM_CMD_ROK);

//            if(getchar())
//            {
//                exit(0);
//            }
        }

        usleep(100 * 1000);
        iROKCounter ++;
    }
}
