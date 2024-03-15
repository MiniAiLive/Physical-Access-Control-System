#include "keytask.h"
#include "msg.h"
#include "drv_gpio.h"
#include "shared.h"
#include "appdef.h"
#include "i2cbase.h"

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>

#define BTN_DOWN    1
#define BTN_UP      0

KeyTask::KeyTask()
{
    m_iRunning = 0;
    m_iCounter = 0;

    ResetKey();

    memset(m_aiBackKeyState, 0, sizeof(m_aiBackKeyState));
}

KeyTask::~KeyTask()
{

}

void KeyTask::Start()
{
    m_iCounter ++;
    m_iRunning = 1;
    Thread::Start();
}

void KeyTask::Stop()
{
    m_iRunning = 0;
    Thread::Wait();
}

void KeyTask::Exit()
{
    m_iRunning = 0;
    Thread::Exit();

    m_xMutex.Unlock();
}

void KeyTask::ResetKey()
{
    m_xMutex.Lock();
    m_iKeyCount = 0;

    memset(m_aiKey, 0, sizeof(m_aiKey));
    memset(m_aiLongFlag, 0, sizeof(m_aiLongFlag));
    memset(m_aiLongTime, 0, sizeof(m_aiLongTime));
    memset(m_aiBackKeyState, 0, sizeof(m_aiBackKeyState));

    m_xMutex.Unlock();
}

void KeyTask::AddKey(int iKeyID, int iLongFlag, int iLongTime)
{
    if(m_iKeyCount >= MAX_KEY_NUM)
        return;

    m_xMutex.Lock();
    int iExist = -1;
    for(int i = 0; i < m_iKeyCount; i ++)
    {
        if(m_aiKey[i] == iKeyID)
        {
            iExist = i;
            break;
        }
    }

    if(iExist >= 0)
    {
        m_xMutex.Unlock();
        return;
    }
    m_aiKey[m_iKeyCount] = iKeyID;
    m_aiLongFlag[m_iKeyCount] = iLongFlag;
    m_aiLongTime[m_iKeyCount] = iLongTime;
    m_iKeyCount ++;
    m_xMutex.Unlock();
}

void KeyTask::RemoveKey(int iKeyID)
{
    m_xMutex.Lock();
    int iExist = -1;
    for(int i = 0; i < m_iKeyCount; i ++)
    {
        if(m_aiKey[i] == iKeyID)
        {
            iExist = i;
            break;
        }
    }
    if(iExist < 0)
    {
        m_xMutex.Unlock();
        return;
    }

    for(int i = iExist; i < m_iKeyCount - 1; i ++)
    {
        m_aiKey[i] = m_aiKey[i + 1];
        m_aiLongFlag[i] = m_aiLongFlag[i + 1];
        m_aiLongTime[i] = m_aiLongTime[i + 1];
    }

    m_xMutex.Unlock();
}

void KeyTask::run()
{
    float rKeyPressTime[MAX_KEY_NUM] = { 0 };
    float rKeyPressTime1[MAX_KEY_NUM] = { 0 };

    int iFirst = 0;
    int iKeyContinue = 0;
    int iOldSTMInt = 0;
    int iLongFlag = 0;
    int iKeyState = 0;

//    float rOld = Now();
    while(m_iRunning)
    {
        int iKeyID = -1;
        unsigned char abKey[6] = { 0 };

        if(g_xSS.iResetFlag == 1)
        {
            usleep(200*1000);
            continue;
        }
        //stm int re config
#if (GPIO_CLASS_MODE == 0)
        int iSTMInt = GPIO_fast_getvalue(STM_INT) == 0 ? 0 : 1;
#else        
        int iSTMInt = GPIO_class_getvalue(STM_INT) == 0 ? 0 : 1;
#endif

#if 1
        if((iOldSTMInt == 0 && iSTMInt != 0) || (iFirst == 0))
        {
            printf("[Key] Old Int=%d, NewInt=%d\n", iOldSTMInt, iSTMInt);

            iFirst ++;
            int iBellState = MainSTM_Command(MAIN_STM_BELL_STATE);
            if(iBellState == 1)
            {
                if(g_xSS.iBellingFlag == 0)
                    SendGlobalMsg(MSG_BELL, 0, 0, m_iCounter);
                continue;
            }
            int iRet = MainSTM_GetKeyInfos(abKey, sizeof(abKey));
            if(iRet == 1)
            {
                if((abKey[1] == 2) && (iOldSTMInt == 0 && iSTMInt != 0))
                {
                    iKeyID = BTN_LEFT;
                    if(abKey[2] == 1)
                        iKeyState = BTN_DOWN;
                    else if(abKey[2] == 0)
                        iKeyState = BTN_UP;
                }
                else if((abKey[1] == 1) && (iOldSTMInt == 0 && iSTMInt != 0))
                {
                    iKeyID = BTN_MENU;
                    if(abKey[2] == 1)
                        iKeyState = BTN_DOWN;
                    else if(abKey[2] == 0)
                        iKeyState = BTN_UP;
                }
                else if((abKey[1] == 3) && (iOldSTMInt == 0 && iSTMInt != 0))
                {
                    iKeyID = BTN_RIGHT;
                    if(abKey[2] == 1)
                        iKeyState = BTN_DOWN;
                    else if(abKey[2] == 0)
                        iKeyState = BTN_UP;
                }

                if(iFirst == 1 && abKey[1] == 3)
                {
                    if(abKey[3] == 1)
                    {
                        g_xSS.iPoweroffFlag = 1;
                        SendGlobalMsg(MSG_ERROR, abKey[3], 0, m_iCounter);
                        printf("-------------   goto poweroff\n");
                    }
                }
                if(iKeyID != -1)
                {
//                    printf("iPressKey: %d, IRQ: %d, %x, %x\n", iPressKey, abKey[3], abKey[4], abKey[5]);
//                    ResetDetectTimeout();
                }
            }
        }
        else if(iKeyContinue == 1)
        {
            float now = Now();
            for(int i = 0; i < m_iKeyCount; i ++)
            {
                if(m_aiLongFlag[i] == 1 && rKeyPressTime1[i] != 0)
                {
                    if(now - rKeyPressTime1[i] > m_aiLongTime[i])
                    {
                        rKeyPressTime1[i] = now;
                        iLongFlag = 1;
                        SendGlobalMsg(MSG_KEY, m_aiKey[i], KEY_LONG_PRESS, m_iCounter);
//                        usleep(150 * 1000);
                        break;
                    }
                }
            }
        }

        if(iKeyID != -1)
            g_xSS.rTouchTime = Now();
#endif
        iOldSTMInt = iSTMInt;

#if 1
        m_xMutex.Lock();

        for(int i = 0; i < m_iKeyCount; i ++)
        {
            if(m_aiLongFlag[i] == 1)
            {
                if(m_aiKey[i] == iKeyID)
                {
                    if(iKeyState == BTN_DOWN)
                    {
                        iKeyContinue = 1;

                        float now = Now();
                        if(rKeyPressTime[i] == 0)
                        {
                            rKeyPressTime[i] = now;
                            rKeyPressTime1[i] = rKeyPressTime[i];
                        }
                    }
                    else if(iKeyState == BTN_UP)
                    {
                        iKeyContinue = 0;
                        if(Now() - rKeyPressTime1[i] < m_aiLongTime[i])
                        {
                            SendGlobalMsg(MSG_KEY, m_aiKey[i], KEY_CLICKED, m_iCounter);
                            memset(rKeyPressTime, 0, sizeof(rKeyPressTime));
                            memset(rKeyPressTime1, 0, sizeof(rKeyPressTime1));
                            break;
                        }
                        memset(rKeyPressTime, 0, sizeof(rKeyPressTime));
                        memset(rKeyPressTime1, 0, sizeof(rKeyPressTime1));
                    }

#if (AUTO_TEST == 1)
                    exit(0);
#endif
                }

            }
            else
            {
                if(m_aiKey[i] == iKeyID && iKeyState == BTN_DOWN)
                {
                    SendGlobalMsg(MSG_KEY, m_aiKey[i], KEY_CLICKED, m_iCounter);

//                    usleep(150 * 1000);
                    break;
                }
            }
        }

        m_xMutex.Unlock();
#endif
#if (AUTO_TEST == 1)
        if (rand() % 5 == 0)
        {
            SendGlobalMsg(MSG_KEY, BTN_MENU, KEY_CLICKED, m_iCounter);
            usleep(500 *1000);
        }
#endif

        usleep(15 * 1000);
    }
}
