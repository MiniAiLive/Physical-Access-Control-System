#include "camerasurface.h"
#include "camera_api.h"
#include "shared.h"
#include "drv_gpio.h"
#include "settings.h"
#include "audiotask.h"

#include <linux/videodev2.h>
#include <stdio.h>
#include <unistd.h>
#include <memory.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <errno.h>
#include <malloc.h>

#include <stdint.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <linux/spi/spidev.h>
#include <sys/types.h>
#include <time.h>
#include <sys/time.h>
#include <dlfcn.h>
#include <string>

#include "i2cbase.h"
#include "keytask.h"
#include "lcdtask.h"
#include "watchtask.h"
#include "msg.h"
#include "themedef.h"
#include "my_lang.h"
#include "uartcomm.h"
#include "uartbase.h"

#include "sys/statvfs.h"
using namespace std;
#define RET_CONTINUE -1
#define RET_POEWROFF 0

CameraSurface* g_pCam = NULL;

WatchTask*  g_pWatchTask = NULL;

#define ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

int GotoMsg(int iMsg);

extern LCDTask* g_pLCDTask;

void DriverInit()
{
    GPIO_fast_init();
#if (GPIO_CLASS_MODE == 0)
    GPIO_fast_config(STM_INT, IN);
    GPIO_fast_config(LCD_BL_EN, OUT);
    GPIO_fast_config(SPK_EN, OUT);
#else
    GPIO_class_config(STM_INT, IN);
    GPIO_class_config(LCD_BL_EN, OUT);
    GPIO_class_config(SPK_EN, OUT);
#endif

    MainSTM_Open();
    UART_Init();
}

void DriverRelease()
{
	UART_Quit();
    MainSTM_Close();
    LCDTask::FB_Release();
    LCDTask::DispClose();
}

void ReleaseAll()
{
    if(g_pWatchTask)
    {
        g_pWatchTask->Stop();
        delete g_pWatchTask;
        g_pWatchTask = NULL;
    }

//    message_queue_destroy(&g_worker);
}

int GotoMsg(int iMsg)
{
    g_xSS.iSystemState = STATUS_NONE;

    int iMsgTimeOut = g_xCS.iShutdownTime;

    LCDTask* pLCDTask = new LCDTask();
    g_pLCDTask = pLCDTask;

    if(iMsg == E_SCENE_MAIN)
        MainSTM_WriteData(MAIN_STM_DISABLE_KEY_FUNCS, 0);
    else
        MainSTM_WriteData(MAIN_STM_DISABLE_KEY_FUNCS, 1);

    KeyTask* pKeyTask = new KeyTask();
    if(iMsg == E_SCENE_MAIN)
    {
        pKeyTask->ResetKey();
        pKeyTask->AddKey(BTN_MENU, 0, 500);
        pKeyTask->AddKey(BTN_LEFT, 1, 800);
        pKeyTask->AddKey(BTN_RIGHT, 1, 500);
        pKeyTask->Start();
    }
    else
    {
        pKeyTask->ResetKey();
        pKeyTask->AddKey(BTN_MENU, 1, 500);
        pKeyTask->AddKey(BTN_LEFT, 1, 300);
        pKeyTask->AddKey(BTN_RIGHT, 1, 300);
        pKeyTask->Start();
    }

    if(iMsg == E_SCENE_MAIN)
    {
#if (AUTO_TEST == 1)
        printf("[TEST]  E_SCENE_MAIN  First Cam  %d\n", g_xSS.iFirstCamInited);
#endif
        if(g_xSS.iFirstCamInited == 0)
        {
            g_pCam = new CameraSurface;
            g_pCam->Start();
        }
        else
            g_xSS.iFirstCamInited = 0;

        ConfigMix();
        StartAudioTask();

        pLCDTask->Init(E_SCENE_MAIN, g_xSS.iSelectedItem, 0);
        LCDTask::LCD_DrawProgress(iMsgTimeOut, iMsgTimeOut);
    }
    else if(iMsg == E_SCENE_SETTING_VIEW)
    {
#if (AUTO_TEST == 1)
        printf("[TEST]  E_SCENE_SETTING_VIEW\n");
#endif
        pLCDTask->Init(E_SCENE_SETTING_VIEW, g_xSS.iSelectedItem, 0);
    }
    else if(iMsg == E_SCENE_LIST_VIEW)
    {
#if (AUTO_TEST == 1)
        printf("[TEST]  E_SCENE_LIST_VIEW\n");
#endif
        pLCDTask->Init(E_SCENE_LIST_VIEW, g_xSS.iSelectedItem, 0);
    }
    else if(iMsg == E_SCENE_MSG_VIEW)
    {
#if (AUTO_TEST == 1)
        printf("[TEST]  E_SCENE_MSG_VIEW\n");
#endif
        pLCDTask->Init(E_SCENE_MSG_VIEW, g_xSS.iSelectedItem, 0);
    }
    else if(iMsg == E_SCENE_MSG_RESET)
    {
#if (AUTO_TEST == 1)
        printf("[TEST]  E_SCENE_MSG_RESET\n");
#endif
        pLCDTask->Init(E_SCENE_MSG_RESET, g_xSS.iSelectedItem, 0);
    }
    else
    {
#if (AUTO_TEST == 1)
        printf("[TEST]  %d\n", iMsg);
#endif
        pLCDTask->Init(iMsg, 1, 0);
    }

    int iTimeoutTimer = -1;
    int iBlinkTimer = -1;
    int iTimeoutCount = iMsgTimeOut;
    iTimeoutTimer = g_pWatchTask->AddTimer(1);


    int iBlinkCount = -1;
    MSG* pMsg = NULL;
    while(1)
    {
        pMsg = (MSG*)message_queue_read(&g_worker);
        if(pMsg->type == MSG_WATCH)
        {
            if(pMsg->data1 == WATCH_TYPE_TIMER)
            {
                if(pMsg->data2 == iTimeoutTimer && g_pWatchTask->GetCounter(iTimeoutTimer) == pMsg->data3)
                {
                    iTimeoutCount--;
                    if(iTimeoutCount < 0)
                    {
                        g_xSS.iSystemState = STATUS_TIMEOUT;
                        break;
                    }
                    if(iMsg == E_SCENE_MAIN)
                    {
                        pLCDTask->DrawClock();
                        LCDTask::LCD_DrawProgress(iTimeoutCount, iMsgTimeOut);
                    }
                }
                else if(pMsg->data2 == iBlinkTimer)
                {
                    iBlinkCount ++;
                    pLCDTask->UpdateStateIcon(iBlinkCount % 2, g_xSS.iVoiceCallFlag);
                    if(iBlinkCount >= 2)
                    {
                        g_pWatchTask->RemoveTimer(iBlinkTimer);
                        g_xSS.iUnlockFlag = 0;
                        iBlinkCount = -1;
                        pLCDTask->UpdateStateIcon(g_xSS.iUnlockFlag, g_xSS.iVoiceCallFlag);
                    }
                }
            }
        }
        else if(pMsg->type == MSG_KEY)
        {
#if (AUTO_TEST == 1)
            int iKey = BTN_MENU + rand() % 3;
            int iEvent = rand() % 2;
            int iRet = pLCDTask->KeyEvent(iKey, iEvent);
#else
            int iRet = pLCDTask->KeyEvent(pMsg->data1, pMsg->data2);
#endif
            if(iMsg == E_SCENE_MAIN)
            {
                if(iRet == LCD_TASK_BACK)
                {
#if (AUTO_TEST == 0)
                    g_xSS.iSystemState = STATUS_TIMEOUT;
                    break;
#endif
                }
                if(iRet == LCD_TASK_LEFT)
                {
                    g_xSS.iVoiceCallFlag = 1 - g_xSS.iVoiceCallFlag;
                    pLCDTask->UpdateStateIcon(g_xSS.iUnlockFlag, g_xSS.iVoiceCallFlag);
//                    MainSTM_WriteData(MAIN_STM_VOICE_CALL_STATE, g_xSS.iVoiceCallFlag);
                    usleep(500 * 1000);
                }
                else if(iRet == LCD_TASK_MENU)
                {
                    g_xSS.iSystemState = STATUS_GOTO_SETTING_VIEW;
                    g_xSS.iSelectedItem = 0;
                    g_xSS.iUnlockFlag = 0;
                    g_xSS.iVoiceCallFlag = 0;
//                    MainSTM_WriteData(MAIN_STM_VOICE_CALL_STATE, 0);
                    usleep(500 * 1000);
                    break;
                }
                else if(iRet == LCD_TASK_OPEN_LOCK)
                {
                    if(g_xSS.iUnlockFlag == 0)
                    {
                        iBlinkTimer = g_pWatchTask->AddTimer(0.5);
                        g_xSS.iUnlockFlag = 1;

                        pLCDTask->UpdateStateIcon(g_xSS.iUnlockFlag, g_xSS.iVoiceCallFlag);
                    }
                }
            }
            else if(iMsg == E_SCENE_SETTING_VIEW)
            {
#if (AUTO_TEST == 0)
                if(iRet == LCD_TASK_BACK)
#else
                if(iRet == LCD_TASK_BACK && rand() % 2 == 0)
#endif
                {
                    g_xSS.iSystemState = STATUS_GOTO_MAIN;
                    break;
                }
                else if(iRet & LCD_TASK_SETTING_BASE)
                {
                    g_xSS.iSelectedItem = iRet - LCD_TASK_SETTING_BASE;

                    if(g_xSS.iSelectedItem == E_SETTING_RESET || g_xSS.iSelectedItem == E_SETTING_VERSION)
                        g_xSS.iSystemState = STATUS_GOTO_MSG_VIEW;
                    else
                        g_xSS.iSystemState = STATUS_GOTO_LIST_VIEW;

                    break;
                }
            }
            else if(iMsg == E_SCENE_LIST_VIEW)
            {
                if(iRet & LCD_TASK_LIST_BASE)
                {
                    int iValue = iRet - LCD_TASK_LIST_BASE;
                    if(g_xSS.iSelectedItem == E_SETTING_AUTO_PWDN)
                    {
                        printf("=============   E_SETTING_AUTO_PWDN  %d\n", iValue);
                        g_xCS.iShutdownTime = (iValue + 2) * SHUTDOWN_TIME_UNIT;
                        UpdateCommonSettings();
                    }
                    else if(g_xSS.iSelectedItem == E_SETTING_LANG)
                    {
                        printf("=============   E_SETTING_LANG  %d\n", iValue);
                        g_xCS.iLang = iValue;
                        SetLang(g_aLangValue[g_xCS.iLang]);
                        UpdateCommonSettings();
                    }

                    g_xSS.iSystemState = STATUS_GOTO_SETTING_VIEW;
                    break;
                }
                else if(iRet == LCD_TASK_SET_TIME)
                {
                    printf("############   set time   %d-%d-%d %d:%d:%d\n", g_xSS.xCurTime.x.iYear, g_xSS.xCurTime.x.iMon, g_xSS.xCurTime.x.iDay, g_xSS.xCurTime.x.iHour, g_xSS.xCurTime.x.iMin, g_xSS.xCurTime.x.iSec);

                    DATETIME_32 xTime = g_xSS.xCurTime;
                    char szDateTime[256];
                    sprintf(szDateTime, "date -s '%d-%02d-%02d %d:%02d:%02d'\n",
                            xTime.x.iYear + 2000, xTime.x.iMon + 1, xTime.x.iDay,
                            xTime.x.iHour, xTime.x.iMin, xTime.x.iSec);
                    system(szDateTime);
                    system("hwclock -u -w");
                    system("date");
                    g_xSS.iSystemState = STATUS_GOTO_SETTING_VIEW;
                    break;
                }
            }
            else if(iMsg == E_SCENE_MSG_VIEW)
            {
                if(iRet == LCD_TASK_YES)
                {
                    if(g_xSS.iSelectedItem == E_SETTING_RESET)
                    {
                        g_xSS.iResetFlag = 1;
                        pLCDTask->Init(E_SCENE_MSG_RESET, -1, 0);
                        ResetSettings();
                        g_xSS.iResetFlag = 0;
                    }
                    g_xSS.iSystemState = STATUS_GOTO_SETTING_VIEW;
                    break;
                }
                else if(iRet == LCD_TASK_NO || iRet == LCD_TASK_BACK)
                {
                    g_xSS.iSystemState = STATUS_GOTO_SETTING_VIEW;
                    break;
                }
            }
            iTimeoutCount = iMsgTimeOut;
            if(iMsg == E_SCENE_MAIN)
                LCDTask::LCD_DrawProgress(iMsgTimeOut, iMsgTimeOut);
        }
        else if(pMsg->type == MSG_ERROR)
        {
            if(pMsg->data1 == 1)
            {
                g_xSS.iSystemState = STATUS_TIMEOUT;
                break;
            }
        }
        else if(pMsg->type == MSG_BELL)
        {
            g_xSS.iBellingFlag = 1;
            if(iMsg == E_SCENE_MAIN)
                StopAudioTask();

            system("tinyplay /test/bell.wav &");

            iTimeoutCount = iMsgTimeOut;
            if(iMsg == E_SCENE_MAIN)
                LCDTask::LCD_DrawProgress(iMsgTimeOut, iMsgTimeOut);

            if(iMsg == E_SCENE_MAIN)
            {
                usleep(1800 * 1000);
                StartAudioTask();
            }

            g_xSS.iBellingFlag = 0;
        }

        message_queue_message_free(&g_worker, (void*)pMsg);
    }

    if(pMsg != NULL)
        message_queue_message_free(&g_worker, (void*)pMsg);

    if(iTimeoutTimer != -1)
        g_pWatchTask->RemoveTimer(iTimeoutTimer);

    if(iBlinkTimer != -1)
        g_pWatchTask->RemoveTimer(iBlinkTimer);

    pKeyTask->Stop();
    delete pKeyTask;

    if(iMsg == E_SCENE_MAIN)
        StopAudioTask();

    if(g_pCam)
    {
        g_pCam->Stop();
        delete g_pCam;
        g_pCam = NULL;
    }

    if (g_xSS.iSystemState != STATUS_GOTO_MSG_VIEW && g_xSS.iSystemState != STATUS_GOTO_LIST_VIEW
            && g_xSS.iSystemState != STATUS_GOTO_SETTING_VIEW && g_xSS.iSystemState != STATUS_GOTO_MAIN)
    {
        LCDTask::LCD_MemClear(0);
        LCDTask::LCD_Update();
    }

    if(g_pLCDTask)
        g_pLCDTask = NULL;

    delete pLCDTask;
    return 0;
}
#if 1
#include <termios.h>
unsigned char abTmp[1024 * 1024];
int main(int argc, char** argv)
{
    printf("Main: %f\n", Now());

#if (AUTO_TEST == 1)
    srand(time(NULL));
#endif

#if 1
    message_queue_init(&g_worker, sizeof(MSG), MAX_MSG_NUM);

    DriverInit();

#if (GPIO_CLASS_MODE == 0)
        GPIO_fast_setvalue(SPK_EN, 1);
#else
        GPIO_class_setvalue(SPK_EN, 1);
#endif

    int iBellState = MainSTM_Command(MAIN_STM_BELL_STATE);
    if(iBellState == 1)
    {
        ConfigMix();
        system("tinyplay /test/bell.wav &");
    }

    float r = Now();

    system("rm /tmp/rec_*");
    int ret = -1;
    for(int i = 0; i < 40; i ++)
    {
        ret = camera_init(CAM_ID, WIDTH_720, HEIGHT_480, FPS, FRAME_NUM);
        if(ret == 0)
        {
            g_xSS.iCamInited = ret;

            g_pCam = new CameraSurface;
            g_pCam->Start();

            g_xSS.iFirstCamInited = 1;
            break;
        }
        usleep(5 * 1000);
    }

    printf("------------------------  %f: (%f)\n", Now() - r, Now());

    float mm = Now();
    system("mount /dev/mtdblock4 /mnt/UDISK");
    printf("@@@@@@@@@@@@@@@   mount time   %f\n", Now() - mm);

    if(IsFirstBoot())
    {
        ResetSettings(false);
        FILE* fp = fopen(MNT_PATH"/first", "wb");
        if(fp)
        {
            fflush(fp);
            fclose(fp);
        }
    }
    else
    {
        int iSetTime = 0;
        DATETIME_32 xCurTime = GetCurDateTime();
        DATETIME_32 xLastTime = GetLastDateTime();
        if(xCurTime.i < xLastTime.i)
        {
            xCurTime = xLastTime;
            iSetTime = 1;
        }

        printf("cur: %d-%d-%d\n", xCurTime.x.iYear + 2000, xCurTime.x.iMon + 1, xCurTime.x.iDay);
        if(xCurTime.x.iYear + 2000 < 2020 || xCurTime.x.iYear + 2000 > 2040)
        {
            char szCmd[256] = { 0 };
            sprintf(szCmd, "date -s '2020-01-01 0:00:00'");
            system(szCmd);
            system("hwclock -u -w");

            xCurTime = GetCurDateTime();
            iSetTime = 0;
        }

        if(iSetTime == 1)
        {
            char szDateTime[256];
            sprintf(szDateTime, "date -s '%d-%02d-%02d %d:%02d:%02d'\n",
                    xCurTime.x.iYear + 2000, xCurTime.x.iMon + 1, xCurTime.x.iDay,
                    xCurTime.x.iHour, xCurTime.x.iMin, xCurTime.x.iSec);
            system(szDateTime);
            system("hwclock -u -w");
            system("date");
        }

        g_xSS.xCurTime = xCurTime;
        system("date");

        ReadCommonSettings();

        SetLang(g_aLangValue[g_xCS.iLang]);
    }

    g_pWatchTask = new WatchTask;
    g_pWatchTask->Start();

    int iMsg = STATUS_GOTO_MAIN;

    int iRet = RET_CONTINUE;
    while(iRet == RET_CONTINUE)
    {
        if(iMsg == STATUS_GOTO_MAIN)
        {
            GotoMsg(E_SCENE_MAIN);
        }
        else if(iMsg == STATUS_GOTO_SETTING_VIEW)
        {
            GotoMsg(E_SCENE_SETTING_VIEW);
        }
        else if(iMsg == STATUS_GOTO_LIST_VIEW)
        {
            GotoMsg(E_SCENE_LIST_VIEW);
        }
        else if(iMsg == STATUS_GOTO_MSG_VIEW)
        {
            GotoMsg(E_SCENE_MSG_VIEW);
        }
        else
        {
            iRet = RET_POEWROFF;
            break;
        }

        if(g_xSS.iSystemState == STATUS_GOTO_MAIN)
        {
            iMsg = STATUS_GOTO_MAIN;
        }
        else if(g_xSS.iSystemState == STATUS_GOTO_SETTING_VIEW)
        {
            iMsg = STATUS_GOTO_SETTING_VIEW;
        }
        else if(g_xSS.iSystemState == STATUS_GOTO_LIST_VIEW)
        {
            iMsg = STATUS_GOTO_LIST_VIEW;
        }
        else if(g_xSS.iSystemState == STATUS_GOTO_MSG_VIEW)
        {
            iMsg = STATUS_GOTO_MSG_VIEW;
        }
        else
            iRet = RET_POEWROFF;
    }

    if(iRet == RET_POEWROFF)
    {
        SaveLastDateTime();
        LCDTask::DispOff();
    }

//    MainSTM_WriteData(MAIN_STM_VOICE_CALL_STATE, 0);

    if(g_pCam)
    {
        g_pCam->Stop();
        delete g_pCam;
        g_pCam = NULL;
    }

#if (GPIO_CLASS_MODE == 0)
        GPIO_fast_setvalue(SPK_EN, 0);
#else
        GPIO_class_setvalue(SPK_EN, 0);
#endif

    ReleaseAll();
    DriverRelease();
    return 0;
#endif
}
#endif
