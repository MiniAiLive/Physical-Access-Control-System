#include "shared.h"
#include "msg.h"
#include "camerasurface.h"
#include "settings.h"
#include "my_lang.h"
#include "keytask.h"
#include "i2cbase.h"

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <linux/fb.h>
#include <wchar.h>
#include <memory.h>
#include <string>
#include <vector>

#include <sys/types.h>
#include "lcdtask.h"
#include "sunxi_display_v1.h"
#include "settings.h"
#include "shared.h"
#include "drv_gpio.h"

using namespace  std;
extern vector<string> g_recordList;

#define SCREEN_0 0
#define ZORDER_MIN 0
#define DISP_DEV_NAME		("/dev/disp")

#define DISP_CLAR(x)        memset(&(x),  0, sizeof(x))

int LCDTask::m_iDispOn = 0;
int LCDTask::m_iDisp = -1;
int LCDTask::m_iVideoStart = 0;

int     LCDTask::m_iFB = -1;
long    LCDTask::m_iScreenSize = 0;
unsigned int*   LCDTask::m_piFB = NULL;
unsigned int    LCDTask::m_piFBMem[MAX_X * MAX_Y] = { 0 };

disp_layer_info g_xLayerInfo;
struct fb_var_screeninfo g_xVinfo;
struct fb_fix_screeninfo g_xFinfo;

unsigned int g_BG_DIALOG_COLOR[3] = {BG_PUXIN_DIALOG_COLOR, BG_PUXIN_DIALOG_COLOR_BR, BG_PUXIN_DIALOG_COLOR_PI};
unsigned int g_BG_BTN_LIGHT_NORMAL_COLOR[3] = {BG_PUXIN_BTN_LIGHT_NORMAL_COLOR, BG_PUXIN_BTN_LIGHT_NORMAL_COLOR_BR, BG_PUXIN_BTN_LIGHT_NORMAL_COLOR_PI};
unsigned int g_BG_BTN_LIGHT_DOWN_COLOR[3] = {BG_PUXIN_BTN_LIGHT_DOWN_COLOR, BG_PUXIN_BTN_LIGHT_DOWN_COLOR_BR, BG_PUXIN_BTN_LIGHT_DOWN_COLOR_PI};
unsigned int g_BG_BTN_LIGHT_NORMAL_COLOR1[3] = {BG_PUXIN_BTN_LIGHT_NORMAL_COLOR1, BG_PUXIN_BTN_LIGHT_NORMAL_COLOR1_BR, BG_PUXIN_BTN_LIGHT_NORMAL_COLOR1_PI};
unsigned int g_BG_BTN_LIGHT_DOWN_COLOR1[3] = {BG_PUXIN_BTN_LIGHT_DOWN_COLOR1, BG_PUXIN_BTN_LIGHT_DOWN_COLOR1_BR, BG_PUXIN_BTN_LIGHT_DOWN_COLOR1_PI};
unsigned int g_BG_MSG_COLOR[3] = {BG_PUXIN_MSG_COLOR, BG_PUXIN_MSG_COLOR_BR, BG_PUXIN_MSG_COLOR_PI};
unsigned int g_BG_MSG_BTN_NORMAL_COLOR[3] = {BG_PUXIN_MSG_BTN_NORMAL_COLOR, BG_PUXIN_MSG_BTN_NORMAL_COLOR_BR, BG_PUXIN_MSG_BTN_NORMAL_COLOR_PI};
unsigned int g_BG_MSG_BTN_DOWN_COLOR[3] = {BG_PUXIN_MSG_BTN_DOWN_COLOR, BG_PUXIN_MSG_BTN_DOWN_COLOR_BR, BG_PUXIN_MSG_BTN_DOWN_COLOR_PI};
unsigned int g_BG_PROGRESS_COLOR[3] = {BG_PUXIN_PROGRESS_COLOR, BG_PUXIN_PROGRESS_COLOR_BR, BG_PUXIN_PROGRESS_COLOR_PI};

static const char* g_background = "/resource/rc/background.png";
static const char* g_ic_unlocked = MY_ROOT_DIR "/resource/rc/ic_unlock.png";
static const char* g_ic_mic_en = MY_ROOT_DIR "/resource/rc/ic_mic_en.png";
static const char* g_ic_mic_dis = MY_ROOT_DIR "/resource/rc/ic_mic_dis.png";



static const char* g_ic_request_unlock_act[3] = {
    "/resource/rc/ic_request_unlock_act.png",
    "/resource/rc/ic_request_unlock_act_br.png",
    "/resource/rc/ic_request_unlock_act_pi.png"
};
static const char* g_ic_request_unlock_nor[3] = {
    "/resource/rc/ic_request_unlock_nor.png",
    "/resource/rc/ic_request_unlock_nor_br.png",
    "/resource/rc/ic_request_unlock_nor_pi.png"
};

static const char* g_ic_shut_time_nor = "/resource/rc/ic_shut_time_nor.png";
static const char* g_ic_shut_time_act = "/resource/rc/ic_shut_time_act.png";
static const char* g_ic_language_nor = "/resource/rc/ic_language_nor.png";
static const char* g_ic_language_act = "/resource/rc/ic_language_act.png";
static const char* g_ic_reset_nor = "/resource/rc/ic_reset_nor.png";
static const char* g_ic_reset_act = "/resource/rc/ic_reset_act.png";
static const char* g_ic_date_time_nor = "/resource/rc/ic_date_time_nor.png";
static const char* g_ic_date_time_act = "/resource/rc/ic_date_time_act.png";
static const char* g_ic_info_nor = "/resource/rc/ic_info_nor.png";
static const char* g_ic_info_act = "/resource/rc/ic_info_act.png";

unsigned int make_ARGB2ABGR(unsigned int color);

LCDTask::LCDTask()
{
    m_iData1 = 0;
    m_iData2 = 0;
    m_iHiddenCodeFlag = 0;
    m_xSetTime = {0};

    m_canvas = new Canvas(MAX_X, MAX_Y, Canvas::DCM_ARGB, (unsigned char*)m_piFBMem);
    m_canvas->SetOtherFont(MyGetDefaultFont());
    m_canvas->SetFontSize(14);
}

LCDTask::~LCDTask()
{
}

void LCDTask::ResetMainButtons()
{
    ResetButtons();
//    AddButton(BTN_ID_ANY, 0, 0, MAX_X - 1, MAX_Y - 1, 0, 0, 0, 0, 0, 0);
}

void LCDTask::Init(int iSceneMode, int iData1, int iData2)
{
    m_iSceneMode = iSceneMode;
    m_iData1 = iData1;
    m_iData2 = iData2;
    m_rResetTime = 0;

    if(iSceneMode == E_SCENE_MAIN)
    {
        ResetMainButtons();
    }
    else if(iSceneMode == E_SCENE_SETTING_VIEW)
    {
        ResetButtons();
        AddSettingButtons(g_xCS.iTheme);
    }
    else if(iSceneMode == E_SCENE_LIST_VIEW)
    {
        ResetButtons();
        DrawMemSceneOverLayAlpha(0x40);
        AddItems(g_xCS.iTheme);
    }
    else if(iSceneMode == E_SCENE_MSG_VIEW)
    {
        ResetButtons();
        int iLang = 1;

        DrawMemSceneOverLayAlpha(0x40);

        if(m_iData1 == E_SETTING_RESET)
            InitMsg(My_Str_Reset_Question, g_xCS.iLang, g_xCS.iTheme);
        else if(m_iData1 == E_SETTING_VERSION)
        {
            char szVersion[256] = { 0 };
            char szSubversion[256] = { 0 };
            int iRet = MainSTM_Command(MAIN_STM_VERSION, (unsigned char*)szSubversion);
            if(iRet == 1)
                sprintf(szVersion, "%s\n%s",DEVICE_FIRMWARE_VERSION, szSubversion);
            else
                sprintf(szVersion, "%s",DEVICE_FIRMWARE_VERSION);

            DrawMsg(My_Str_Version, szVersion, 0, g_xCS.iTheme);
        }
    }
    else if(iSceneMode == E_SCENE_MSG_RESET)
    {
        ResetButtons();
        AddSettingButtons(g_xCS.iTheme);
    }

    Update();
}

void LCDTask::DrawStateIcon(int iUnlockFlag, int iVoiceCallFlag)
{
    if(iUnlockFlag == 1)
        DrawMemOverImage(MAX_X - 2 * IC_STATE_WIDTH - RIGHT_MARGIN, MAX_Y - LCD_STATE_HEIGHT, MAX_X - IC_STATE_WIDTH - RIGHT_MARGIN - 1, MAX_Y - PROGRESSBAR_HEIGHT - 1, g_ic_unlocked, ALIGN_CENTER);

    if(iVoiceCallFlag == 1)
        DrawMemOverImage(MAX_X - IC_STATE_WIDTH - RIGHT_MARGIN, MAX_Y - LCD_STATE_HEIGHT, MAX_X - RIGHT_MARGIN - 1, MAX_Y - PROGRESSBAR_HEIGHT - 1, g_ic_mic_en, ALIGN_CENTER);
    else if(iVoiceCallFlag == 0)
        DrawMemOverImage(MAX_X - IC_STATE_WIDTH - RIGHT_MARGIN, MAX_Y - LCD_STATE_HEIGHT, MAX_X - RIGHT_MARGIN - 1, MAX_Y - PROGRESSBAR_HEIGHT - 1, g_ic_mic_dis, ALIGN_CENTER);
}

void LCDTask::DrawClock()
{
    if(m_iSceneMode == E_SCENE_MAIN)
    {
        DrawMemFillRect(0, MAX_Y - LCD_STATE_HEIGHT, LCD_CLOCK_WIDTH - 1, MAX_Y - PROGRESSBAR_HEIGHT - 1, C_NoBack);
        DrawDateTime(g_xCS.iLang, 1, 0);
        LCD_Update(0, MAX_Y - LCD_STATE_HEIGHT, LCD_CLOCK_WIDTH - 1, MAX_Y - PROGRESSBAR_HEIGHT - 1);
    }
}

void LCDTask::LCD_DrawProgress(int value, int maximum)
{
    int i, j, x_pos, y_pos;
    long location;

    if (value > maximum || maximum == 0)
        return;

    int time_bar_width = value * MAX_X / maximum;
    for (i = 2; i < MAX_X; i++)
    {
        for (j = 0; j < PROGRESSBAR_HEIGHT; j++)
        {
            y_pos = MAX_Y - 1 - j;

            location = y_pos * MAX_X + i;
            if (i <= time_bar_width)
                m_piFB[location] = make_ARGB2ABGR(BG_PUXIN_BTN_LIGHT_DOWN_COLOR);
            else
                m_piFB[location] = 0x60000000;
        }
    }
}

void LCDTask::DrawDateTime(int iLang, int iHourFormat, int iDateFormat)
{
    char* szDayOfWeek[7] =
    {
        My_Str_Monday,
        My_Str_Tuesday,
        My_Str_Wednesday,
        My_Str_Thursday,
        My_Str_Friday,
        My_Str_Saturday,
        My_Str_Sunday
    };

    DATETIME_32 xCurTime = GetCurDateTime();
    if(xCurTime.x.iYear < (2015 - 2000))
    {
        xCurTime.x.iYear = (2015 - 2000);
        xCurTime.x.iMon = 0;
        xCurTime.x.iDay = 1;
        xCurTime.x.iHour = 0;
        xCurTime.x.iMin = 0;
        xCurTime.x.iSec = 0;
    }

    MY_WCHAR text_time[STR_MAX_LEN];
    MY_WCHAR text_date[STR_MAX_LEN];
    MY_WCHAR text_am[STR_MAX_LEN];
    MY_WCHAR text_day[STR_MAX_LEN];

    int iPos = 10;
    int iPosY = MAX_Y - LCD_CLOCK_HEIGHT;
    char szMsg[256] ={ 0 };
    char szTemp[256];
    int iIsAm = -1;

    if(iHourFormat == 0)
    {
        int iHour = xCurTime.x.iHour % 12;
        if(iHour == 0)
            iHour = 12;
        iIsAm = xCurTime.x.iHour / 12;

        sprintf(szMsg, "%d:%02d ", iHour, xCurTime.x.iMin);
    }
    else
    {
        sprintf(szMsg, "%d:%02d ", xCurTime.x.iHour, xCurTime.x.iMin);
    }

    sprintf(szTemp, "%s", szMsg);
    MY_TR(szTemp, text_time);
    if (iIsAm == 0)
        MY_TR("AM", text_am);
    else if (iIsAm == 1)
        MY_TR("PM", text_am);
    else
        text_am[0] = 0;

    wcscat(text_time, text_am);

    if(iDateFormat == 0)
    {
        sprintf(szMsg, "%d/%d/%d ", xCurTime.x.iMon + 1, xCurTime.x.iDay , xCurTime.x.iYear + 2000);
    }
    else
    {
        sprintf(szMsg, "%d-%d-%d ", xCurTime.x.iYear + 2000, xCurTime.x.iMon + 1, xCurTime.x.iDay);
    }

    MY_TR(szMsg, text_date);

    int iDayIndex = GetDayIndexByDate(xCurTime.x.iYear + 2000, xCurTime.x.iMon + 1, xCurTime.x.iDay);
    sprintf(szTemp, "%s", szDayOfWeek[iDayIndex]);
    MY_TR(szTemp, text_day);
    wcscat(text_date, text_day);

    m_canvas->DrawStart();
    m_canvas->SetBackColor(0);
    m_canvas->SetForeColor(C_White);
    m_canvas->SetFontSize(LCD_FOOTER_FONT_SIZE_1);
    m_canvas->DrawTextOut(iPos, iPosY, text_time);
    iPosY = MAX_Y - LCD_CLOCK_HEIGHT / 2 + 5;
    m_canvas->SetFontSize(LCD_FOOTER_FONT_SIZE);
    m_canvas->DrawTextOut(iPos, iPosY, text_date);
    m_canvas->Sync();
    m_canvas->DrawEnd();
}

void LCDTask::DrawMain()
{
    if(g_xSS.iCamInited == -1)
    {
        LCD_MemClear(0xFF000000);
        LCD_DrawText(0, 5, MAX_X - 5, 30, My_Str_CameraError,
                    Canvas::C_TA_Right | Canvas::C_TA_Middle, LCD_NORMAL_FONT_SIZE,
                    make_ARGB2ABGR(C_Red), C_NoBack);
        DrawClock();
        DrawStateIcon(g_xSS.iUnlockFlag, g_xSS.iVoiceCallFlag);
    }
}

void LCDTask::Update()
{
    int iTheme = g_xCS.iTheme;
    if(m_iSceneMode == E_SCENE_MAIN && g_xSS.iSystemError == ERROR_NONE)
    {
        LCD_MemClear(0);
        DrawStateIcon(g_xSS.iUnlockFlag, g_xSS.iVoiceCallFlag);
        DrawClock();
        DrawMain();

        LCD_Update();
    }
    else if(m_iSceneMode == E_SCENE_SETTING_VIEW)
    {
        LCD_MemClear(g_BG_DIALOG_COLOR[iTheme]);
        DrawSettingScene();
        LCD_Update();
    }
    else if(m_iSceneMode == E_SCENE_LIST_VIEW)
    {
        char szTitle[64] = {0};
        if(m_iData1 == E_SETTING_AUTO_PWDN)
        {
            sprintf(szTitle, My_Str_Auto_PowerOff_1);
            DrawMemFillRect((MAX_X - ITEM_BASE_WIDTH) / 2 - ITEM_BASE_GAP_X, ITEM_BASE_Y1(3), (MAX_X + ITEM_BASE_WIDTH) / 2 + ITEM_BASE_GAP_X, ITEM_BASE_Y1(3) + ITEM_TITLE_HEIGHT + 3 * (ITEM_BASE_HEIGHT + ITEM_BASE_GAP_Y), g_BG_MSG_BTN_NORMAL_COLOR[iTheme]);
            DrawListViewScene(szTitle, 3);
        }
        else if(m_iData1 == E_SETTING_LANG)
        {
            sprintf(szTitle, My_Str_Language);
            DrawMemFillRect((MAX_X - ITEM_BASE_WIDTH) / 2 - ITEM_BASE_GAP_X, ITEM_BASE_Y1(3), (MAX_X + ITEM_BASE_WIDTH) / 2 + ITEM_BASE_GAP_X, ITEM_BASE_Y1(3) + ITEM_TITLE_HEIGHT + 3 * (ITEM_BASE_HEIGHT + ITEM_BASE_GAP_Y), g_BG_MSG_BTN_NORMAL_COLOR[iTheme]);
            DrawListViewScene(szTitle, 3);
        }
        else if(m_iData1 == E_SETTING_DATE_TIME)
        {
            sprintf(szTitle, My_Str_Date_Time);
            DrawMemFillRect((MAX_X - ITEM_BASE_WIDTH) / 2 - ITEM_BASE_GAP_X, ITEM_BASE_Y1(2), (MAX_X + ITEM_BASE_WIDTH) / 2 + ITEM_BASE_GAP_X, ITEM_BASE_Y1(2) + ITEM_TITLE_HEIGHT + 2 * (ITEM_BASE_HEIGHT + ITEM_BASE_GAP_Y) + ITEM_BASE_GAP_Y, g_BG_MSG_BTN_NORMAL_COLOR[iTheme]);
            DrawSetTimeViewScene();
        }
        else if(m_iData1 == E_SETTING_VERSION)
        {
            sprintf(szTitle, My_Str_Version);
            DrawMemFillRect((MAX_X - ITEM_BASE_WIDTH) / 2 - ITEM_BASE_GAP_X, ITEM_BASE_Y1(0), (MAX_X + ITEM_BASE_WIDTH) / 2 + ITEM_BASE_GAP_X, ITEM_BASE_Y1(2) + ITEM_TITLE_HEIGHT + 2 * (ITEM_BASE_HEIGHT + ITEM_BASE_GAP_Y), g_BG_MSG_BTN_NORMAL_COLOR[iTheme]);
            DrawListViewScene(szTitle, 0);
        }

        LCD_Update();
    }
    else if(m_iSceneMode == E_SCENE_MSG_VIEW && g_xSS.iSystemError == ERROR_NONE)
    {
        if(m_iData1 == E_SETTING_RESET)
            DrawMsg(My_Str_Warning, My_Str_Reset_Question, 1, iTheme);

        DrawButtons();
        LCD_Update();
    }
    else if(m_iSceneMode == E_SCENE_MSG_RESET && g_xSS.iSystemError == ERROR_NONE)
    {
        LCD_MemClear(g_BG_DIALOG_COLOR[iTheme]);
        DrawSettingScene();
        DrawMemSceneOverLayAlpha(0x40);
        DrawMsg(My_Str_Warning, My_Str_Waiting_Reset, 0, iTheme);
        LCD_Update();
    }
}

void LCDTask::UpdateStateIcon(int iUnlockFlag, int iVoiceCallFlag)
{
    if(m_iSceneMode == E_SCENE_MAIN && g_xSS.iSystemError == ERROR_NONE)
    {
        DrawMemFillRect(MAX_X - LCD_STATE_WIDTH, MAX_Y - LCD_STATE_HEIGHT, MAX_X - 1, MAX_Y - PROGRESSBAR_HEIGHT - 1, C_NoBack);
        DrawStateIcon(iUnlockFlag, iVoiceCallFlag);
        LCD_Update(MAX_X - LCD_STATE_WIDTH, MAX_Y - LCD_STATE_HEIGHT, MAX_X - 1, MAX_Y - PROGRESSBAR_HEIGHT - 1);
    }
}

void LCDTask::AddButton(int iID, int iX1, int iY1, int iX2, int iY2, const char* szTxt, int iFontSize, unsigned int iNormalColor, int iPressColor, const char* szNormalImg, const char* szPressImg, signed char bSelected)
{
    {
        BUTTON xBtn = { 0 };
        xBtn.iID = iID;
        xBtn.iX1 = iX1;
        xBtn.iY1 = iY1;
        xBtn.iX2 = iX2;
        xBtn.iY2 = iY2;

        if(szTxt)
            strcpy(xBtn.szTxt, szTxt);

        if(szNormalImg)
            strcpy(xBtn.szNormalImg, szNormalImg);

        if(szPressImg)
            strcpy(xBtn.szPressImg, szPressImg);

        xBtn.iFontSize = iFontSize;

        xBtn.iNormalColor = iNormalColor;
        xBtn.iPressColor = iPressColor;

        if(bSelected == 1)
            xBtn.iState = BTN_STATE_PRESSED;
        else
            xBtn.iState = BTN_STATE_NONE;

        m_axBtns[m_iBtnCnt] = xBtn;
        m_iBtnCnt ++;
    }
}

void LCDTask::ResetButtons()
{
    m_iBtnCnt = 0;
}

void LCDTask::AddSettingButtons(int iTheme)
{
    BUTTON axPassBtns[E_SETTING_END] = { 0 };

    for(int i = 0; i < E_SETTING_END; i ++)
    {
        int x = i % 3;
        int y = i / 3;
        axPassBtns[i].iX1 = BTN_SETTING_BASE_X1 + x * (BTN_SETTING_BASE_WIDTH + BTN_SETTING_BASE_GAP_X);
        axPassBtns[i].iY1 = BTN_SETTING_BASE_Y1 + y * (BTN_SETTING_BASE_HEIGHT + BTN_SETTING_BASE_GAP_Y);
        axPassBtns[i].iX2 = BTN_SETTING_BASE_X1 + x * (BTN_SETTING_BASE_WIDTH + BTN_SETTING_BASE_GAP_X) + BTN_SETTING_BASE_WIDTH - 1;
        axPassBtns[i].iY2 = BTN_SETTING_BASE_Y1 + y * (BTN_SETTING_BASE_HEIGHT + BTN_SETTING_BASE_GAP_Y) + BTN_SETTING_BASE_HEIGHT - 1;
        axPassBtns[i].iNormalColor = g_BG_DIALOG_COLOR[iTheme];
        axPassBtns[i].iPressColor= g_BG_DIALOG_COLOR[iTheme];
        axPassBtns[i].iID = -1;
    }

    AddButton(E_SETTING_AUTO_PWDN + BTN_ID_SETTING_BASE, axPassBtns[E_SETTING_AUTO_PWDN].iX1, axPassBtns[E_SETTING_AUTO_PWDN].iY1, axPassBtns[E_SETTING_AUTO_PWDN].iX2, axPassBtns[E_SETTING_AUTO_PWDN].iY2,
              My_Str_Auto_PowerOff, LCD_ICON_FONT_SIZE, axPassBtns[E_SETTING_AUTO_PWDN].iNormalColor, axPassBtns[E_SETTING_AUTO_PWDN].iPressColor, g_ic_shut_time_nor, g_ic_shut_time_act, m_iData1 == E_SETTING_AUTO_PWDN);

    AddButton(E_SETTING_LANG + BTN_ID_SETTING_BASE, axPassBtns[E_SETTING_LANG].iX1, axPassBtns[E_SETTING_LANG].iY1, axPassBtns[E_SETTING_LANG].iX2, axPassBtns[E_SETTING_LANG].iY2,
              My_Str_Language, LCD_ICON_FONT_SIZE, axPassBtns[E_SETTING_LANG].iNormalColor, axPassBtns[E_SETTING_LANG].iPressColor, g_ic_language_nor, g_ic_language_act, m_iData1 == E_SETTING_LANG);

    AddButton(E_SETTING_RESET + BTN_ID_SETTING_BASE, axPassBtns[E_SETTING_RESET].iX1, axPassBtns[E_SETTING_RESET].iY1, axPassBtns[E_SETTING_RESET].iX2, axPassBtns[E_SETTING_RESET].iY2,
              My_Str_Factory_Reset, LCD_ICON_FONT_SIZE, axPassBtns[E_SETTING_RESET].iNormalColor, axPassBtns[E_SETTING_RESET].iPressColor, g_ic_reset_nor, g_ic_reset_act, m_iData1 == E_SETTING_RESET);

    AddButton(E_SETTING_DATE_TIME + BTN_ID_SETTING_BASE, axPassBtns[E_SETTING_DATE_TIME].iX1, axPassBtns[E_SETTING_DATE_TIME].iY1, axPassBtns[E_SETTING_DATE_TIME].iX2, axPassBtns[E_SETTING_DATE_TIME].iY2,
              My_Str_Date_Time, LCD_ICON_FONT_SIZE, axPassBtns[E_SETTING_DATE_TIME].iNormalColor, axPassBtns[E_SETTING_DATE_TIME].iPressColor, g_ic_date_time_nor, g_ic_date_time_act, m_iData1 == E_SETTING_DATE_TIME);

    AddButton(E_SETTING_VERSION + BTN_ID_SETTING_BASE, axPassBtns[E_SETTING_VERSION].iX1, axPassBtns[E_SETTING_VERSION].iY1, axPassBtns[E_SETTING_VERSION].iX2, axPassBtns[E_SETTING_VERSION].iY2,
              My_Str_Version, LCD_ICON_FONT_SIZE, axPassBtns[E_SETTING_VERSION].iNormalColor, axPassBtns[E_SETTING_VERSION].iPressColor, g_ic_info_nor, g_ic_info_act, m_iData1 == E_SETTING_VERSION);
}

void LCDTask::AddItems(int iTheme)
{
    if(m_iData1 == E_SETTING_AUTO_PWDN)
    {
        BUTTON axPassBtns[3]  = { 0 };
        for(int i = 0; i < 3; i ++)
        {
            axPassBtns[i].iX1 = (MAX_X - ITEM_BASE_WIDTH) / 2;
            axPassBtns[i].iY1 = ITEM_BASE_Y1(3) + ITEM_TITLE_HEIGHT + i * (ITEM_BASE_HEIGHT + ITEM_BASE_GAP_Y) - 1;
            axPassBtns[i].iX2 = (MAX_X + ITEM_BASE_WIDTH) / 2;
            axPassBtns[i].iY2 = ITEM_BASE_Y1(3) + ITEM_TITLE_HEIGHT + i * (ITEM_BASE_HEIGHT + ITEM_BASE_GAP_Y) + ITEM_BASE_HEIGHT - 1;
            axPassBtns[i].iNormalColor = g_BG_DIALOG_COLOR[iTheme];
            axPassBtns[i].iPressColor = g_BG_BTN_LIGHT_DOWN_COLOR[iTheme];
            axPassBtns[i].iID = -1;
        }

        AddButton(BTN_ID_DETAIL_BASE + 0, axPassBtns[0].iX1, axPassBtns[0].iY1, axPassBtns[0].iX2, axPassBtns[0].iY2,
                  My_Str_10_Seconds, LCD_ITEM_FONT_SIZE, axPassBtns[0].iNormalColor, axPassBtns[0].iPressColor, 0, 0, (g_xCS.iShutdownTime / SHUTDOWN_TIME_UNIT - 2) == 0);

        AddButton(BTN_ID_DETAIL_BASE + 1, axPassBtns[1].iX1, axPassBtns[1].iY1, axPassBtns[1].iX2, axPassBtns[1].iY2,
                  My_Str_15_Seconds, LCD_ITEM_FONT_SIZE, axPassBtns[1].iNormalColor, axPassBtns[1].iPressColor, 0, 0, (g_xCS.iShutdownTime / SHUTDOWN_TIME_UNIT - 2) == 1);

        AddButton(BTN_ID_DETAIL_BASE + 2, axPassBtns[2].iX1, axPassBtns[2].iY1, axPassBtns[2].iX2, axPassBtns[2].iY2,
                  My_Str_20_Seconds, LCD_ITEM_FONT_SIZE, axPassBtns[2].iNormalColor, axPassBtns[2].iPressColor, 0, 0, (g_xCS.iShutdownTime / SHUTDOWN_TIME_UNIT - 2) == 2);
    }
    else if(m_iData1 == E_SETTING_LANG)
    {
        BUTTON axPassBtns[3]  = { 0 };
        for(int i = 0; i < 3; i ++)
        {
            axPassBtns[i].iX1 = (MAX_X - ITEM_BASE_WIDTH) / 2;
            axPassBtns[i].iY1 = ITEM_BASE_Y1(3) + ITEM_TITLE_HEIGHT + i * (ITEM_BASE_HEIGHT + ITEM_BASE_GAP_Y) - 1;
            axPassBtns[i].iX2 = (MAX_X + ITEM_BASE_WIDTH) / 2;
            axPassBtns[i].iY2 = ITEM_BASE_Y1(3) + ITEM_TITLE_HEIGHT + i * (ITEM_BASE_HEIGHT + ITEM_BASE_GAP_Y) + ITEM_BASE_HEIGHT - 1;
            axPassBtns[i].iNormalColor = g_BG_DIALOG_COLOR[iTheme];
            axPassBtns[i].iPressColor = g_BG_BTN_LIGHT_DOWN_COLOR[iTheme];
            axPassBtns[i].iID = -1;
        }

        AddButton(BTN_ID_DETAIL_BASE + 0, axPassBtns[0].iX1, axPassBtns[0].iY1, axPassBtns[0].iX2, axPassBtns[0].iY2,
                  My_Str_CH, LCD_ITEM_FONT_SIZE, axPassBtns[0].iNormalColor, axPassBtns[0].iPressColor, 0, 0, g_xCS.iLang == 0);

        AddButton(BTN_ID_DETAIL_BASE + 1, axPassBtns[1].iX1, axPassBtns[1].iY1, axPassBtns[1].iX2, axPassBtns[1].iY2,
                  My_Str_TW, LCD_ITEM_FONT_SIZE, axPassBtns[1].iNormalColor, axPassBtns[1].iPressColor, 0, 0, g_xCS.iLang == 1);

        AddButton(BTN_ID_DETAIL_BASE + 2, axPassBtns[2].iX1, axPassBtns[2].iY1, axPassBtns[2].iX2, axPassBtns[2].iY2,
                  My_Str_EN, LCD_ITEM_FONT_SIZE, axPassBtns[2].iNormalColor, axPassBtns[2].iPressColor, 0, 0, g_xCS.iLang == 2);

    }
    else if(m_iData1 == E_SETTING_DATE_TIME)
    {
        BUTTON axPassBtns[6]  = { 0 };
        for(int i = 0; i < 6; i ++)
        {
            int x = i % 3;
            int y = i / 3;
            axPassBtns[i].iX1 = ITEM_TIME_X1 + x * (ITEM_TIME_WIDTH + ITEM_TIME_GAP_X);
            axPassBtns[i].iY1 = ITEM_BASE_Y1(2) + ITEM_TITLE_HEIGHT + y * (ITEM_TIME_HEIGHT + ITEM_TIME_GAP_Y);
            axPassBtns[i].iX2 = ITEM_TIME_X1 + x * (ITEM_TIME_WIDTH + ITEM_TIME_GAP_X) + ITEM_TIME_WIDTH - 1;
            axPassBtns[i].iY2 = ITEM_BASE_Y1(2) + ITEM_TITLE_HEIGHT + y * (ITEM_TIME_HEIGHT + ITEM_TIME_GAP_Y) + ITEM_TIME_HEIGHT - 1;
            axPassBtns[i].iNormalColor = g_BG_MSG_BTN_NORMAL_COLOR[iTheme];
            axPassBtns[i].iPressColor = g_BG_BTN_LIGHT_DOWN_COLOR[iTheme];
            axPassBtns[i].iID = -1;
        }

        m_xSetTime = GetCurDateTime();
        printf("%d-%d-%d %d:%d\n",m_xSetTime.x.iYear+2000, m_xSetTime.x.iMon + 1, m_xSetTime.x.iDay, m_xSetTime.x.iHour, m_xSetTime.x.iMin);
        char szTmp[32] = {0};
        sprintf(szTmp, "%d", 2000 + m_xSetTime.x.iYear);
        AddButton(BTN_ID_DETAIL_BASE + 0, axPassBtns[0].iX1, axPassBtns[0].iY1, axPassBtns[0].iX2, axPassBtns[0].iY2,
                  szTmp, LCD_ITEM_FONT_SIZE, axPassBtns[0].iNormalColor, axPassBtns[0].iPressColor, 0, 0, 1);

        sprintf(szTmp, "%d", m_xSetTime.x.iMon + 1);
        AddButton(BTN_ID_DETAIL_BASE + 1, axPassBtns[1].iX1, axPassBtns[1].iY1, axPassBtns[1].iX2, axPassBtns[1].iY2,
                  szTmp, LCD_ITEM_FONT_SIZE, axPassBtns[1].iNormalColor, axPassBtns[1].iPressColor, 0, 0);

        sprintf(szTmp, "%d", m_xSetTime.x.iDay);
        AddButton(BTN_ID_DETAIL_BASE + 2, axPassBtns[2].iX1, axPassBtns[2].iY1, axPassBtns[2].iX2, axPassBtns[2].iY2,
                  szTmp, LCD_ITEM_FONT_SIZE, axPassBtns[2].iNormalColor, axPassBtns[2].iPressColor, 0, 0);

        sprintf(szTmp, "%d", m_xSetTime.x.iHour);
        AddButton(BTN_ID_DETAIL_BASE + 3, axPassBtns[3].iX1, axPassBtns[3].iY1, axPassBtns[3].iX2, axPassBtns[3].iY2,
                  szTmp, LCD_ITEM_FONT_SIZE, axPassBtns[3].iNormalColor, axPassBtns[3].iPressColor, 0, 0);

        sprintf(szTmp, "%d", m_xSetTime.x.iMin);
        AddButton(BTN_ID_DETAIL_BASE + 4, axPassBtns[4].iX1, axPassBtns[4].iY1, axPassBtns[4].iX2, axPassBtns[4].iY2,
                  szTmp, LCD_ITEM_FONT_SIZE, axPassBtns[4].iNormalColor, axPassBtns[4].iPressColor, 0, 0);

        sprintf(szTmp, "%d", m_xSetTime.x.iSec);
        AddButton(BTN_ID_DETAIL_BASE + 5, axPassBtns[5].iX1, axPassBtns[5].iY1, axPassBtns[5].iX2, axPassBtns[5].iY2,
                  szTmp, LCD_ITEM_FONT_SIZE, axPassBtns[5].iNormalColor, axPassBtns[5].iPressColor, 0, 0);
    }
}

int LCDTask::KeyEvent(int iKey, int iEvent)
{
    printf("[LCDTask]::%s   %d    %d\n", __FUNCTION__, iKey, iEvent);
    int iRet = LCD_TASK_CONTINUE;

    if(iEvent == KEY_CLICKED)
    {
        if(iKey == BTN_MENU)
        {
            if(m_iSceneMode == E_SCENE_MAIN)
                return LCD_TASK_OPEN_LOCK;

            if(m_iBtnCnt == 0)
                return LCD_TASK_BACK;

            int iSelectedIdx = -1;

            for(int i = 0; i < m_iBtnCnt; i++)
            {
                if(m_axBtns[i].iState == BTN_STATE_PRESSED)
                {
                    iSelectedIdx = i;
                    break;
                }
            }

            if(iSelectedIdx < 0)
                return iRet;

            if(m_iSceneMode == E_SCENE_SETTING_VIEW)
            {
                iRet = LCD_TASK_SETTING_BASE + iSelectedIdx;
                m_axBtns[iSelectedIdx].iState = BTN_STATE_NONE;
                DrawButtons(m_axBtns[iSelectedIdx].iID);
            }
            else if(m_iSceneMode == E_SCENE_LIST_VIEW)
            {
                if(m_iData1 == E_SETTING_DATE_TIME)
                {
                    m_axBtns[iSelectedIdx].iState = BTN_STATE_NONE;
                    DrawButtons(m_axBtns[iSelectedIdx].iID);
                    int i = 0;
                    {
                        i ++;
                        iSelectedIdx ++;
                        if(iSelectedIdx >= m_iBtnCnt)
                        {
                            iRet = LCD_TASK_SET_TIME;
                            g_xSS.xCurTime = m_xSetTime;
                        }

                        m_axBtns[iSelectedIdx].iState = BTN_STATE_PRESSED;
                        DrawButtons(m_axBtns[iSelectedIdx].iID);
                    }
                }
                else
                    iRet = LCD_TASK_LIST_BASE + iSelectedIdx;
            }
            else if(m_iSceneMode == E_SCENE_MSG_VIEW)
                iRet = (iSelectedIdx == 0) ? LCD_TASK_YES : LCD_TASK_NO;
        }
        else if(iKey == BTN_RIGHT)
        {
            if(m_iSceneMode == E_SCENE_MAIN)
                return LCD_TASK_RIGHT;

            if(m_iBtnCnt == 0)
                return LCD_TASK_BACK;

            int iSelectedIdx = -1;
            for(int i = 0; i < m_iBtnCnt; i++)
            {
                if(m_axBtns[i].iState == BTN_STATE_PRESSED)
                {
                    iSelectedIdx = i;
                    break;
                }
            }

            if(iSelectedIdx < 0)
                return iRet;

            if(m_iSceneMode == E_SCENE_LIST_VIEW && m_iData1 == E_SETTING_DATE_TIME)
            {
                int iVal = 0;
                if(iSelectedIdx == 0)
                {
                    m_xSetTime.x.iYear++;
                    if(m_xSetTime.x.iYear > 40)
                        m_xSetTime.x.iYear = 20;

                    iVal = (2000 + m_xSetTime.x.iYear);
                }
                else if(iSelectedIdx == 1)
                {
                    m_xSetTime.x.iMon++;
                    if(m_xSetTime.x.iMon > 11)
                        m_xSetTime.x.iMon = 0;
                    iVal = m_xSetTime.x.iMon + 1;
                }
                else if(iSelectedIdx == 2)
                {
                    int days_of_month[12] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
                    m_xSetTime.x.iDay++;
                    if (m_xSetTime.x.iYear % 400 == 0 || (m_xSetTime.x.iYear % 4 == 0 && m_xSetTime.x.iYear % 100 != 0)) //leap year
                        days_of_month[1] = 29;
                    if (m_xSetTime.x.iDay > days_of_month[m_xSetTime.x.iMon])
                        m_xSetTime.x.iDay = 1;
                    iVal = m_xSetTime.x.iDay;
                }
                else if(iSelectedIdx == 3)
                {
                    m_xSetTime.x.iHour++;
                    if(m_xSetTime.x.iHour > 23)
                        m_xSetTime.x.iHour = 0;
                    iVal = m_xSetTime.x.iHour;
                }
                else if(iSelectedIdx == 4)
                {
                    m_xSetTime.x.iMin++;
                    if(m_xSetTime.x.iMin >= 60)
                        m_xSetTime.x.iMin = 0;
                    iVal = m_xSetTime.x.iMin;
                }
                else if(iSelectedIdx == 5)
                {
                    m_xSetTime.x.iSec++;
                    if(m_xSetTime.x.iSec >= 60)
                        m_xSetTime.x.iSec = 0;
                    iVal = m_xSetTime.x.iSec;
                }

                sprintf(m_axBtns[iSelectedIdx].szTxt, "%d", iVal);
                DrawButtons(m_axBtns[iSelectedIdx].iID);
            }
            else
            {
                m_axBtns[iSelectedIdx].iState = BTN_STATE_NONE;
                DrawButtons(m_axBtns[iSelectedIdx].iID);

                int i = 0;
    //            while(i < m_iBtnCnt)
                {
                    i ++;
                    iSelectedIdx ++;
                    if(iSelectedIdx >= m_iBtnCnt)
                        iSelectedIdx = 0;

                    m_axBtns[iSelectedIdx].iState = BTN_STATE_PRESSED;
                    DrawButtons(m_axBtns[iSelectedIdx].iID);
                }
            }
        }
        else if(iKey == BTN_LEFT)
        {
            if(m_iSceneMode == E_SCENE_MAIN)
                return LCD_TASK_LEFT;

            if(m_iBtnCnt == 0)
                return LCD_TASK_BACK;

            int iSelectedIdx = -1;
            for(int i = 0; i < m_iBtnCnt; i++)
            {
                if(m_axBtns[i].iState == BTN_STATE_PRESSED)
                {
                    iSelectedIdx = i;
                    break;
                }
            }

            if(iSelectedIdx < 0)
                return iRet;

            if(m_iSceneMode == E_SCENE_LIST_VIEW && m_iData1 == E_SETTING_DATE_TIME)
            {
                int iVal = 0;

                if(iSelectedIdx == 0)
                {
                    m_xSetTime.x.iYear--;
                    if(m_xSetTime.x.iYear < 20)
                        m_xSetTime.x.iYear = 40;

                    iVal = (2000 + m_xSetTime.x.iYear);
                }
                else if(iSelectedIdx == 1)
                {
                    if(m_xSetTime.x.iMon == 0)
                        m_xSetTime.x.iMon = 12;
                    m_xSetTime.x.iMon--;
                    iVal = m_xSetTime.x.iMon + 1;
                }
                else if(iSelectedIdx == 2)
                {
                    int days_of_month[12] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
                    m_xSetTime.x.iDay--;
                    if (m_xSetTime.x.iYear % 400 == 0 || (m_xSetTime.x.iYear % 4 == 0 && m_xSetTime.x.iYear % 100 != 0)) //leap year
                        days_of_month[1] = 29;
                    if (m_xSetTime.x.iDay < 1)
                        m_xSetTime.x.iDay = days_of_month[m_xSetTime.x.iMon];
                    iVal = m_xSetTime.x.iDay;
                }
                else if(iSelectedIdx == 3)
                {
                    if(m_xSetTime.x.iHour == 0)
                        m_xSetTime.x.iHour = 24;
                    m_xSetTime.x.iHour--;

                    iVal = m_xSetTime.x.iHour;
                }
                else if(iSelectedIdx == 4)
                {
                    if(m_xSetTime.x.iMin == 0)
                        m_xSetTime.x.iMin = 60;

                    m_xSetTime.x.iMin--;
                    iVal = m_xSetTime.x.iMin;
                }
                else if(iSelectedIdx == 5)
                {
                    if(m_xSetTime.x.iSec == 0)
                        m_xSetTime.x.iSec = 60;
                    m_xSetTime.x.iSec--;
                    iVal = m_xSetTime.x.iSec;
                }

                sprintf(m_axBtns[iSelectedIdx].szTxt, "%d", iVal);
                DrawButtons(m_axBtns[iSelectedIdx].iID);
            }
            else
            {
                m_axBtns[iSelectedIdx].iState = BTN_STATE_NONE;
                DrawButtons(m_axBtns[iSelectedIdx].iID);


                int i = 0;
    //            while(i < m_iBtnCnt)
                {
                    i ++;
                    iSelectedIdx --;
                    if(iSelectedIdx < 0)
                        iSelectedIdx = m_iBtnCnt - 1;

                    m_axBtns[iSelectedIdx].iState = BTN_STATE_PRESSED;
                    DrawButtons(m_axBtns[iSelectedIdx].iID);
                }
            }
        }
    }
    else if(iEvent == KEY_LONG_PRESS)
    {              
        if(iKey == BTN_MENU)
        {
            if(m_iSceneMode != E_SCENE_MAIN)
                iRet = LCD_TASK_BACK;
        }
        else if(iKey == BTN_LEFT)
        {
            if(m_iSceneMode == E_SCENE_MAIN)
                return LCD_TASK_MENU;
            else
                iRet = KeyEvent(BTN_LEFT, KEY_CLICKED);
        }
        else if(iKey == BTN_RIGHT)
        {
            if(m_iSceneMode == E_SCENE_MAIN)
                return LCD_TASK_BACK;
            else
                iRet = KeyEvent(BTN_RIGHT, KEY_CLICKED);
        }
    }

    LCD_Update();
    return iRet;
}

void LCDTask::DrawSettingScene()
{
    int TITLE_X1 = BTN_SETTING_BASE_X1;
    int TITLE_Y1 = 0;
    int TITLE_X2 = (MAX_X - 1);
    int TITLE_Y2 = 40;

    LCD_DrawText(TITLE_X1, TITLE_Y1, TITLE_X2, TITLE_Y2, My_Str_Settings,
                     Canvas::C_TA_Left | Canvas::C_TA_Middle,
                     LCD_TITLE_FONT_SIZE, C_White, C_NoBack);

    DrawButtons();
}

void LCDTask::DrawListViewScene(const char* szTitle, int iItemCount)
{
    int TITLE_X1 = (MAX_X - ITEM_BASE_WIDTH) / 2 + 12;
    int TITLE_Y1 = ITEM_BASE_Y1(iItemCount);
    int TITLE_X2 = (MAX_X - TITLE_X1 - 1);
    int TITLE_Y2 = TITLE_Y1 + ITEM_TITLE_HEIGHT;

    LCD_DrawText(TITLE_X1, TITLE_Y1, TITLE_X2, TITLE_Y2, szTitle,
                     Canvas::C_TA_Left | Canvas::C_TA_Middle,
                     LCD_MSGBOX_TITLE_FONT_SIZE, C_White, C_NoBack);

    DrawButtons();
}

void LCDTask::DrawSetTimeViewScene()
{
    int TITLE_X1 = (MAX_X - ITEM_BASE_WIDTH) / 2 + 12;
    int TITLE_Y1 = ITEM_BASE_Y1(2);
    int TITLE_X2 = (MAX_X - TITLE_X1 - 1);
    int TITLE_Y2 = TITLE_Y1 + ITEM_TITLE_HEIGHT;

    int iX1 = ITEM_TIME_X1 + ITEM_TIME_WIDTH;
    int iY1 = ITEM_BASE_Y1(2) + ITEM_TITLE_HEIGHT;
    int iY2 = ITEM_BASE_Y1(2) + ITEM_TITLE_HEIGHT + ITEM_TIME_HEIGHT - 1;

    LCD_DrawText(TITLE_X1, TITLE_Y1, TITLE_X2, TITLE_Y2, My_Str_Date_Time,
                     Canvas::C_TA_Left | Canvas::C_TA_Middle,
                     LCD_MSGBOX_TITLE_FONT_SIZE, C_White, C_NoBack);

    LCD_DrawText(iX1, iY1, iX1 + ITEM_TIME_GAP_X, iY2, "/",
                     Canvas::C_TA_Center | Canvas::C_TA_Middle,
                     LCD_TITLE_FONT_SIZE, C_Grey, C_NoBack);

    LCD_DrawText(iX1 + ITEM_TIME_GAP_X + ITEM_TIME_WIDTH, iY1, iX1 + 2 * ITEM_TIME_GAP_X + ITEM_TIME_WIDTH, iY2, "/",
                     Canvas::C_TA_Center | Canvas::C_TA_Middle,
                     LCD_TITLE_FONT_SIZE, C_Grey, C_NoBack);

    LCD_DrawText(iX1, iY1 + ITEM_TIME_HEIGHT + ITEM_TIME_GAP_Y - 5, iX1 + ITEM_TIME_GAP_X, iY2 + ITEM_TIME_HEIGHT + ITEM_TIME_GAP_Y - 5, ":",
                     Canvas::C_TA_Center | Canvas::C_TA_Middle,
                     LCD_TITLE_FONT_SIZE, C_Grey, C_NoBack);

    LCD_DrawText(iX1 + ITEM_TIME_GAP_X + ITEM_TIME_WIDTH, iY1 + ITEM_TIME_HEIGHT + ITEM_TIME_GAP_Y - 5, iX1 + 2 * ITEM_TIME_GAP_X + ITEM_TIME_WIDTH, iY2 + ITEM_TIME_HEIGHT + ITEM_TIME_GAP_Y - 5, ":",
                     Canvas::C_TA_Center | Canvas::C_TA_Middle,
                     LCD_TITLE_FONT_SIZE, C_Grey, C_NoBack);
    DrawButtons();
}

void LCDTask::DrawButtons(int iID)
{
    for(int i = 0; i < m_iBtnCnt; i ++)
    {
        if(iID != -1 && m_axBtns[i].iID != iID)
            continue;

        if(m_axBtns[i].iID == BTN_ID_ANY)
            continue;

        if(m_axBtns[i].iState == BTN_STATE_NONE)
        {
            if(m_axBtns[i].iNormalColor != 0)
                DrawMemFillRect(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2, m_axBtns[i].iY2, m_axBtns[i].iNormalColor);
            {
                int iAlign = Canvas::C_TA_Center | Canvas::C_TA_Bottom;

                if(strlen(m_axBtns[i].szNormalImg))
                    DrawMemOverImage(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2, m_axBtns[i].iY2, m_axBtns[i].szNormalImg, ALIGN_TOP | ALIGN_HOR_CENTER);
                else
                    iAlign = Canvas::C_TA_Center | Canvas::C_TA_Middle;

                LCD_DrawText(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2, m_axBtns[i].iY2, m_axBtns[i].szTxt,
                         iAlign, m_axBtns[i].iFontSize, C_Grey, C_NoBack);
            }
        }
        else
        {
            if(m_axBtns[i].iPressColor != 0)
                DrawMemFillRect(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2, m_axBtns[i].iY2, m_axBtns[i].iPressColor);
            {
                int iAlign = Canvas::C_TA_Center | Canvas::C_TA_Bottom;

                if(strlen(m_axBtns[i].szPressImg))
                    DrawMemOverImage(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2, m_axBtns[i].iY2, m_axBtns[i].szPressImg, ALIGN_TOP | ALIGN_HOR_CENTER);
                else
                    iAlign = Canvas::C_TA_Center | Canvas::C_TA_Middle;

                LCD_DrawText(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2, m_axBtns[i].iY2, m_axBtns[i].szTxt,
                             iAlign, m_axBtns[i].iFontSize, C_White, C_NoBack);
            }
        }
    }
}

void LCDTask::InitMsg(const char* szMsg, int iLang, int iTheme)
{
    int iMsgX = 0, iMsgY = 0, iMsgWidth = 0, iMsgHeight = 0, iContentsHeight = 0;
    GetMsgPosInfo(szMsg, 1, iMsgX, iMsgY, iMsgWidth, iMsgHeight, iContentsHeight, LCD_NORMAL_FONT_SIZE);

    int iBtnWidth = (iMsgWidth - MSG_MARGIN_X * 2 - BTN_MARGIN_LEFT) / 2;
    int iYes_x1 = iMsgX + MSG_MARGIN_X + BTN_MARGIN_LEFT;
    int iYes_y1 = iMsgY + iMsgHeight - MSG_MARGIN_X - BTN_HEIGHT + 5;
    int iYes_x2 = iYes_x1 + iBtnWidth;
    int iYes_y2 = iYes_y1 + BTN_HEIGHT;

    int iNo_x1 = iYes_x2;
    int iNo_y1 = iYes_y1;
    int iNo_x2 = iMsgX + iMsgWidth - MSG_MARGIN_X;
    int iNo_y2 = iYes_y2;

    AddButton(BTN_ID_YES, iYes_x1, iYes_y1, iYes_x2, iYes_y2, My_Str_Yes,
              LCD_NORMAL_FONT_SIZE, g_BG_MSG_BTN_NORMAL_COLOR[iTheme], g_BG_MSG_BTN_DOWN_COLOR[iTheme], 0, 0);

    AddButton(BTN_ID_NO, iNo_x1, iNo_y1, iNo_x2, iNo_y2, My_Str_No,
              LCD_NORMAL_FONT_SIZE, g_BG_MSG_BTN_NORMAL_COLOR[iTheme], g_BG_MSG_BTN_DOWN_COLOR[iTheme], 0, 0, 1);
}

void LCDTask::DrawMsg(const char* szTitlePath, const char* szMsgPath, int iHasBtn, int iTheme)
{
    int iMsgX = 0, iMsgY = 0, iMsgWidth = 0, iMsgHeight = 0, iContentsHeight = 0;
    GetMsgPosInfo(szMsgPath, iHasBtn, iMsgX, iMsgY, iMsgWidth, iMsgHeight, iContentsHeight, LCD_NORMAL_FONT_SIZE);

    DrawMemFillRect(iMsgX, iMsgY, iMsgX + iMsgWidth, iMsgY + iMsgHeight, g_BG_MSG_COLOR[iTheme]);
    LCD_DrawText(iMsgX + MSG_MARGIN_X, iMsgY + MSG_MARGIN_Y, iMsgX + iMsgWidth - MSG_MARGIN_X, iMsgY + MSG_MARGIN_Y + TITLE_HEIGHT, szTitlePath,
                 Canvas::C_TA_Left | Canvas::C_TA_Top, LCD_MSGBOX_TITLE_FONT_SIZE, C_White, C_NoBack);
    LCD_DrawText(iMsgX + MSG_MARGIN_X, iMsgY + MSG_MARGIN_Y + TITLE_HEIGHT, iMsgX + iMsgWidth - MSG_MARGIN_X, iMsgY + MSG_MARGIN_Y + TITLE_HEIGHT + iContentsHeight,
                 szMsgPath,
                 Canvas::C_TA_Center | Canvas::C_TA_Top, LCD_NORMAL_FONT_SIZE, C_White, C_NoBack);
}

void LCDTask::DrawMemFullImg(unsigned char* pbImg32)
{
    memcpy(m_piFBMem, pbImg32, MAX_X * MAX_Y * 4);
}


void LCDTask::DrawMemFillRect(int x0, int y0, int x1, int y1, int color)
{
    short temp;

    if( x0 > x1 )
    {
      temp = x1;
      x1 = x0;
      x0 = temp;
    }
    if( y0 > y1 )
    {
      temp = y1;
      y1 = y0;
      y0 = temp;
    }

    if(x0 < 0)
        x0 = 0;
    if(y0 < 0)
        y0 = 0;
    if(x0 >= MAX_X)
        x0 = MAX_X - 1;
    if(y0 >= MAX_Y)
        y0 = MAX_Y - 1;

    if(x1 < 0)
        x1 = 0;
    if(y1 < 0)
        y1 = 0;
    if(x1 >= MAX_X)
        x1 = MAX_X - 1;
    if(y1 >= MAX_Y)
        y1 = MAX_Y - 1;

    for(int i = y0; i <= y1; i ++)
    {
        for(int j = x0; j <= x1; j ++)
            m_piFBMem[MAX_X * i + j] = make_ARGB2ABGR(color);
    }
}

void LCDTask::DrawMemSceneOverLayAlpha(int iApha)
{
    for(int i = 0; i < MAX_X * MAX_Y; i ++)
    {
        int iColor = m_piFBMem[i];
        unsigned int out_R = MIN(255, C_RED(iColor) * iApha / 0xFF);
        unsigned int out_G = MIN(255, C_GREEN(iColor) * iApha / 0xFF);
        unsigned int out_B = MIN(255, C_BLUE(iColor) * iApha / 0xFF);

        m_piFBMem[i] = MAKE_COLOR(C_ALPHA(iColor), out_R, out_G, out_B);
    }
}

void LCDTask::GetImgInfo(const char* filePath, int* pnImgWidth, int* pnImgHeight)
{
    FILE* fp = fopen(filePath, "rb");
    if(fp == NULL)
        return;

    int imgWidth = 0, imgHeight = 0;
    fread(&imgWidth, sizeof(int), 1, fp);
    fread(&imgHeight, sizeof(int), 1, fp);
    fclose(fp);

    if(pnImgWidth)
        *pnImgWidth = imgWidth;

    if(pnImgHeight)
        *pnImgHeight = imgHeight;
}


void LCDTask::GetMsgPosInfo(const char* szMsgPath, int iHasBtn, int& iMsgX, int& iMsgY, int& iMsgWidth, int& iMsgHeight, int& iContentHeight, int iFontSize)
{
#ifndef _NO_ENGINE_
    MY_WCHAR text[STR_MAX_LEN];

    m_canvas->DrawStart();
    m_canvas->SetFontSize(iFontSize);

    iContentHeight = m_canvas->GetTextHeight(MY_TR(szMsgPath, text));
    iContentHeight += MSG_MARGIN_Y * 2;

    m_canvas->DrawEnd();

    iMsgHeight = iContentHeight + MSG_MARGIN_Y * 2 + TITLE_HEIGHT + (BTN_HEIGHT) * iHasBtn + 10;
    iMsgWidth = MSG_WIDTH;
    iMsgX = (MAX_X - iMsgWidth) / 2;
    iMsgY = (MAX_Y - iMsgHeight) / 2;
#endif
}

int LCDTask::LCD_DrawText(int x0, int y0, int x1, int y1, const char* str_text, int iAlign, int iFontSize, unsigned int iColor, unsigned int iBackColor)
{
#ifndef _NO_ENGINE_
    MY_WCHAR text[STR_MAX_LEN];
    m_canvas->DrawStart();
    if (iBackColor == C_NoBack)
        iBackColor = 0;
    m_canvas->SetBackColor(iBackColor);
    m_canvas->SetForeColor(iColor);
    m_canvas->SetFontSize(iFontSize);
    //iAlign = Canvas::C_TA_Center | Canvas::C_TA_Top;
    m_canvas->DrawTextOut(x0, y0, MY_TR(str_text, text), x1 - x0, iAlign, y1 - y0);
    m_canvas->Sync();
    m_canvas->DrawEnd();
#endif
    return 0;
}

int LCDTask::DispOpen()
{
    if(m_iDisp > 0)
        return -1;

    if ((m_iDisp = open(DISP_DEV_NAME, O_RDWR)) == -1)
    {
        printf("can't open /dev/disp(%s)\n", strerror(errno));
        return -1;
    }

    printf("LCDTask: Disp = %d\n", m_iDisp);

    return m_iDisp;
}

void LCDTask::DispClose()
{
    int ret = 0;

    ret = close(m_iDisp);
    if(ret != 0){
        printf("%s:close disp layer frame buffer handle failed!\r\n", __func__);
    }

    m_iDisp = -1;
}

int LCDTask::FB_Init()
{
    unsigned int args[6] = { 0 };
    if(m_iDisp == -1)
        return -1;

    if(m_iFB != -1)
        return -1;

    //////////////////////////////FB Init/////////////////////////
    int iFB = -1;
    if((iFB = open("/dev/fb0", O_RDWR)) < 0)
    {
        printf("open fb0 fail!!!\n");
        return -1;
    }

    if (ioctl(iFB, FBIOGET_FSCREENINFO, &g_xFinfo) == -1)
    {
        printf("Error reading fixed information.\n");
    }

    // Get variable screen information
    if (ioctl(iFB, FBIOGET_VSCREENINFO, &g_xVinfo) == -1)
    {
        printf("Error reading variable information.\n");
    }

    ///////////////////////////FB Map/////////////////////////////////////

    m_iScreenSize = g_xVinfo.xres * g_xVinfo.yres * g_xVinfo.bits_per_pixel / 8;

    m_piFB = (unsigned int*)mmap(NULL, m_iScreenSize, PROT_READ | PROT_WRITE, MAP_SHARED, iFB, 0);
    memset(m_piFB, 0xFF000000, m_iScreenSize);

#if 1
    printf("vinfo.xoffset = %d\n vinfo.yoffset = %d\n vinfo.bits_per_pixel = %d\n finfo.line_length=%d\n", g_xVinfo.xoffset, g_xVinfo.yoffset, g_xVinfo.bits_per_pixel, g_xFinfo.line_length);
    printf("vinfo.xres = %d\n", g_xVinfo.xres);
    printf("vinfo.yres = %d\n", g_xVinfo.yres);
#endif

    m_iFB = iFB;
    return m_iFB;
}

void LCDTask::FB_Release()
{
    unsigned int args[4];
    if(m_iFB == -1)
        return;

    if(m_piFB)
    {
        munmap(m_piFB, m_iScreenSize);
        m_piFB = NULL;
    }

    if (close (m_iFB) < 0)
        perror ("framebuffer close");

    m_iFB = -1;
}

static int LayerConfig(int fd, __disp_cmd_t cmd, unsigned int hlay, disp_layer_info *pinfo)
{
    unsigned long args[4] = {0};
    unsigned int ret = 0;
    args[0] = SCREEN_0;
    args[1] = 0;
    args[2] = (unsigned long)pinfo;
    ret = ioctl(fd, cmd, args);
    if(ret != 0)
    {
        printf("fail to get para\n");
        return -1;
    }

    return 0;
}

static int LayerGetPara(int disp_fd, unsigned int hlay, disp_layer_info *pinfo)
{
    return LayerConfig(disp_fd, DISP_CMD_LAYER_GET_INFO, hlay, pinfo);
}

static int LayerSetPara(int disp_fd, unsigned int hlay, disp_layer_info *pinfo)
{
    return LayerConfig(disp_fd, DISP_CMD_LAYER_SET_INFO, hlay, pinfo);
}

int LCDTask::VideoStart(int iFormat, int iWinX, int iWinY, int iWinW, int iWinH)
{
    int ret = 0;
    unsigned int width, height;
    unsigned int ioctlParam[4];

    if(m_iVideoStart != 0)
        return -1;

//    disp_layer_info config;
    if(m_iDisp < 0)
    {
        printf("%s:unavailable disp handle!\r\n", __func__);
        return -1;
    }

    DISP_CLAR(ioctlParam);

    ioctlParam[0] = SCREEN_0;
    ioctlParam[1] = 0;
    width = ioctl(m_iDisp, DISP_CMD_GET_SCN_WIDTH, ioctlParam);
    height = ioctl(m_iDisp, DISP_CMD_GET_SCN_HEIGHT, ioctlParam);
    memset(&g_xLayerInfo, 0, sizeof(disp_layer_info));
    g_xLayerInfo.fb.size.width = width;
    g_xLayerInfo.fb.size.height = height;
    g_xLayerInfo.fb.format = (disp_pixel_format)iFormat;

    g_xLayerInfo.screen_win.x = iWinX;
    g_xLayerInfo.screen_win.y = iWinY;
    g_xLayerInfo.screen_win.width = iWinW;
    g_xLayerInfo.screen_win.height = iWinH;

    g_xLayerInfo.alpha_mode = 0;
    g_xLayerInfo.fb.pre_multiply = 0;
    g_xLayerInfo.alpha_value = 0xff;
    g_xLayerInfo.zorder = 0;
    g_xLayerInfo.mode = DISP_LAYER_WORK_MODE_SCALER;
    g_xLayerInfo.pipe = 0;

    DISP_CLAR(ioctlParam);
    ioctlParam[0] = SCREEN_0;
    ioctlParam[1] = 0;

    ret = ioctl(m_iDisp, DISP_CMD_LAYER_ENABLE, (void*)ioctlParam);
    if(ret < 0)
    {
        printf("%s:ioctl disp disp off failed!\r\n", __func__);
        return -1;
    }

    m_iVideoStart = 1;

    return 1;
}

void LCDTask::VideoStop()
{
    unsigned int ioctlParam[4];

    if(m_iVideoStart == 0)
        return;

    if(m_iDisp < 0)
    {
        printf("%s:unavailable disp handle!\r\n", __func__);
        return;
    }

    DISP_CLAR(ioctlParam);
    ioctlParam[0] = SCREEN_0;
    ioctlParam[1] = 0;

    int ret = ioctl(m_iDisp, DISP_CMD_LAYER_DISABLE, (void*)ioctlParam);
    if(ret < 0)
    {
        printf("%s:ioctl disp disp off failed!\r\n", __func__);
        return;
    }

    m_iVideoStart = 0;
}

int LCDTask::VideoMap(int iSrcWidth, int iSrcHeight, unsigned int* piAddr)
{
    return LCDTask::VideoMap(iSrcWidth, iSrcHeight, *piAddr, *piAddr + iSrcWidth * iSrcHeight, *piAddr + iSrcWidth * iSrcHeight);
}

int LCDTask::VideoMap(int iSrcWidth, int iSrcHeight, unsigned int iAddr0, unsigned int iAddr1, unsigned int iAddr2)
{
    int iRet;

    if(m_iVideoStart == 0)
        return -1;

    g_xLayerInfo.fb.size.width  = iSrcWidth;
    g_xLayerInfo.fb.size.height = iSrcHeight;
    g_xLayerInfo.fb.src_win.x = 0;
    g_xLayerInfo.fb.src_win.y = 0;
    g_xLayerInfo.fb.src_win.width = iSrcWidth;
    g_xLayerInfo.fb.src_win.height = iSrcHeight;

    g_xLayerInfo.fb.addr[0] = iAddr0;
    g_xLayerInfo.fb.addr[1] = iAddr1;
    g_xLayerInfo.fb.addr[2] = iAddr2;
    iRet = LayerSetPara(m_iDisp, 0, &g_xLayerInfo);

    m_iVideoStart = 2;

    return iRet;
}

int LCDTask::DispOn()
{
    if(g_xSS.iPoweroffFlag != 1)
    {
#if (GPIO_CLASS_MODE == 0)
        GPIO_fast_config(LCD_BL_EN, LCD_BL_ON);
#else
        GPIO_class_setvalue(LCD_BL_EN, LCD_BL_ON);
#endif
    }
    return 0;
}


int LCDTask::DispOff()
{
#if (GPIO_CLASS_MODE == 0)
    GPIO_fast_config(LCD_BL_EN, LCD_BL_OFF);
#else
    GPIO_class_setvalue(LCD_BL_EN, LCD_BL_OFF);
#endif
    return 0;
}


void LCDTask::LCD_Update()
{
    if(m_iFB == -1 || m_piFB == NULL)
        return;

    memcpy(m_piFB, m_piFBMem, sizeof(m_piFBMem));
}

void LCDTask::LCD_Update(int left, int top, int right, int bottom)
{
    if(m_iFB == -1 || m_piFB == NULL)
        return;

    for(int y = top; y <= bottom; y ++)
        memcpy(m_piFB + y * MAX_X + left, m_piFBMem + y * MAX_X + left, (right - left + 1) * sizeof(uint32_t));
}

void LCDTask::LCD_MemClear(unsigned int Color)
{
    unsigned int i;

    for (i=0; i< MAX_X * MAX_Y; i++)
        m_piFBMem[i] = make_ARGB2ABGR(Color);
}


void LCDTask::LCD_SetImage(unsigned char* pbBayer, int iWidth, int iHeight)
{
    for(int i = 0; i < MAX_Y; i ++)
    {
        for(int j = 0; j < MAX_X; j ++)
        {
            int iR = pbBayer[i * iWidth + j];
            m_piFBMem[i * MAX_X + j] = (0xFF << 24) | (iR << 16) | (iR << 8) | (iR);
        }
    }
}

void LCDTask::DrawMemFocus(int x_pos, int y_pos, int x_size, int y_size, int color)
{
    int i, j;
    long location;

    x_size -= 1;
    y_size -= 1;

    for (i = 0; i < x_size / 6; i++)
    {
        for (j = 0; j < 4; j++)
        {
            //left top corner
            if((y_pos + j) < MAX_Y && (x_pos + i) < MAX_X && (x_pos + i) >= 0 && (y_pos + j) >= 0)
            {
                location = (y_pos + j) * MAX_X + (x_pos + i);
                m_piFBMem[location] = make_ARGB2ABGR(color);
            }

            if((y_pos + i) < MAX_Y && (x_pos + j) < MAX_X && (y_pos + i) >=0 && (x_pos + j) >=0)
            {
                location = (y_pos + i) * MAX_X + (x_pos + j);
                m_piFBMem[location] = make_ARGB2ABGR(color);
            }

            //rigth top corner
            if((y_pos + j) < MAX_Y && (x_pos + x_size - i) < MAX_X && (y_pos + j) >= 0 && (x_pos + x_size - i) >= 0)
            {
                location = (y_pos + j) * MAX_X + (x_pos + x_size - i);
                m_piFBMem[location] = make_ARGB2ABGR(color);
            }

            if((y_pos + i) < MAX_Y && (x_pos + x_size - j) < MAX_X && (y_pos + i) >= 0 && (x_pos + x_size - j) >= 0)
            {
                location = (y_pos + i) * MAX_X + (x_pos + x_size - j);
                m_piFBMem[location] = make_ARGB2ABGR(color);
            }

            //right bottom corner
            if((y_pos + y_size - j) < MAX_Y && (x_pos + x_size - i) < MAX_X && (y_pos + y_size - j) >= 0 && (x_pos + x_size - i) >= 0)
            {
                location = (y_pos + y_size - j) * MAX_X + (x_pos + x_size - i);
                m_piFBMem[location] = make_ARGB2ABGR(color);
            }

            if((y_pos + y_size - i) < MAX_Y && (x_pos + x_size - j) < MAX_X && (y_pos + y_size - i) >= 0 && (x_pos + x_size - j) >= 0)
            {
                location = (y_pos + y_size - i) * MAX_X + (x_pos + x_size - j);
                m_piFBMem[location] = make_ARGB2ABGR(color);
            }

            //left bottom corner
            if((y_pos + y_size - j) < MAX_Y && (x_pos + i) < MAX_X && (y_pos + y_size - j) >= 0 && (x_pos + i) >= 0)
            {
                location = (y_pos + y_size - j) * MAX_X + (x_pos + i);
                m_piFBMem[location] = make_ARGB2ABGR(color);
            }

            if((y_pos + y_size - i) < MAX_Y && (x_pos + j) < MAX_X && (y_pos + y_size - i) >= 0 && (x_pos + j) >= 0)
            {
                location = (y_pos + y_size - i) * MAX_X + (x_pos + j);
                m_piFBMem[location] = make_ARGB2ABGR(color);
            }
        }
    }
}

int LCDTask::DrawMemOverImage(int x0, int y0, int x1, int y1, const char* filePath, int iAlign)
{
    Image img;
    if(!img.load(filePath))
        return 0;
    int imgWidth = 0, imgHeight = 0;

    imgWidth = img.width();
    imgHeight = img.height();

    int rectWidth = x1 - x0 + 1;
    int rectHeight = y1 - y0 + 1;

    if(rectWidth >= imgWidth && (iAlign & ALIGN_HOR_CENTER))
        x0 = x0 + (rectWidth - imgWidth) / 2;

    if(rectHeight >= imgHeight && (iAlign & ALIGN_VER_CENTER))
        y0 = y0 + (rectHeight - imgHeight) / 2;

    for(int i = 0; i < imgHeight; i ++)
    {
        for(int j = 0; j < imgWidth; j ++)
        {
            unsigned int src = (m_piFBMem[MAX_X * (i + y0) + (j + x0)]);
            unsigned int dst = img.pixel(j, i);

//            printf("i=%d,j=%d,src=%x,dst=%x\n", i, j, src, dst);
//            dst = 0x80ff0000;
            int out_A = C_ALPHA(src) + C_ALPHA(dst) * (0x100 - C_ALPHA(src)) / 0x100;
            if(out_A == 0)
                m_piFBMem[MAX_X * (i + y0) + (j + x0)] = 0;
            else
            {
                if(C_ALPHA(dst) == 0xFF && C_ALPHA(src) == 0xFF)
                {
                    m_piFBMem[MAX_X * (i + y0) + (j + x0)] = make_ARGB2ABGR(dst);
                }
                else if(C_ALPHA(dst) == 0xFF)
                {
                    out_A = 0xFF;
                    unsigned int out_R = MIN(C_BLUE(src) * C_ALPHA(src) / 0xFF + C_RED(dst), 0xFF);
                    unsigned int out_G = MIN(C_GREEN(src) * C_ALPHA(src) / 0xFF + C_GREEN(dst), 0xFF);
                    unsigned int out_B = MIN(C_RED(src) * C_ALPHA(src) / 0xFF + C_BLUE(dst), 0xFF);

                    m_piFBMem[MAX_X * (i + y0) + (j + x0)] = MAKE_COLOR(out_A, out_B, out_G, out_R);
                }
                else if(C_ALPHA(src) == 0xFF && dst != 0)
                {
#if 0
                    out_A = 0xFF;
                    unsigned int out_R = MIN(255, C_RED(src) + C_RED(dst) * C_ALPHA(dst) / 0xFF);
                    unsigned int out_G = MIN(255, C_GREEN(src) + C_GREEN(dst) * C_ALPHA(dst) / 0xFF);
                    unsigned int out_B = MIN(255, C_BLUE(src) + C_BLUE(dst) * C_ALPHA(dst) / 0xFF);

                    printf("%x, %x---%x, %x, %x, %x\n",  src, dst, out_A, out_R, out_G, out_B);
                    m_piFBMem[MAX_X * (i + y0) + (j + x0)] = MAKE_COLOR(out_A, out_B, out_G, out_R);
#else
                    out_A = 0xFF;
                    unsigned int out_R = MIN(255, (C_RED(dst) * C_ALPHA(dst) + C_BLUE(src) * (0xFF - C_ALPHA(dst))) / 0xFF);
                    unsigned int out_G = MIN(255, (C_GREEN(dst) * C_ALPHA(dst) + C_GREEN(src) * (0xFF - C_ALPHA(dst))) / 0xFF);
                    unsigned int out_B = MIN(255, (C_BLUE(dst) * C_ALPHA(dst) + C_RED(src) * (0xFF - C_ALPHA(dst))) / 0xFF);

                    m_piFBMem[MAX_X * (i + y0) + (j + x0)] = MAKE_COLOR(out_A, out_B, out_G, out_R);
#endif
                }
                else
                {
                    unsigned int out_R = (C_BLUE(src) * C_ALPHA(src) / 0xFF + C_RED(dst) * C_ALPHA(dst) * (0xFF - C_ALPHA(src)) / (0xFF * 0xFF)) * 0xFF / out_A;
                    unsigned int out_G = (C_GREEN(src) * C_ALPHA(src) / 0xFF + C_GREEN(dst) * C_ALPHA(dst) * (0xFF - C_ALPHA(src)) / (0xFF * 0xFF)) * 0xFF / out_A;
                    unsigned int out_B = (C_RED(src) * C_ALPHA(src) / 0xFF + C_BLUE(dst) * C_ALPHA(dst) * (0xFF - C_ALPHA(src)) / (0xFF * 0xFF)) * 0xFF / out_A;

                    m_piFBMem[MAX_X * (i + y0) + (j + x0)] = MAKE_COLOR(out_A, out_B, out_G, out_R);
                }
            }
        }
    }

    return imgWidth;
}

void LCDTask::DrawMemStateIcon(const char* szPath, int iStep)
{
    int ZIGBEE_WIDTH = 48;
    int ZIGBEE_HEIGHT = 48;
    int STEP_WIDTH = 50;
    int ZIGBEE_POS_X = MAX_X - 60;
    int ZIGBEE_POS_Y = 1;

    DrawMemOverImage(ZIGBEE_POS_X - (iStep * STEP_WIDTH), ZIGBEE_POS_Y, ZIGBEE_POS_X - (iStep * STEP_WIDTH) + ZIGBEE_WIDTH - 1, ZIGBEE_POS_Y + ZIGBEE_HEIGHT - 1, szPath, ALIGN_CENTER);
}

unsigned int make_ARGB2ABGR(unsigned int color)
{
    unsigned int tmp;
    tmp = (color & 0xFF000000) | ((color & 0x00FF0000) >> 16) | (color & 0x0000FF00) | (((color & 0x000000FF) << 16));
    return tmp;
}
