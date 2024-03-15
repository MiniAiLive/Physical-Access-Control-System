#ifndef LCDTASK_H
#define LCDTASK_H

#include "my_draw.h"
#include "my_image.h"
#include "themedef.h"
#include "shared.h"
#include <sys/types.h>

#define LCD_BL_ON   1
#define LCD_BL_OFF  0

#define MAX_X   320
#define MAX_Y   240

#define ALIGN_LEFT 1
#define ALIGN_TOP 2
#define ALIGN_HOR_CENTER 4
#define ALIGN_VER_CENTER 8
#define ALIGN_CENTER (ALIGN_HOR_CENTER | ALIGN_VER_CENTER)

#define COLOR_RED 0xFFFF0000
#define COLOR_GREEN 0xFF00FF00

#define BG_BLACK_COLOR 0xFF020D1E
#define BG_BLACK_COLOR_BR 0xFF1B1205
#define BG_BLACK_COLOR_PI 0xFF200017

#define BG_DIALOG_COLOR 0xFF06214B
#define BG_DIALOG_COLOR_BR 0xFF442E0D
#define BG_DIALOG_COLOR_PI 0xFF50013A

#define BG_BTN_NORMAL_COLOR 0xFF1D3A69
#define BG_BTN_NORMAL_COLOR_BR 0xFF5D4929
#define BG_BTN_NORMAL_COLOR_PI 0xFF711557

#define BG_BTN_ACTIVE_COLOR 0xFF0C7BD6
#define BG_BTN_ACTIVE_COLOR_BR 0xFFC26920
#define BG_BTN_ACTIVE_COLOR_PI 0xFFE200C8

#define LINE_COLOR 0xFF5E84C5
#define LINE_COLOR_BR 0xFFB19A72
#define LINE_COLOR_PI 0xFFD251AC

#define BG_PUXIN_BLACK_COLOR 0xFF000000

#define BG_PUXIN_MSG_COLOR 0xFF1D3A69
#define BG_PUXIN_MSG_COLOR_BR 0xFF5D4929
#define BG_PUXIN_MSG_COLOR_PI 0xFF711557

#define BG_PUXIN_MSG_BTN_NORMAL_COLOR 0xFF1D3A69
#define BG_PUXIN_MSG_BTN_NORMAL_COLOR_BR 0xFF5D4929
#define BG_PUXIN_MSG_BTN_NORMAL_COLOR_PI 0xFF711557

#define BG_PUXIN_MSG_BTN_DOWN_COLOR 0xFFBEB4EC
#define BG_PUXIN_MSG_BTN_DOWN_COLOR_BR 0xFFDDE4BC
#define BG_PUXIN_MSG_BTN_DOWN_COLOR_PI 0xFFF2AEB9

#define BG_PUXIN_DIALOG_COLOR 0xFF06214B
#define BG_PUXIN_DIALOG_COLOR_BR 0xFF442E0D
#define BG_PUXIN_DIALOG_COLOR_PI 0xFF50013A

#define BG_PUXIN_BTN_LIGHT_NORMAL_COLOR 0xFF1D3A69
#define BG_PUXIN_BTN_LIGHT_NORMAL_COLOR_BR 0xFF5D4929
#define BG_PUXIN_BTN_LIGHT_NORMAL_COLOR_PI 0xFF711557

#define BG_PUXIN_BTN_LIGHT_DOWN_COLOR 0xFF96A4BA
#define BG_PUXIN_BTN_LIGHT_DOWN_COLOR_BR 0xFFAAA9A6
#define BG_PUXIN_BTN_LIGHT_DOWN_COLOR_PI 0xFFC48CB5

#define BG_PUXIN_BTN_LIGHT_NORMAL_COLOR1 0xFF2C57A0
#define BG_PUXIN_BTN_LIGHT_NORMAL_COLOR1_BR 0xFF8E703E
#define BG_PUXIN_BTN_LIGHT_NORMAL_COLOR1_PI 0xFFAC2083

#define BG_PUXIN_BTN_LIGHT_DOWN_COLOR1 0xFF9DB1D3
#define BG_PUXIN_BTN_LIGHT_DOWN_COLOR1_BR 0xFFC6BCAA
#define BG_PUXIN_BTN_LIGHT_DOWN_COLOR1_PI 0xFFDC94C6

#define BG_PUXIN_BTN_LIGHT_NORMAL_COLOR2 0xFFFF4081
#define BG_PUXIN_BTN_LIGHT_DOWN_COLOR2 0xFFFFA7C5

#define BG_PUXIN_PROGRESS_COLOR 0xFF5E84C5
#define BG_PUXIN_PROGRESS_COLOR_BR 0xFFB19A72
#define BG_PUXIN_PROGRESS_COLOR_PI 0xFFD251AC


#define BG_PUXIN_BTN_DARK_NORMAL_COLOR 0xFF3C3C3C

#define BG_PUXIN_BTN_ACTIVE_COLOR 0xFF04132C
#define BG_PUXIN_BTN_ACTIVE_COLOR_BR 0xFF281C08
#define BG_PUXIN_BTN_ACTIVE_COLOR_PI 0xFF2F0122

#define PUXIN_LINE_COLOR 0xFF33B5E5
#define PUXIN_BORDER_COLOR 0xFF3C3C3C

#define PUXIN_EDIT_COLOR 0xFFC0C0C0

#define XINGLIAN_PROGRESSBAR_COLOR 0xFF00FF4E

#define C_White          0xFFFFFFFF
#define C_Black          0xFF000000
#define C_Grey           0x00808080
#define C_Red            0xFFFF0000
#define C_Green          0x0000FF00
#define C_Blue           0x000000FF
#define C_Blue2          0x000014FF
#define C_Magenta        0x00FF00FF
#define C_Cyan           0x0000FFFF
#define C_Yellow         0xFFFFFF00
#define C_NoBack		   0x00000000

#define C_ALPHA(a) ((a >> 24) & 0xFF)
#define C_RED(a) ((a >> 16) & 0xFF)
#define C_GREEN(a) ((a >> 8) & 0xFF)
#define C_BLUE(a) ((a) & 0xFF)
#define MAKE_COLOR(a, r, g, b) ((a & 0xFF) << 24 | (r & 0xFF) << 16 | (g & 0xFF) << 8 | (b & 0xFF))

#define BTN_NO 0
#define BTN_YES 1
#define LCD_INTENT_X 0

#define LCD_HEADER_FONT_SIZE 25
#define LCD_FOOTER_FONT_SIZE 20
#define LCD_FOOTER_FONT_SIZE_1 40
#define LCD_TIME_FONT_SIZE 60
#define LCD_TITLE_FONT_SIZE 24
#define LCD_NORMAL_FONT_SIZE 16
#define LCD_ALERT_FONT_SIZE 18
#define LCD_MSGBOX_TITLE_FONT_SIZE 19
#define LCD_NUM_BTN_FONT_SIZE 21
#define LCD_ICON_FONT_SIZE 12
#define LCD_ITEM_FONT_SIZE 16

#define LCD_HEADER_FONT_COLOR White //white
#define LCD_TITLE_FONT_COLOR Green //green

#define LCD_HEADER_COLOR    0x28000000

#define BTN_SETTING_BASE_X1 (20)
#define BTN_SETTING_BASE_Y1 (40)
#define BTN_SETTING_BASE_WIDTH (80)
#define BTN_SETTING_BASE_HEIGHT (85)
#define BTN_SETTING_BASE_GAP_X (20)
#define BTN_SETTING_BASE_GAP_Y (15)

#define ITEM_BASE_X1 (13)
#define ITEM_TITLE_HEIGHT (50)
#define ITEM_BASE_WIDTH (200)
#define ITEM_BASE_HEIGHT (40)
#define ITEM_BASE_GAP_X (6)
#define ITEM_BASE_GAP_Y (3)
#define ITEM_BASE_Y1(x) ((MAX_Y - ITEM_TITLE_HEIGHT - ITEM_BASE_HEIGHT * x) / 2)

#define ITEM_TIME_WIDTH (40)
#define ITEM_TIME_HEIGHT (35)
#define ITEM_TIME_GAP_X (20)
#define ITEM_TIME_GAP_Y (10)
#define ITEM_TIME_X1 ((MAX_X - 3 * ITEM_TIME_WIDTH - 2 * ITEM_TIME_GAP_X) / 2)

#define TIME_PROGRESSBAR_WIDTH 180
#define TIME_PROGRESSBAR_HEIGHT 50
#define PROGRESSBAR_HEIGHT 5

#define IC_STATE_WIDTH  30
#define IC_STATE_HEGITH 30
#define RIGHT_MARGIN    10

typedef struct _tagBUTTON
{
    int     iID;
    int     iX1;
    int     iY1;
    int     iX2;
    int     iY2;
    int     iState;

    char    szTxt[64];
    int     iFontSize;

    char    szNormalImg[64];
    char    szPressImg[64];

    unsigned int    iNormalColor;
    unsigned int    iPressColor;
} BUTTON;

enum
{
    BTN_ID_NONE = 0,
    BTN_ID_ANY = 1,
    BTN_ID_NO = 3,
    BTN_ID_YES = 4,
    BTN_ID_MENU_BASE = 50,
    BTN_ID_SETTING_BASE = 60,
    BTN_ID_DETAIL_BASE = 60,
};

#define MAX_BUTTON_CNT 50

#define BTN_STATE_NONE 0
#define BTN_STATE_PRESSED 1

enum
{
    E_SCENE_MAIN,

    E_SCENE_SETTING_VIEW,
    E_SCENE_LIST_VIEW,
    E_SCENE_MSG_VIEW,
    E_SCENE_MSG_RESET,
    E_SCENE_END
};

enum E_SETTING_BTNS
{
    E_SETTING_AUTO_PWDN = 0,
    E_SETTING_LANG,
    E_SETTING_RESET,
    E_SETTING_DATE_TIME,
    E_SETTING_VERSION,
    E_SETTING_END,
};

class LCDTask
{
public:
    LCDTask();
    virtual ~LCDTask();

    void    Init(int iSceneMode, int iData1, int iData2);
    void    Update();
    void    UpdateStateIcon(int iUnlockFlag, int iVoiceCallFlag);
    void    ResetMainButtons();
    void    DrawClock();
    int     KeyEvent(int iKey, int iEvent);

    int     LCD_DrawText(int x0, int y0, int x1, int y1, const char* str_text, int iAlign, int iFontSize, unsigned int iColor, unsigned int iBackColor);

    static int  DispOn();
    static int  DispOff();
    static int  DispOpen();
    static void DispClose();
    static int  VideoStart(int iFormat, int iWinX = 0, int iWinY = 0, int iWinW = 320, int iWinH = 240);
    static int  VideoMap(int iSrcWidth, int iSrcHeight, unsigned int* piAddr);
    static int  VideoMap(int iSrcWidth, int iSrcHeight, unsigned int iAddr0, unsigned int iAddr1, unsigned int iAddr2);
    static void VideoStop();
    static int  FB_Init();
    static void FB_Release();

    static void LCD_Update();
    static void LCD_Update(int left, int top, int right, int bottom);
    static void LCD_MemClear(unsigned int Color);

    static void LCD_SetImage(unsigned char* pbBayer, int iWidth, int iHeight);

    static void DrawMemFullImg(unsigned char* pbImg32);
    static void DrawMemImageOverlay(int iX0, int iY0, int iX1, int iY1, int iOverColor, const char* szFilePath);
    static int  DrawMemOverImage(int x0, int y0, int x1, int y1, const char* filePath, int iAlign);
    static void DrawMemFace(int x_pos, int y_pos, int x_size, int y_size, int color);
    static void DrawMemLine( unsigned int x0, unsigned int y0, unsigned int x1, unsigned int y1 , unsigned int color );
    static void DrawMemRectangle(int x0, int y0, int x1, int y1 , unsigned int color);
    static void DrawMemFillRect(int x0, int y0, int x1, int y1, int color);
    static void DrawMemFocus(int x_pos, int y_pos, int x_size, int y_size, int color);
    static void DrawMemSceneOverLayAlpha(int iApha);
    static void DrawMemStateIcon(const char* szPath, int iStep);
    static void LCD_DrawProgress(int value, int maximum);

protected:
    void    DrawStateIcon(int iUnlockFlag, int iVoiceCallFlag);
    void    DrawDateTime(int iLang, int iHourFormat, int iDateFormat);
    void    DrawMain();

    void    DrawSettingScene();
    void    DrawListViewScene(const char* szTitle, int iItemCount);
    void    DrawSetTimeViewScene();
    void    DrawButtons(int iID = -1);

    void    InitMsg(const char* szMsg, int iLang, int iTheme);
    void    DrawMsg(const char* szTitle, const char* szMsg, int iHasBtn, int iTheme);

    void    AddButton(int iID, int iX1, int iY1, int iX2, int iY2, const char* szTxt, int iFontSize, unsigned int iNormalColor, int iPressColor, const char* szNormalImg, const char* szPressImg, signed char bSelected = 0);
    void    ResetButtons();
    void    AddMenuButtons(int iTheme);
    void    AddSettingButtons(int iTheme);
    void    AddItems(int iTheme);

    void    GetImgInfo(const char* filePath, int* pnImgWidth, int* pnImgHeight);
    void    GetMsgPosInfo(const char* szMsgPath, int iHasBtn, int& iMsgX, int& iMsgY, int& iMsgWidth, int& iMsgHeight, int& iContentHeight, int iFontSize);

public:
    static int      m_iDispOn;
    static int      m_iDisp;

    static int      m_iFB;
    static long     m_iScreenSize;

    static int      m_iVideoStart;

    static unsigned int*    m_piFB;
    static unsigned int     m_piFBMem[MAX_X * MAX_Y];

private:
    int     m_iSceneMode;
    int     m_iData1;
    int     m_iData2;

    int             m_iBtnCnt;
    BUTTON          m_axBtns[MAX_BUTTON_CNT];

    float           m_rResetTime;
    int             m_iHiddenCodeFlag;
    DATETIME_32     m_xSetTime;

    Canvas* m_canvas;
};

#endif // WATCHTASK_H
