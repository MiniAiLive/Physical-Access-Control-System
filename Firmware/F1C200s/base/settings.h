#ifndef SETTINGS_H
#define SETTINGS_H

#include "appdef.h"
#include "shared.h"
#include <stdint.h>

enum
{
    LANG_START = -1,
    LANG_KP,
    LANG_CH,
    LANG_TW,
    LANG_EN,
    LANG_RU,
    LANG_PT,
    LANG_END
};

enum
{
    THEME_BLUE,
    THEME_GREEN,
    THEME_RED,
    THEME_END
};

//key
#define CUS_KEY_KO 0
#define CUS_KEY_CH 1
#define CUS_KEY_EN 2
#define CUS_KEY_NUM 3

#define VAL_ON  1
#define VAL_OFF 0

#define LAST_FACE_WID 120
#define LAST_FACE_HEI 150

#define DEFAULT_LANG LANG_CH

enum
{
    STATUS_NONE,
    STATUS_TIMEOUT,
    STATUS_GOTO_MAIN,
    STATUS_GOTO_SETTING_VIEW,
    STATUS_GOTO_LIST_VIEW,
    STATUS_GOTO_MSG_VIEW,
};

enum
{
    ERROR_NONE = 0,
    ERROR_CAMERA_CLR = 1,
    ERROR_CAMERA_IR = 2,
    ERROR_THREAD = 3,
    ERROR_LOOP = 4,
    ERROR_I2C = 5,
};

enum
{
    BTN_MENU = 1001,
    BTN_LEFT = 1002,
    BTN_RIGHT = 1003,
};

typedef struct _tagCOMMON_SETTINGS
{
    unsigned int    iShutdownTime;
    unsigned int    iLang;
    unsigned int    iTheme;
} COMMON_SETTINGS;

typedef struct _tagSYSTEM_STATE
{
    //System
    int             iAppType;
    int             iSystemState;
    int             iSelectedItem;
    int             iResetFlag;

    int             iSystemError;
    DATETIME_32     xCurTime;

    
    int             iTouchEvent;
    float           rTouchTime;
    
    int             iRunningCamSurface;
    int             iCamInited;
    int             iFirstCamInited;

    int             iPoweroffFlag;
    int             iUnlockFlag;
    int             iVoiceCallFlag;
    int             iBellingFlag;
} SYSTEM_STATE;

extern COMMON_SETTINGS g_xCS;
extern SYSTEM_STATE g_xSS;



void ResetSystemState();

unsigned char GetSettingsCheckSum(unsigned char* pbData, int iSize);
void ReadCommonSettings();
void UpdateCommonSettings();
void ResetCommonSettings();

unsigned int ReadLastRecordIdx();
void WriteLastRecordIdx(unsigned int iLastIdx);

void RestoreBackupSettings();
void UpdateBackupSettings();
void ResetCS(COMMON_SETTINGS* pxCS);

extern int g_iLangCount;
extern int g_aLangValue[];

extern int g_iKeyCount;
extern int g_aKeyValue[];

#endif // SETTINGS_H
