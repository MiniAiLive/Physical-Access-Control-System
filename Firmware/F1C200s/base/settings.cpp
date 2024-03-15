#include "settings.h"
#include "appdef.h"
//#include "i2cbase.h"
//#include "lcdtask.h"
#include "shared.h"

#include <unistd.h>
#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

SYSTEM_STATE            g_xSS = { 0 };
COMMON_SETTINGS         g_xCS = { 0 };

int g_aLangValue[] =
{
    LANG_CH,
    LANG_TW,
    LANG_EN
};

int g_iLangCount = sizeof(g_aLangValue) / sizeof(int);

int g_aKeyValue[] = {
    CUS_KEY_CH,
    CUS_KEY_EN,
    CUS_KEY_NUM
};

int g_iKeyCount = sizeof(g_aKeyValue) / sizeof(int);

void ResetSystemState()
{
    memset(&g_xSS, 0, sizeof(SYSTEM_STATE));
}

/**

 * @param pbData
 * @return
 */
unsigned char GetSettingsCheckSum(unsigned char* pbData, int iSize)
{
    int iCheckSum = 0;
    for(int i = 0; i < iSize - 1; i ++)
        iCheckSum += pbData[i];

    iCheckSum = 0xFF - (iCheckSum & 0xFF);
    return (unsigned char)iCheckSum;
}


/**

 */
void ReadCommonSettings()
{
    memset(&g_xCS, 0, sizeof(g_xCS));
    FILE* fp = fopen(MNT_PATH"/common_setting.off", "rb");
    if(fp)
    {
        unsigned char bCheckSum = 0;

        fread(&g_xCS, sizeof(g_xCS), 1, fp);
        fread(&bCheckSum, sizeof(bCheckSum), 1, fp);
        fclose(fp);

        unsigned char bCurCheckSum = 0;
        unsigned char* pbData = (unsigned char*)&g_xCS;
        for(int i = 0; i < sizeof(g_xCS); i ++)
            bCurCheckSum ^= pbData[i];

        if(bCurCheckSum != bCheckSum)
        {
            printf("Common Setting: CheckSum Error!\n");
            remove(MNT_PATH"/common_setting.off");
            ReadCommonSettings();
        }
    }
    else
    {
        ResetCommonSettings();
    }
}


/**

 */
void UpdateCommonSettings()
{
    FILE* fp = fopen(MNT_PATH"/common_setting.off", "wb");
    if(fp)
    {
        unsigned char bCurCheckSum = 0;
        unsigned char* pbData = (unsigned char*)&g_xCS;
        for(int i = 0; i < sizeof(g_xCS); i ++)
            bCurCheckSum ^= pbData[i];

        fwrite(&g_xCS, sizeof(g_xCS), 1, fp);
        fwrite(&bCurCheckSum, sizeof(bCurCheckSum), 1, fp);
        fflush(fp);
        fclose(fp);
    }
}

/**

 */
void ResetCommonSettings()
{
    ResetCS(&g_xCS);
    UpdateCommonSettings();
}

void ResetCS(COMMON_SETTINGS* pxCS)
{
    memset(pxCS, 0, sizeof(COMMON_SETTINGS));
    g_xCS.iShutdownTime = DEFAULT_SHUTDOWN_TIME;
    g_xCS.iLang = DEFAULT_LANGUAGE;
}

unsigned int ReadLastRecordIdx()
{
    unsigned int iIdx = 0;
    FILE* fp = fopen(MNT_PATH"/last_record_idx.off", "rb");
    if(fp)
    {
        fread(&iIdx, sizeof(iIdx), 1, fp);
        fclose(fp);
    }
    else
    {
        WriteLastRecordIdx(iIdx);
        return iIdx;
    }

    return iIdx % 1000000;
}

void WriteLastRecordIdx(unsigned int iLastIdx)
{
    unsigned int iIdx = iLastIdx % 1000000;
    FILE* fp = fopen(MNT_PATH"/last_record_idx.off", "wb");
    if(fp)
    {
        fwrite(&iIdx, sizeof(iIdx), 1, fp);
        fflush(fp);
        fclose(fp);
    }
}
