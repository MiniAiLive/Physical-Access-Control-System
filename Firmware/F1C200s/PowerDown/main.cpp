
#include "shared.h"
#include "i2cbase.h"
#include "settings.h"
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

COMMON_SETTINGS g_xCS = { 0 };
SYSTEM_STATE g_xSS = { 0 };

float g_clrCamTime = 0;

float Now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1000.f + ts.tv_nsec/1000000.f;
}

void ResetCommonSettings()
{

}

void UpdateCommonSettings()
{
}

void LOG_PRINT(const char * format, ...)
{
}

unsigned char GetSettingsCheckSum(unsigned char* pbData, int iSize)
{
    int iCheckSum = 0;
    for(int i = 0; i < iSize - 1; i ++)
        iCheckSum += pbData[i];

    iCheckSum = 0xFF - (iCheckSum & 0xFF);
    return (unsigned char)iCheckSum;
}

int main(int argc, char *argv[])
{
    MainSTM_Open();
    printf("send power down\n");

    MainSTM_Command(MAIN_STM_CMD_POWER_DOWN);
    MainSTM_Close();
    return 0;
}
