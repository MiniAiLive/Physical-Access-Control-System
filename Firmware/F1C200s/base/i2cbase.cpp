#include "i2cbase.h"
#include "settings.h"
#include "mutex.h"
#include "shared.h"
//#include "drv_gpio.h"

#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <linux/i2c-dev.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>

#define CLR_I2C_TIME 800

static int g_iMainSTM = -1;
static Mutex g_xMainSTM_Mutex;

Mutex g_xI2CMutex;

int I2C_Open(const char* szFilePath, int iAddr, int iMode)
{
    int iFile;
    if ((iFile = open(szFilePath,iMode)) < 0)
    {
        printf("Failed to open the bus: Addr = %x, %s\n", iAddr, strerror(errno));
        return 0;
    }

    if (ioctl(iFile,I2C_SLAVE,iAddr) < 0)
    {
        printf("Failed to acquire bus access and/or talk to slave.\n");
        return 0;
    }
    return iFile;
}

/**
 * @brief I2C_Close
 * @param iFile
 * @return
 */
int I2C_Close(int iFile)
{
    close(iFile);
    return 0;
}

/**
 * @brief I2C_SetPointer8
 * @param iFile
 * @param iAddr
 * @return
 */
int I2C_SetPointer8(int iFile, int iAddr)
{
    char szBuf[2] = { 0 };
    if(iFile <= 0)
        return -1;

    szBuf[0] = iAddr;
    if (write(iFile, szBuf, 1) != 1)
    {
        return -1;
    }
    else
    {
    }

    return 0;
}

/**
 * @brief I2C_Read8
 * @param iFile
 * @param iAddr
 * @param pbData
 * @return
 */
int I2C_Read8(int iFile, int iAddr, unsigned char* pbData, int iLen)
{
    int iRet = -1;
    g_xI2CMutex.Lock();
    if(iLen > WORD_SIZE + 2)
    {
        g_xI2CMutex.Unlock();
        return iRet;
    }

    iRet = I2C_SetPointer8(iFile, iAddr);
    if(iRet < 0)
    {
        g_xI2CMutex.Unlock();
        return -1;
    }

    int a = read(iFile, pbData, iLen);
    if (a != iLen)
    {
    }
    else
    {
        printf("[I2C-1] Read 0x%0*x: ", 2, iAddr);
        for (int i=0; i< iLen; i++)
            printf("%0*x, ", 2, pbData[i]);
        printf("\n");

        g_xI2CMutex.Unlock();
        return 0;
    }
    g_xI2CMutex.Unlock();
    return -1;
}

/**
 * @brief I2C_Write8
 * @param iFile
 * @param iAddr
 * @param pbData
 * @return
 */
int I2C_Write8(int iFile, int iAddr, unsigned char* pbData, int iLen)
{
    unsigned char abData[WORD_SIZE * 2];
    g_xI2CMutex.Lock();
    if(iFile < 0)
    {
        g_xI2CMutex.Unlock();
        return -1;
    }

    abData[0] = iAddr;
    memcpy(abData + 1, pbData, iLen);

    if (write(iFile, abData, iLen + 1) != iLen + 1)
    {
        printf("[I2C-1] Error writing %i bytes\n", iLen);
    }
    else
    {
        printf("[I2C-1] Write 0x%0*x: ", 2, iAddr);
        for (int i = 0; i < iLen; i++)
        {
            printf("%0*x, ", 2, pbData[i]);
        }
        printf("\n\r");

        g_xI2CMutex.Unlock();
        return 0;
    }

    g_xI2CMutex.Unlock();
    return -1;
}

/**
 * @brief I2C_SetPointer16
 * @param iFile
 * @param iAddr
 * @return
 */
int I2C_SetPointer16(int iFile, int iAddr)
{
    char szBuf[4] = { 0 };
    if(iFile <= 0)
        return -1;

    szBuf[0] = (iAddr >> 8);
    szBuf[1] = (iAddr & 0xFF);
    if (write(iFile, szBuf, 2) != 2)
    {
        return -1;
    }
    else
    {
    }

    return 0;
}

int I2C_Read16(int iFile, int iAddr, unsigned char* pbData, int iLen)
{
    int iRet = -1;
    g_xI2CMutex.Lock();
    if(iFile < 0)
    {
        g_xI2CMutex.Unlock();
        return -1;
    }

    iRet = I2C_SetPointer16(iFile, iAddr);
    if(iRet < 0)
    {
        g_xI2CMutex.Unlock();
        return -1;
    }

    if (read(iFile, pbData, iLen) != iLen)
    {
        printf("[I2C-2] Error reading %i bytes\n", iLen);
    }
    else
    {
        LOG_PRINT("[I2C-2] Read 0x%0*x: ", 2, iAddr);
        for (int i=0; i< iLen; i++)
            LOG_PRINT("%0*x, ", 2, pbData[i]);
        LOG_PRINT("\n");

        g_xI2CMutex.Unlock();
        return 0;
    }

    g_xI2CMutex.Unlock();
    return -1;
}

int I2C_Write16(int iFile, int iAddr, unsigned char* pbData, int iLen)
{
    unsigned char abData[WORD_SIZE * 2];
    g_xI2CMutex.Lock();
    if(iFile < 0)
    {
        g_xI2CMutex.Unlock();
        return -1;
    }

    abData[0] = (iAddr >> 8) & 0xFF;
    abData[1] = (iAddr & 0xFF);
    memcpy(abData + 2, pbData, iLen);

    if (write(iFile, abData, iLen + 2) != iLen + 2)
    {
        printf("[I2C-2] Error writing %i bytes\n", iLen);
    }
    else
    {
        LOG_PRINT("[I2C-2] Write 0x%0*x: ", 2, iAddr);
        for (int i = 0; i < iLen; i++)
        {
            LOG_PRINT("%0*x, ", 2, pbData[i]);
        }
        LOG_PRINT("\n\r");
        usleep(10000);

        g_xI2CMutex.Unlock();
        return 0;
    }

    usleep(10000);
    g_xI2CMutex.Unlock();
    return -1;
}

int MainSTM_Open()
{
    if (g_iMainSTM > -1)
        return g_iMainSTM;

    g_iMainSTM = I2C_Open("/dev/i2c-0", I2C_ADDR_MAIN_STM, O_RDWR);
    printf("MainSTM: %d\n", g_iMainSTM);
    return g_iMainSTM;
}

void MainSTM_Close()
{
    if (g_iMainSTM >-1)
    {
        I2C_Close(g_iMainSTM);
        g_iMainSTM = -1;
    }
}

int MainSTM_Command(int iCmd, unsigned char* pbData, int* piData1)
{
    unsigned char abData[WORD_SIZE * 2] = { 0 };
    if(g_iMainSTM < 0)
        return 0;

    g_xMainSTM_Mutex.Lock();

    int iRet = I2C_Read8(g_iMainSTM, iCmd, abData, WORD_SIZE);
    if(iRet < 0)
    {
        LOG_PRINT("[Main STM] Card Command Failed: %d\n", iCmd);
        g_xMainSTM_Mutex.Unlock();
        return 0;
    }

    g_xMainSTM_Mutex.Unlock();

    if(iCmd == MAIN_STM_BELL_STATE)
    {
        if(abData[1] == 1)
            return 1;
        else
            return 0;
    }
    else if(iCmd == MAIN_STM_VERSION)
    {
        if(abData[0] != 0)
        {
            if(pbData != NULL)
                memcpy(pbData, abData + 1, 16);

            return 1;
        }
        else
            return 0;
    }
    else if(abData[0] == iCmd)
    {
        LOG_PRINT("[Main STM] Recv Card Command: %d\n", *(unsigned char*)abData);
        return 1;
    }
    else
        LOG_PRINT("[Main STM] Recv Card Command: %d, %d  failed\n", *(unsigned char*)abData, iCmd);

    return 0;
}

int MainSTM_WriteData(int iCmd, int iData)
{
    unsigned char abData[WORD_SIZE * 2] = { 0 };
    if(g_iMainSTM < 0)
        return 0;

    abData[0] = iData;
    g_xMainSTM_Mutex.Lock();
    int iRet = I2C_Write8(g_iMainSTM, iCmd, abData, 1);
    if(iRet < 0)
    {
        LOG_PRINT("[Main STM] Write Failed: %d\n", iCmd);
        g_xMainSTM_Mutex.Unlock();
        return 0;
    }

    g_xMainSTM_Mutex.Unlock();

    return 1;
}

int MainSTM_GetKeyInfos(unsigned char* pbData, int iSize)
{
    if(g_iMainSTM < 0)
        return 0;

    g_xMainSTM_Mutex.Lock();
    int iRet = I2C_Read8(g_iMainSTM, MAIN_STM_GET_KEY_INFOS, pbData, iSize);
    if(iRet < 0)
    {
        LOG_PRINT("[Main STM] Card Command Failed: %d\n", MAIN_STM_GET_KEY_INFOS);
        g_xMainSTM_Mutex.Unlock();
        return 0;
    }

    g_xMainSTM_Mutex.Unlock();

    if(pbData[0] != MAIN_STM_GET_KEY_INFOS)
    {
        printf("[Main STM] Read Cmd Failed!  %d\n", MAIN_STM_GET_KEY_INFOS);
        return 0;
    }

    return 1;
}
