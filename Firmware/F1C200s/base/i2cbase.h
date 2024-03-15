
#ifndef _I2C_BASE_H_
#define _I2C_BASE_H_

#include "appdef.h"
#include "mutex.h"


#define WORD_SIZE               (16)
#define WORD_SIZE_XX            (64)

#define I2C_ADDR_MAIN_STM       (0x70)

/////////////////////////Base ///////////////////////////

int I2C_Open(const char* szFilePath, int iAddr, int iMode);
int I2C_Close(int iFile);

int I2C_SetPointer8(int iFile, int iAddr);
int I2C_Read8(int iFile, int iAddr, unsigned char* pbData, int iLen);
int I2C_Write8(int iFile, int iAddr, unsigned char* pbData, int iLen);

int I2C_SetPointer16(int iFile, int iAddr);
int I2C_Read16(int iFile, int iAddr, unsigned char* pbData, int iLen);
int I2C_Write16(int iFile, int iAddr, unsigned char* pbData, int iLen);

int I2C_SetPointer32(int iFile, int iAddr);
int I2C_4Read16(int iFile, int iAddr, unsigned char* pbData, int iLen);
int I2C_4Write16(int iFile, int iAddr, unsigned char* pbData, int iLen);

/////////////////////////Main STM /////////////////////////

#define MAIN_STM_CMD_ROK                0x01        // [R]
#define MAIN_STM_CMD_POWER_DOWN         0x02        // [R]
#define MAIN_STM_GET_KEY_INFOS          0x04        // [R]
#define MAIN_STM_BELL_STATE             0x05        // [R]
#define MAIN_STM_VOICE_CALL_STATE       0x06        // [W]
#define MAIN_STM_DISABLE_KEY_FUNCS       0x07        // [W]
#define MAIN_STM_VERSION                0x10

////////////////////////GUOGU////////////////////////////

#define GUOGU_WAKE_UP_BY_STM            (1 << 0)
#define GUOGU_WAKE_UP_BY_FUNC           (1 << 2)

int     MainSTM_Open();
void    MainSTM_Close();
int     MainSTM_Command(int iCmd, unsigned char* pbData = 0, int* piData = 0);
int     MainSTM_WriteData(int iCmd, int iData);
int     MainSTM_GetKeyInfos(unsigned char* pbData, int iSize);
int     MainSTM_SetLcdBL(int iMode);
/////////////////////////////////////////////////////
extern Mutex g_xI2CMutex;
#endif //_M24C64_H_

