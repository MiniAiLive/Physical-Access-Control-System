// C File:
// File Name: I2C_Slave.c
// Author:DaSha
// Date:2022/03/02
//***********************************************************************************************************
//  N76E003-series I2C slave mode demo code, the Slave address = 0xA4
//
//   ____________            _____________ 
//  |            |   SDA    |             |
//  |            |<-------->|             |
//  |            |          |             |
//  |   RV1108(M)   |          |  N76E003(S) |
//  |(I2C_Master)|          | (I2C_Slave) |
//  |            |   SCL    |             |
//  |            |--------->|             |
//  |____________|          |_____________|
//
//***********************************************************************************************************
#include "string.h"
#include "N76E003.h"
#include "SFR_Macro.h"
#include "Function_define.h"
#include "Common.h"
#include "Delay.h"
#include "I2C_Slave.h"
#include "hw_Config.h"

u8 version[9] = "PM-1.03";

#define MY_SLAVE_ADDRESS        0xE0
#define MAX_BUFFER  			16

u8 u8_My_Buffer[MAX_BUFFER];
u8 *u8_MyBuffp = &u8_My_Buffer[0];
u8 receiveNum = 0;
u8 MessageBegin;
extern u8 receivedI2CCmd;
extern u8 prevPressedKeyType;
extern u8 pressedKeyState;
extern u8 nROK;
extern u8 powerDown;
extern u8 bellTouched;
extern u8 g_MPURunning;
extern u8 isAudioCall;
extern u8 isSetting;
extern u8 noMainAck;

void processA20Cmd(void);

void clearBuffer(void)
{
	u8 iPos = 0;
	do { 
		u8_My_Buffer[iPos] = 0;
		iPos ++;
	}while(iPos < MAX_BUFFER);
	u8_MyBuffp = &u8_My_Buffer[0];
}

void init_I2C_Slave(void)
{
	clr_I2CEN;
	clr_EA;
	P13_Quasi_Mode;                         //set SCL (P13) is Quasi mode
	P14_Quasi_Mode;                         //set SDA (P14) is Quasi mode
  
	SDA = 1;                                //set SDA and SCL pins high
	SCL = 1;

	set_P1SR_3;                             //set SCL (P13) is  Schmitt triggered input select.
	set_EI2C;                               //enable I2C interrupt by setting IE1 bit 0
	set_EA;

	I2ADDR = MY_SLAVE_ADDRESS;              //define own slave address
	set_I2CEN;                              //enable I2C circuit
	set_AA;
}

void I2C_byte_received(u8 u8_RxData)
{

	if (MessageBegin == 1) 
	{
		receivedI2CCmd = u8_RxData;
		
		//analizeV3SCommand(u8_RxData);
		MessageBegin = 0;
		clearBuffer();
	}
	else if(u8_MyBuffp < &u8_My_Buffer[MAX_BUFFER])
	{
		*(u8_MyBuffp++) = u8_RxData;
		receiveNum++;
	}
}


void I2C_ISR(void) interrupt 6
{
		g_MPURunning = 1;
	
    switch (I2STAT)
    {
        case 0x00:
            STO = 1;// 00H, bus error occurs,  recover from bus error
            break;

        case 0x60:	//Own SLA+W has been received ACK has been transmitted I2DAT = own SLA+W
			MessageBegin = 1;  
            AA = 1;
            break;
        case 0x68:	//Arbitration lost and own SLA+W has been received ACK has been transmitted I2DAT = own SLA+W
            AA = 1;
            break;

        case 0x80: //Data byte has been received ACK has been transmitted I2DAT = Data Byte
			I2C_byte_received(I2DAT);
			AA = 1;
            break;

        case 0x88: // Data byte has been received NACK has been transmitted I2DAT = Data Byte
            I2C_byte_received(I2DAT);
			AA = 1;
            break;

        case 0xA0: // A STOP or repeated START has been received AA = 1;
			AA = 1;
			if (receivedI2CCmd > 0) processA20Cmd();
            break;

        case 0xA8: // Own SLA+R has been received ACK has been transmitted I2DAT = own SLA+R 
			if (u8_MyBuffp < &u8_My_Buffer[MAX_BUFFER])
				I2DAT = *(u8_MyBuffp++);
			else
				I2DAT = 0x00;
            AA = 1;
            break;
        
        case 0xB8: // Data byte has been transmitted ACK has been received
			if (u8_MyBuffp < &u8_My_Buffer[MAX_BUFFER])
				I2DAT = *(u8_MyBuffp++);
			else
				I2DAT = 0x00;
            AA = 1;
            break;

        case 0xC0: // Data byte has been transmitted NACK has been received
            AA = 1;
            break; 

        case 0xC8: // Last Data byte has been transmitted ACK has been received
            AA = 1;
            break;        
    }
    SI = 0;
    while(STO);
}

void processA20Cmd(void)
{
	u8 i;
	
	switch (receivedI2CCmd)
	{
	case I2C_CMD_ROK:
		nROK = 0;
		break;

	case I2C_CMD_POWERDOWN:
		powerDown = 1;
		break;

	case I2C_CMD_VOICE_CALL_STATE:
		isAudioCall = u8_My_Buffer[0];
		break;
	
	case I2C_CMD_SETTING_STATE:
		isSetting = u8_My_Buffer[0];
		break;
	
	case I2C_CMD_GET_KEY_INFO:
		u8_My_Buffer[0] = receivedI2CCmd;
		u8_My_Buffer[1] = prevPressedKeyType;
		u8_My_Buffer[2] = pressedKeyState;
		u8_My_Buffer[3] = noMainAck;
		prevPressedKeyType = E_KEY_NONE;
		noMainAck = 0;
		break;
		
	case I2C_CMD_GET_BELL_STATE:
		u8_My_Buffer[0] = receivedI2CCmd;
		u8_My_Buffer[1] = bellTouched;
		bellTouched = 0;
		break;
	
	case I2C_CMD_GET_VERSION:
		u8_My_Buffer[0] = receivedI2CCmd;
		for (i = 0; i < sizeof(version); i++)
		{
			u8_My_Buffer[i + 1] = version[i];
		}
		break;

	default:
		break;
  }
}
