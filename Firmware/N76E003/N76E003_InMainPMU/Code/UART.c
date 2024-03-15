// C File:
// File Name: UART.c
// Author:DaSha
// Date:2022/03/29
#include "N76E003.h"
#include "Common.h"
#include "Delay.h"
#include "SFR_Macro.h"
#include "Function_define.h"
#include "hw_Config.h"
#include "UART.h"
/******************************************************************************
 * FUNCTION_PURPOSE: Serial interrupt, echo received data.
 * FUNCTION_INPUTS : P0.7(RXD) serial input
 * FUNCTION_OUTPUTS: P0.6(TXD) serial output
 * Following setting in Common.c
 ******************************************************************************/

u8 len = 0;
extern u8 fSoundMute;
bit fReceivedHeader = 0;
u8 gUartPack[20];
UART_RX_PACKET bmotor_Cmd;
extern UART_RX_PACKET gRxCmd;
extern u8 receivedUartCmd;
bit fSendByteData = 0;
u8 str[10];

void itoa(u32 value, u8 fEnter)
{
	u8 index, i;
	u32 nIntValue;

	nIntValue = value;
	index = NUMBER_OF_DIGITS;
	i = 0;

	do {
		str[--index] = (u8)('0' + (nIntValue % 10));
		nIntValue /= 10;
	} while (nIntValue != 0);

	if (value < 0)
		str[i++] = '-';

	len = NUMBER_OF_DIGITS - index;

	do {
		str[i++] = str[index++];
	} while ( index < NUMBER_OF_DIGITS );

	if (fEnter)
	{
		str[i++] = 0x0d;
		str[i++] = 0x0a;
	}
	len += 2;
	str[i] = 0; /* string terminator */
}

 void UartSendByte(u8 dat)
 {
	u16 nUartDelay = 0;
	TI = 0;
	fSendByteData = 0;
	SBUF = dat;
	while ((!TI) && (fSendByteData == 0))
	{
		if (++nUartDelay > 1000) break;
	}
 }

void UartSendData(u8 direction, u32 val)
{
	u8 i = 0;
	u8 pStr[] = "-C";

	set_ES;
	itoa(val, 1);
	
	if (direction >= 1) UartSendByte(pStr[direction - 1]);

	for (i = 0; i < len; i++)
	{
		UartSendByte(str[i]);
	}
	clr_ES;
}

void UartSendString(char* string)
{
	u8 i = 0;
	u8 pStr[] = "-C";

	set_ES;
	
	for (i = 0; i < sizeof(string); i++)
	{
		UartSendByte(string[i]);
	}
	clr_ES;
}


u8 UartReadData(void)
{
	u16 nUartDelay = 0;
	u8 rData;
	while (!RI)
	{
		if (++nUartDelay > UART_TIMEOUT) return 0;
	}

	rData = SBUF;
	clr_RI;
	return rData;
}

void UART0_ISR(void) interrupt 4
{
	u8 rData;
	u8 i;
	u16 chkSum = 0;

	if (RI == 1)
	{	/* if reception occur */
		clr_RI;	/* clear reception flag for next reception */
		rData = SBUF;
		if (fReceivedHeader == 0)
		{
			if (rData == 0x7E)
			{
				gUartPack[0] = 0x7E;
				for	(i = 1; i < 4; i++)
				{
					gUartPack[i] = UartReadData();
				}
				if (gUartPack[1] == 0x55 && gUartPack[2] == 0xAA && gUartPack[3] == 0xAA)
				{
					clr_ES;
					clr_EA;
					fReceivedHeader = 1;

					for (i = 4; i < 16; i++)
					{
						gUartPack[i] = UartReadData();
					}
					bmotor_Cmd = *(UART_RX_PACKET *)(gUartPack);

					for (i = 4; i < bmotor_Cmd.motor_Info.bDataLen; i++)
					{
						if (i != 6)
							chkSum ^= bmotor_Cmd.dat[i];
					}
					if (bmotor_Cmd.motor_Info.bCheckSum == chkSum)
					{
						receivedUartCmd = gUartPack[4];
						switch (gUartPack[4])
						{
							case BELL_TOUCHED_CMD:
								MPU_POWER_ON(1);
								set_TR2;
								gRxCmd = *(UART_RX_PACKET *)(gUartPack);
								break;
							
							case KEY_HALT_TIME_CMD:
							case E_VD_RESULT:
							case E_PWN_RESULT:
							case E_BTN_RESULT:
								gRxCmd = *(UART_RX_PACKET *)(gUartPack);
								break;
								
							default:
								receivedUartCmd = 0;
								break;
						}
					}

					fReceivedHeader = 0;
					set_EA;
					set_ES;
				}
			}
		}
	}
	if (TI == 1)
	{
		clr_TI;                             /* if emission occur */
		fSendByteData = 1;
	}
 }


