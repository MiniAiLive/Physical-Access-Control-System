#include "N76E003.h"
#include "Common.h"
#include "Delay.h"
#include "SFR_Macro.h"
#include "Function_define.h"
#include "I2C_Slave.h"
#include "hw_Config.h"
#include "UART.h"


UART_RX_PACKET gRxCmd;
UART_TX_PACKET gTxCmd;
u8 bellTouched = 0;
u8 receivedUartCmd;

void initTimer2(void);

u8 pressedKeyType = 0;
u8 prevPressedKeyType = 0;
u8 pressedKeyState = 0;
u8 nROK = 0;
u8 powerDown = 0;
u8 g_MPURunning = 0;
u8 receivedUartCmd = 0;
u8 receivedI2CCmd = 0;
u8 isTimeout = 0;
u8 isAudioCall = 0;
u8 isSetting = 0;
u8 noMainAck = 0;

void GPIO_Config(void)
{
	Enable_BIT0_FallEdge_Trig;
	P10_PushPull_Mode; // Voice IN
	P10 = 1;
	
	Enable_BIT5_FallEdge_Trig;
	P15_PushPull_Mode; // Lock IN
	P15 = 1;
	
	Enable_BIT7_FallEdge_Trig;
	P17_PushPull_Mode; // Power IN
	P17 = 1;
	
	P04_PushPull_Mode; // POWER_EN
	P04 = 0;
	
	P05_PushPull_Mode; // MCU_INT
	P05 = 0;
	
	Enable_INT_Port1;
}

void UART_Config(void)
{
	InitialUART0_Timer1(9600);

	//Enable_BIT7_FallEdge_Trig;
}

void initTimer2(void)
{
	TIMER2_DIV_512;
	TIMER2_Auto_Reload_Delay_Mode;

	RCMP2L = TIMER_DIV512_VALUE_1s;
	RCMP2H = TIMER_DIV512_VALUE_1s >> 8;
	TL2 = 0;
	TH2 = 0;
	set_ET2;                                    // Enable Timer2 interrupt
}

void PinInterrupt_ISR (void) interrupt 7
{ 
	PIF = 0x00;
}

void Timer2_ISR (void) interrupt 5
{
	clr_TF2; 
	if (++nROK > 7)
	{
		isTimeout = 1;
	}
}

void sendInt(void)
{
	MCU_INT(1);
	Timer0_Delay1ms(30);
	MCU_INT(0);
}

u8 isClickedKey(UINT8 pin)
{
	u8 cntDelay = 100;

	do
	{
		if (pin) return 0;

		if (cntDelay == 100)
		{
			Timer0_Delay1ms(1);
			cntDelay -= 9;
		}
		else
			Timer0_Delay100us(1);
	} while (--cntDelay > 0);
	
	return 1;
}

unsigned char getChecksum(u8* buf)
{
	unsigned char i;
	unsigned char ret = 0;

	for (i = 4; i < 16; i++)
	{
		if (i != 6)
			ret ^= buf[i];
	}

	return ret;
}

void sendAck(u8 cmd, u8 isFormat)
{
	u8 i;

	gTxCmd.dat[0] = 0x7E;
	gTxCmd.dat[1] = 0x55;
	gTxCmd.dat[2] = 0xAA;
	gTxCmd.dat[3] = 0xAA;

	gTxCmd.dat[5] = 16;

	switch (cmd)
	{
		case E_CMD_ACK:
			gTxCmd.ack.bCmd = receivedUartCmd;
			gTxCmd.ack.bCheckSum = getChecksum(gTxCmd.dat);
			break;
		
		case E_VD_RESULT:
		case E_PWN_RESULT:
		case E_BTN_RESULT:
			gTxCmd.ack.bCmd = cmd;
			gTxCmd.ack.bCheckSum = getChecksum(gTxCmd.dat);
			break;

		default:
			break;
	}

	for (i = 0; i < 10; i++)
		UartSendByte(0);

	for (i = 0; i < 16; i++)
	{
		UartSendByte(gTxCmd.dat[i]);
		if (isFormat)
			gTxCmd.dat[i] = 0;
	}
}

unsigned char receiveAck(u8 cmd, u16 delay, u8 reply)
{
	u8 cnt = 0, i = 0;

	while (--reply > 0){
		set_WDCLR;
		cnt = 0;
		sendAck(cmd, 0);
		while (++cnt < delay)
		{
			set_WDCLR;
			Timer0_Delay1ms(1);
			if (receivedUartCmd == cmd)
			{
				for (i = 0; i < 16; i++)
					gTxCmd.dat[i] = 0;
				return 0;
			}
		}
	}
	for (i = 0; i < 16; i++)
		gTxCmd.dat[i] = 0;

	return -1;
}

void processKeys(void)
{
	u8 tmp = 0;
	int ret = 0;
	int cnt = 0;
	u8 isSetting1 = isSetting;
	
	set_WDCLR;
	if (isClickedKey(LOCK_IN))
	{
		pressedKeyType = E_KEY_LOCK;
		pressedKeyState = E_KEY_PRESS_DOWN;
		prevPressedKeyType = pressedKeyType;
		sendInt();

		while (++cnt < 200)
		{
			set_WDCLR;
			Timer0_Delay1ms(100);
			
			if (receivedUartCmd == KEY_HALT_TIME_CMD)
			{
				sendAck(E_CMD_ACK, 1);
				receivedUartCmd = 0;
			}
			
			if (isClickedKey(LOCK_IN) == 0)
				break;
		}
		
		prevPressedKeyType = pressedKeyType;
		pressedKeyState = E_KEY_PRESS_UP;
		sendInt();
		if (!isSetting1)
		{
			gTxCmd.btn_Result.bBtnNum = pressedKeyType;
			gTxCmd.btn_Result.bBtnStatus = pressedKeyState;
			ret = receiveAck(E_BTN_RESULT, 200, 4);
		}
		
		pressedKeyType = E_KEY_NONE;
	}
	else if (isClickedKey(VOICE_IN))
	{
		if (!g_MPURunning)
			return;
		
		pressedKeyType = E_KEY_VOICE;
		pressedKeyState = E_KEY_PRESS_DOWN;
		prevPressedKeyType = pressedKeyType;
		sendInt();
		
		while (++cnt < 200)
		{
			set_WDCLR;
			Timer0_Delay1ms(100);
			
			if (receivedUartCmd == KEY_HALT_TIME_CMD)
			{
				sendAck(E_CMD_ACK, 1);
				receivedUartCmd = 0;
			}
			
			if (isClickedKey(VOICE_IN) == 0)
				break;
		}
		
		prevPressedKeyType = pressedKeyType;
		pressedKeyState = E_KEY_PRESS_UP;
		sendInt();
		pressedKeyType = E_KEY_NONE;
	}
	else if (isClickedKey(POWER_IN))
	{
		set_TR2;
		pressedKeyType = E_KEY_POWER;
		pressedKeyState = E_KEY_PRESS_DOWN;
		prevPressedKeyType = pressedKeyType;
		
		if (!isSetting1)
		{
			gTxCmd.btn_Result.bBtnNum = pressedKeyType;
			gTxCmd.btn_Result.bBtnStatus = pressedKeyState;
			ret = receiveAck(E_BTN_RESULT, 200, 4);
			if (ret)
				noMainAck = 1;
		}
		
		sendInt();
		if(!noMainAck)
			MPU_POWER_ON(1);

		while (++cnt < 200)
		{
			set_WDCLR;
			Timer0_Delay1ms(100);
			
			if (receivedUartCmd == KEY_HALT_TIME_CMD)
			{
				sendAck(E_CMD_ACK, 1);
				receivedUartCmd = 0;
			}
			
			if (isClickedKey(POWER_IN) == 0)
				break;
		}
		
		prevPressedKeyType = pressedKeyType;
		pressedKeyState = E_KEY_PRESS_UP;
		sendInt();
		
		pressedKeyType = E_KEY_NONE;
	}
	
	return;
}

void Enable_WDT_Reset_Config(void)
{
	set_IAPEN;
	IAPAL = 0x04;
	IAPAH = 0x00;
	IAPFD = 0x0F;
	IAPCN = 0xE1;
	set_CFUEN;
	set_IAPGO;                                  	//trigger IAP

	while ((CHPCON & SET_BIT6) == SET_BIT6);         //check IAPFF (CHPCON.6)
	clr_CFUEN;
	clr_IAPEN;

	TA = 0xAA;
	TA = 0x55;
	WDCON = 0x07;  						//Setting WDT prescale
	set_WDCLR;							//Clear WDT timer
	while ((WDCON | ~SET_BIT6) == 0xFF);
}

void main(void)
{
	int i = 0, flag = 0, ret;
	GPIO_Config();
	UART_Config();
	set_ES;
	init_I2C_Slave();
	set_EPI;							// Enable pin interrupt
	initTimer2();
	set_EA;								// global enable bit
	clr_BODEN;

	Enable_WDT_Reset_Config();
	
	while(1)
	{
		set_WDCLR;
		processKeys();
		Timer0_Delay1ms(10);
		
		if (powerDown)
		{
			for (i = 0; i < 100; i++)
			{
				set_WDCLR;
				Timer0_Delay1ms(10);
			}
			MPU_POWER_ON(0);
			g_MPURunning = 0;
			clr_TR2;
			nROK = 0;
			powerDown = 0;
			
			gTxCmd.power_Result.bWhat = BY_CMD;
			ret = receiveAck(E_PWN_RESULT, 200, 4);
		}
		
		if (isTimeout)
		{
			MPU_POWER_ON(0);
			clr_TR2;
			g_MPURunning = 0;
			nROK = 0;
			isTimeout = 0;
			
			gTxCmd.power_Result.bWhat = BY_TIMEOUT;
			ret = receiveAck(E_PWN_RESULT, 200, 4);
		}
		
		if (receivedI2CCmd == I2C_CMD_VOICE_CALL_STATE)
		{
			gTxCmd.vd_Result.bState = isAudioCall;
			ret = receiveAck(E_VD_RESULT, 200, 4);
			receivedI2CCmd = 0;
		}
		
		if (receivedUartCmd == BELL_TOUCHED_CMD)
		{
			bellTouched = 1;
			sendAck(E_CMD_ACK, 1);
			receivedUartCmd = 0;
			sendInt();
		}
		else if (receivedUartCmd == KEY_HALT_TIME_CMD)
		{
			sendAck(E_CMD_ACK, 1);
			receivedUartCmd = 0;
		}
#if 0		
		if (g_MPURunning == 0)
		{
			set_EPI;
			set_PD;// Enter in Lowest Power Mode
			clr_EPI;
			set_ES;
		}
#endif
	}
}
