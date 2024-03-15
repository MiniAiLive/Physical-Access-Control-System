// Header:
// File Name: UART.c
// Author:DaSha
// Date:2022/03/03

#define UART_TIMEOUT		2000
#define NUMBER_OF_DIGITS				10

void itoa(unsigned long value, unsigned char fEnter);
void UartSendData(u8 direction, u32 val);
void UartSendByte(u8 dat);
void UartSendString(char* string);

typedef xdata union{
	struct
	{
		unsigned char		bHeader[4];    //0x7E55AAAA
		unsigned char		bCmd;    //
		unsigned char		bDataLen;     //16
		unsigned char		bCheckSum;

		unsigned char		bReserved2[9];
	} motor_Info;     //Size: 16

	unsigned char dat[16];
}UART_RX_PACKET;

typedef xdata union{
	struct
	{
		unsigned char		bHeader[4];    //0x7E55AAAA
		unsigned char		bCmd;    //
		unsigned char		bDataLen;     //16
		unsigned char		bCheckSum;

		unsigned char		Reserved[2];

		unsigned char		sVersion[7];
	} ack;     //Size: 16

	struct
	{
		unsigned char		bHeader[4];    //0x7E55AAAA
		unsigned char		bCmd;    //
		unsigned char		bDataLen;     //16
		unsigned char		bCheckSum;

		unsigned char		bWhat;

		unsigned char		Reserved1[8];
	} power_Result;     //Size: 16

	struct
	{
		unsigned char		bHeader[4];    //0x7E55AAAA
		unsigned char		bCmd;    //
		unsigned char		bDataLen;     //16
		unsigned char		bCheckSum;

		unsigned char		bState;

		unsigned char		Reserved1[8];
	} vd_Result;     //Size: 16
	
	struct
	{
		unsigned char		bHeader[4];    //0x7E55AAAA
		unsigned char		bCmd;    //
		unsigned char		bDataLen;     //16
		unsigned char		bCheckSum;

		unsigned char		bBtnNum;
		unsigned char		bBtnStatus;
		unsigned char		Reserved3[7];

	} btn_Result;     //Size: 16
	
	unsigned char dat[16];
}UART_TX_PACKET;

