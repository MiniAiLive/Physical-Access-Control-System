// Header:
// File Name: hw_Config.h
// Author:DaSha
// Date:2022/3/2

#define MCU_INT(x)				(x ? (set_P05) : (clr_P05))
#define LOCK_IN					P15
#define VOICE_IN				P10
#define POWER_IN				P17

#define MPU_POWER_ON(x)				(x ? (set_P04) : (clr_P04))

#define BELL_TOUCHED_CMD		1
#define KEY_HALT_TIME_CMD		4

enum BACK_RESPONSE {
	E_CMD_ACK = 			0x7E,
	E_VD_RESULT	=		0x81,
	E_PWN_RESULT = 			0x82,
	E_BTN_RESULT = 			0x83
};

enum I2C_CMD {
	I2C_CMD_ROK = 			0x01,
	I2C_CMD_POWERDOWN = 		0x02,
	I2C_CMD_GET_KEY_INFO =	0x04,
	I2C_CMD_GET_BELL_STATE =	0x05,
	I2C_CMD_VOICE_CALL_STATE =	0x06,
	I2C_CMD_SETTING_STATE =	0x07,
	I2C_CMD_GET_VERSION = 	0x10,
};

enum KEY_INFO {
	E_KEY_NONE		=		0,
	E_KEY_LOCK		=		1,
	E_KEY_VOICE		=		2,
	E_KEY_POWER		=		3,
};

enum KEY_STATE {
	E_KEY_PRESS_UP =		0x00,
	E_KEY_PRESS_DOWN =		0x01,
};

enum POWER_STATE {
	BY_TIMEOUT =		0x00,
	BY_CMD =		0x01,
};

