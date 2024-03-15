#ifndef _DRV_GPIO_H
#define _DRV_GPIO_H

#include "appdef.h"

#ifdef __cplusplus
extern	"C"
{
#endif

#define LCD_BL_EN       132 /* PE04 */
#define STM_INT         136 /* PE08 */
#define SPK_EN          133 /* PE05 */

#define IN  0
#define OUT 1

#define OFF     0
#define ON      1
#define FLICK   2

//----------------------------------//
//       PORT BASE DEFINITIONS      //
//----------------------------------//

#define SUNXI_PORT_A_BASE      (0*0x24)
#define SUNXI_PORT_B_BASE      (1*0x24)
#define SUNXI_PORT_C_BASE      (2*0x24)
#define SUNXI_PORT_D_BASE      (3*0x24)
#define SUNXI_PORT_E_BASE      (4*0x24)
#define SUNXI_PORT_F_BASE      (5*0x24)
#define SUNXI_PORT_G_BASE      (6*0x24)
#define SUNXI_PORT_H_BASE      (7*0x24)
#define SUNXI_PORT_I_BASE      (8*0x24)


//----------------------------------//
//         PIO DEFINITIONS          //
//----------------------------------//

#define SUNXI_PIO_00           (0x00000001L <<  0)
#define SUNXI_PIO_01           (0x00000001L <<  1)
#define SUNXI_PIO_02           (0x00000001L <<  2)
#define SUNXI_PIO_03           (0x00000001L <<  3)
#define SUNXI_PIO_04           (0x00000001L <<  4)
#define SUNXI_PIO_05           (0x00000001L <<  5)
#define SUNXI_PIO_06           (0x00000001L <<  6)
#define SUNXI_PIO_07           (0x00000001L <<  7)
#define SUNXI_PIO_08           (0x00000001L <<  8)
#define SUNXI_PIO_09           (0x00000001L <<  9)
#define SUNXI_PIO_10           (0x00000001L <<  10)
#define SUNXI_PIO_11           (0x00000001L <<  11)
#define SUNXI_PIO_12           (0x00000001L <<  12)
#define SUNXI_PIO_13           (0x00000001L <<  13)
#define SUNXI_PIO_14           (0x00000001L <<  14)
#define SUNXI_PIO_15           (0x00000001L <<  15)
#define SUNXI_PIO_16           (0x00000001L <<  16)
#define SUNXI_PIO_17           (0x00000001L <<  17)
#define SUNXI_PIO_18           (0x00000001L <<  18)
#define SUNXI_PIO_19           (0x00000001L <<  19)
#define SUNXI_PIO_20           (0x00000001L <<  20)
#define SUNXI_PIO_21           (0x00000001L <<  21)
#define SUNXI_PIO_22           (0x00000001L <<  22)


//----------------------------------//
//       CONSTANT DEFINITIONS       //
//----------------------------------//


#define SUNXI_SW_PORTC_IO_BASE  (0x01c20800)
#define SUNXI_GPIO_DATA_OFFSET  (0x10)
#define SUNXI_GPIO_PULL_OFFSET	(0x1C)
#define SUNXI_GPIO_INPUT        (0)
#define SUNXI_GPIO_OUTPUT       (1)

#define DISABLE 	0
#define FULL_UP		1
#define FULL_DOWN	2
#define RESERVED	3
// Debug function
//#define SUNXI_GPIO_DEBUG

//----------------------------------//
//        METHOD DEFINITIONS        //
//----------------------------------//

#define SUNXI_PIO_GET_BIT_INDEX(a) ( (a&SUNXI_PIO_00)?0:(a&SUNXI_PIO_01)?1:(a&SUNXI_PIO_02)?2:(a&SUNXI_PIO_03)?3:(a&SUNXI_PIO_04)?4:(a&SUNXI_PIO_05)?5:(a&SUNXI_PIO_06)?6:(a&SUNXI_PIO_07)?7:(a&SUNXI_PIO_08)?8:(a&SUNXI_PIO_09)?9:(a&SUNXI_PIO_10)?10:(a&SUNXI_PIO_11)?11:(a&SUNXI_PIO_12)?12:(a&SUNXI_PIO_13)?13:(a&SUNXI_PIO_14)?14:(a&SUNXI_PIO_15)?15:(a&SUNXI_PIO_16)?16:(a&SUNXI_PIO_17)?17:(a&SUNXI_PIO_18)?18:(a&SUNXI_PIO_19)?19:(a&SUNXI_PIO_20)?20:(a&SUNXI_PIO_21)?21:(a&SUNXI_PIO_22)?22:0)

int GPIO_fast_init();

int Get_addr_value(int iOffset);

void Set_addr_value(int iOffset, int iValue);

int GPIO_fast_config(int gpio, int inout);

int GPIO_fast_pull_config(int gpio, int value);

int GPIO_fast_setvalue(int gpio, int value);

int GPIO_fast_getvalue(int gpio);

int APB_fast_init();

unsigned int APB_fast_getvalue();

int APB_fast_setvalue(unsigned int value);

int GPIO_class_config(int gpio, int inout);
int GPIO_class_setvalue(int gpio, int value);
int GPIO_class_getvalue(int gpio);

#ifdef __cplusplus
}
#endif

#endif /* _DRV_GPIO_H */

