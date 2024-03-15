

#include "drv_gpio.h"
#include "appdef.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <sys/mman.h>
#include <unistd.h>


static const int verbose = 0;

/* Fast Mode */
static void *SUNXI_PIO_BASE;

//------------------------------------------------------------------------------
int GPIO_fast_init()
//------------------------------------------------------------------------------
{
    int fd;
    unsigned int addr_start, addr_offset;
    unsigned int PageSize, PageMask;
    void *pc;

    fd = open("/dev/mem", O_RDWR);
    if(fd < 0) {
       perror("Unable to open /dev/mem");
       return(-1);
    }

    PageSize = sysconf(_SC_PAGESIZE);
    PageMask = (~(PageSize-1));

    addr_start  = SUNXI_SW_PORTC_IO_BASE &  PageMask;
    addr_offset = SUNXI_SW_PORTC_IO_BASE & ~PageMask;

    pc = (void *)mmap(0, PageSize*2, PROT_READ|PROT_WRITE, MAP_SHARED, fd, addr_start);
    if(pc == MAP_FAILED) {
       perror("Unable to mmap file");
       printf("pc:%8.8x\n", (unsigned int)pc);
       return(-1);
    }

    SUNXI_PIO_BASE = pc;
    SUNXI_PIO_BASE = (void*)((unsigned int)SUNXI_PIO_BASE + addr_offset);

 //   SUNXI_PIO_I_DATA=(unsigned int *) (SUNXI_PIO_BASE +  SUNXI_PORT_I_BASE + SUNXI_GPIO_DATA_OFFSET);

    close(fd);
    return 0;
}

int Get_addr_value(int iOffset)
{
    return *(int*)((int)SUNXI_PIO_BASE + iOffset);
}

void Set_addr_value(int iOffset, int iValue)
{
    *(int*)((int)SUNXI_PIO_BASE + iOffset) = iValue;
}

//------------------------------------------------------------------------------
void my_gpio_cfg_output(unsigned int port_base, unsigned int pin)
//------------------------------------------------------------------------------
{
    unsigned int cfg;

    unsigned int index  = (SUNXI_PIO_GET_BIT_INDEX(pin)>>3);
    unsigned int offset = ((SUNXI_PIO_GET_BIT_INDEX(pin)& 0x7) << 2);
    unsigned int *c     = (unsigned int *)  ((SUNXI_PIO_BASE + port_base + index*4));

    cfg = *c;
    cfg &= ~(0xF << offset);
    cfg |= SUNXI_GPIO_OUTPUT << offset;

    *c = cfg;
}

//------------------------------------------------------------------------------
void my_gpio_cfg_input(unsigned int port_base, unsigned int pin)
//------------------------------------------------------------------------------
{
   unsigned int cfg;

   unsigned int index  = (SUNXI_PIO_GET_BIT_INDEX(pin)>>3);
   unsigned int offset = ((SUNXI_PIO_GET_BIT_INDEX(pin)& 0x7) << 2);
   unsigned int *c     = (unsigned int *) ((SUNXI_PIO_BASE + port_base + index*4));

   cfg = *c;
   cfg &= ~(0xF << offset);
   cfg |= SUNXI_GPIO_INPUT << offset;

    *c = cfg;
}

void my_gpio_cfg(unsigned int port_base, unsigned int pin, unsigned int mulsel)
{
    unsigned int cfg;

    unsigned int index  = (SUNXI_PIO_GET_BIT_INDEX(pin)>>3);
    unsigned int offset = ((SUNXI_PIO_GET_BIT_INDEX(pin)& 0x7) << 2);
    unsigned int *c     = (unsigned int *) ((SUNXI_PIO_BASE + port_base + index*4));

    cfg = *c;
    cfg &= ~(0xF << offset);
    cfg |= mulsel << offset;

     *c = cfg;
}

void my_gpio_pull_cfg(unsigned int port_base, unsigned int pin, unsigned int value)
{
    unsigned int cfg;

    unsigned int index  = (SUNXI_PIO_GET_BIT_INDEX(pin) >> 4);
    unsigned int offset = ((SUNXI_PIO_GET_BIT_INDEX(pin)& 0xF) << 1);
    unsigned int *c     = (unsigned int *) ((SUNXI_PIO_BASE + port_base + SUNXI_GPIO_PULL_OFFSET + index*4));

    cfg = *c;
    cfg &= ~(0x3 << offset);
    cfg |= value << offset;

     *c = cfg;
}


//------------------------------------------------------------------------------
void my_gpio_set_output(unsigned int port_base,unsigned int pin)
//------------------------------------------------------------------------------
{
  unsigned int  *dat =  (unsigned int *) (SUNXI_PIO_BASE  +  port_base + SUNXI_GPIO_DATA_OFFSET);
  *(dat) |= pin;
}

//------------------------------------------------------------------------------
void my_gpio_clear_output(unsigned int port_base,unsigned int pin)
//------------------------------------------------------------------------------
{
  unsigned int  *dat = (unsigned int *) (SUNXI_PIO_BASE +  port_base + SUNXI_GPIO_DATA_OFFSET);
  *(dat) &= ~(pin);
}

//------------------------------------------------------------------------------
int my_gpio_get_input(unsigned int port_base ,unsigned int pin)
//------------------------------------------------------------------------------
{
  unsigned int  *dat =  (unsigned int *) (SUNXI_PIO_BASE +  port_base + SUNXI_GPIO_DATA_OFFSET);
  return (*dat & pin);
}

int GPIO_fast_config(int gpio, int inout)
{
    int port = 0x24 * (gpio / 32);
    int pin = 1 << (gpio % 32);
    if (inout == IN) {
        my_gpio_cfg_input(port, pin);
    } else if (inout == OUT) {
        my_gpio_cfg_output(port, pin);
    } else {
        my_gpio_cfg(port, pin, inout);
    }
    return 0;
}

int GPIO_fast_pull_config(int gpio, int value)
{
    int port = 0x24 * (gpio / 32);
    int pin = 1 << (gpio % 32);
    my_gpio_pull_cfg(port, pin, value);
    return 0;
}


int GPIO_fast_setvalue(int gpio, int value)
{
    int port = 0x24 * (gpio / 32);
    int pin = 1 << (gpio % 32);
    if (value == 1)
        my_gpio_set_output(port, pin);
    else
        my_gpio_clear_output(port, pin);
    usleep(1);
    return 0;
}

int GPIO_fast_getvalue(int gpio)
{
    int port = 0x24 * (gpio / 32);
    int pin = 1 << (gpio % 32);
    return my_gpio_get_input(port, pin);
    return 0;
}

static void *SUNXI_APB_BASE;
#define SUNXI_APB_DATA_OFFSET  (0x58)
int APB_fast_init()
//------------------------------------------------------------------------------
{
    int fd;
    unsigned int addr_start, addr_offset, addr;
    unsigned int PageSize, PageMask;
    void *pc;

    fd = open("/dev/mem", O_RDWR);
    if(fd < 0) {
       perror("Unable to open /dev/mem");
       return(-1);
    }

    PageSize = sysconf(_SC_PAGESIZE);
    PageMask = (~(PageSize-1));

    addr_start  = 0x01c20000 &  PageMask;
    addr_offset = 0x01c20000 & ~PageMask;

    pc = (void *)mmap(0, PageSize*2, PROT_READ|PROT_WRITE, MAP_SHARED, fd, addr_start);
    if(pc == MAP_FAILED) {
       perror("Unable to mmap file");
       printf("pc:%8.8x\n", (unsigned int)pc);
       return(-1);
    }

    SUNXI_APB_BASE = pc;
    SUNXI_APB_BASE += addr_offset;

 //   SUNXI_PIO_I_DATA=(unsigned int *) (SUNXI_PIO_BASE +  SUNXI_PORT_I_BASE + SUNXI_GPIO_DATA_OFFSET);

    close(fd);
    return 0;
}

unsigned int my_apb_get_input()
//------------------------------------------------------------------------------
{
  unsigned int  *dat =  (unsigned int *) (SUNXI_APB_BASE + SUNXI_APB_DATA_OFFSET);
  return (*dat);
}


unsigned int APB_fast_getvalue()
{
    return my_apb_get_input();
}

//------------------------------------------------------------------------------
void my_apb_set_value(unsigned int value)
//------------------------------------------------------------------------------
{
  unsigned int  *dat =  (unsigned int *) (SUNXI_APB_BASE  +  SUNXI_APB_DATA_OFFSET);
  *(dat) = value;
}

int APB_fast_setvalue(unsigned int value)
{
    my_apb_set_value(value);

    usleep(1);
    return 0;
}


int GPIO_class_config(int gpio, int inout)
{
    char szCmd[256] = { 0 };

    sprintf(szCmd, "echo %d > /sys/class/gpio/export", gpio);
    system(szCmd);

    if(inout == IN)
    {
        sprintf(szCmd, "echo in > /sys/class/gpio/gpio%d/direction", gpio);
        system(szCmd);
    }
    else if(inout == OUT)
    {
        sprintf(szCmd, "echo out > /sys/class/gpio/gpio%d/direction", gpio);
        system(szCmd);
    }

    return 0;
}

int GPIO_class_setvalue(int gpio, int value)
{
    char szCmd[256] = { 0 };

    sprintf(szCmd, "echo %d > /sys/class/gpio/gpio%d/value", value, gpio);
    system(szCmd);

    return 0;
}

int GPIO_class_getvalue(int gpio)
{
    int value = 0;
    char szPath[256] = { 0 };
    char szLine[257] = { 0 };
    FILE* fp = NULL;

    sprintf(szPath, "/sys/class/gpio/gpio%d/value", gpio);

    fp = fopen(szPath, "r");
    if(fp)
    {
        fgets(szLine, sizeof(szLine), fp);
        fclose(fp);

        value = atoi(szLine);
    }

    return value;
}
