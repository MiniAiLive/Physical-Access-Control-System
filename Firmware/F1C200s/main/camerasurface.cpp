#include "camerasurface.h"
#include "camera_api.h"
#include "shared.h"
#include "i2cbase.h"
#include "drv_gpio.h"
#include "msg.h"
#include "settings.h"

#include <math.h>
#include <string.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/poll.h>
#include <asm/types.h>
#include <linux/fb.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <linux/videodev2.h>
#include <sys/types.h>
#include "ion_mem_alloc.h"
#include "lcdtask.h"
#include "sunxi_display_v1.h"

struct SunxiMemOpsS* GetMemAdapterOpsS();
extern "C" void nv_rotage90(unsigned int width, unsigned int height,
    unsigned char* src_addr, unsigned char* dst_addr);

extern "C" void nv_rotage270(unsigned int width, unsigned int height,
    unsigned char* src_addr, unsigned char* dst_addr);

LCDTask*        g_pLCDTask = NULL;

CameraSurface::CameraSurface()
{
    m_iRunning = 0;
}

CameraSurface::~CameraSurface()
{

}


void CameraSurface::Start()
{
    m_iRunning = 1;
    Thread::Start();
}

void CameraSurface::Stop()
{
    m_iRunning = 0;
    Thread::Wait();
}

void CameraSurface::run()
{
    struct v4l2_buffer buf;

    int ret = -1;

    if(g_xSS.iFirstCamInited == 0)
        g_xSS.iCamInited = camera_init(CAM_ID, WIDTH_720, HEIGHT_480, FPS, FRAME_NUM);

    if(g_xSS.iCamInited != 0)
    {
        printf("camera init error!\n");
        LCDTask::DispOpen();
        LCDTask::FB_Init();
        LCDTask::DispOn();
        if(g_pLCDTask != NULL)
            g_pLCDTask->Update();
        usleep(30 * 1000);
        return;
    }

    float rOldTime = Now();
    int iFrameCount = 0;

    SunxiMemOpsS* pMemops = GetMemAdapterOpsS();
    SunxiMemOpen(pMemops);

    unsigned char* pVirBuf = (unsigned char*)SunxiMemPalloc(pMemops, (WIDTH_720 * HEIGHT_480 * 3 / 2) * FRAME_NUM);
    unsigned char* pPhyBuf = (unsigned char*)SunxiMemGetPhysicAddressCpu(pMemops, pVirBuf);

    int iVideoStart = 0;
    int iSkipCount = g_xSS.iFirstCamInited == 1 ? 0 : 5;

    unsigned int iOldAddr = 0;

    while(m_iRunning)
    {
        ret = wait_camera_ready (CAM_ID);
        if (ret < 0)
        {
            g_xSS.iCamInited = -1;
            LCDTask::DispOpen();
            LCDTask::FB_Init();
            LCDTask::DispOn();
            if(g_pLCDTask != NULL)
                g_pLCDTask->Update();
            usleep(30 * 1000);
            break;
        }

        CLEAR(buf);
        memset(&buf, 0, sizeof(struct v4l2_buffer));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        ioctl(cam_fd, VIDIOC_DQBUF, &buf);

//        printf("Frame: %d, BufLen=%x, %f\n", iFrameCount, buffers[buf.index].length, Now() - rOldTime);
        rOldTime = Now();

        unsigned int phyaddr;
        phyaddr = buf.m.offset;

        float rOld = Now();

        memcpy(pVirBuf + buf.index * (WIDTH_720 * HEIGHT_480 * 3 / 2), buffers[buf.index].start, WIDTH_720 * HEIGHT_480 * 3 / 2);
//        printf("roate time; %f, %d\n", Now() - rOld, buf.index);

        if(iFrameCount % 30 == 0 && 0)
        {
            char szName[256] = {0};
            sprintf(szName, "/tmp/cvbs_%d.bin", iFrameCount);
            FILE* fp = fopen(szName, "wb");
            if(fp)
            {
                fwrite(buffers[buf.index].start, WIDTH_720 * HEIGHT_480 * 3 / 2, 1, fp);
                fflush(fp);
                fclose(fp);
            }
        }
        phyaddr = (unsigned int)pPhyBuf + buf.index * (WIDTH_720 * HEIGHT_480 * 3 / 2);

        if(phyaddr && iFrameCount >= iSkipCount)
        {
            if(iVideoStart == 0)
            {
                iVideoStart = 1;
                LCDTask::DispOpen();
                LCDTask::FB_Init();
                LCDTask::VideoStart(DISP_FORMAT_YUV420_SP_VUVU);
                LCDTask::DispOn();
                printf("First Cam Frame: %f\n", Now());
            }
            LCDTask::VideoMap(WIDTH_720, HEIGHT_480, &phyaddr);
        }

        iOldAddr = phyaddr;

        ioctl(cam_fd, VIDIOC_QBUF, &buf);
        iFrameCount ++;
    }

    LCDTask::VideoStop();

    SunxiMemPfree(pMemops, pVirBuf);
    SunxiMemClose(pMemops);

    camera_release(CAM_ID);
}

