
#include "camera_api.h"
#include "i2cbase.h"
#include "appdef.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <fcntl.h>
#include <getopt.h>
#include <unistd.h>
#include <errno.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>

#include <sys/mman.h>
#include <sys/ioctl.h>
#include <asm/types.h>
#include "videodev2.h"
#include <linux/fb.h>
#include <sys/poll.h>
#include <linux/i2c-dev.h>
#include <time.h>
#include "_mm/buffer.h"
#include "_mm/cma_allocator.h"
#include "settings.h"

using namespace rk;

int cam_fd[2] = { -1, -1 };

struct buffer *buffers[2];
static int nbuffers[2];
static const char* _iq_file="/etc/cam_iq.xml";
static void* _rkisp_engine[2] = { 0 };

static int xioctl(int fh, int request, void *arg)
{
    int r;
    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);
    return r;
}


int wait_camera_ready(int id)
{
#if 0
    fd_set fds;
    struct timeval tv;
    int r;

    FD_ZERO(&fds);
    FD_SET(cam_fd[id], &fds);

    /* Timeout */
    tv.tv_sec  = 2;
    tv.tv_usec = 0;

    r = select(cam_fd[id] + 1, &fds, NULL, NULL, &tv);
    if (r == -1)
    {
        printf("select err   %d, %s, %d\n", id, strerror(errno), errno);
        return -1;
    }
    else if (r == 0)
    {
        printf("select timeout  %d, %s, %d\n", id, strerror(errno), errno);
        return -2;
    }
#else
    int ret;
    struct pollfd pfd;
    pfd.fd = cam_fd[id];
    pfd.events = POLLIN | POLLERR;
    ret = poll(&pfd, 1, 2000);
    if (ret < 0) {
      printf("%s: polling error (error %d)\n", __func__, ret);
      return ret;
    } else if (!ret) {
      printf("%s: no data in %ld millisecs\n", __func__, (long)2000);
      return -ETIMEDOUT;
    }
#endif
    return 0;
}

int wait_camera_ready_ext(int id, int timeout)
{
#if 0
    fd_set fds;
    struct timeval tv;
    int r;

    FD_ZERO(&fds);
    FD_SET(cam_fd[id], &fds);

    /* Timeout */
    tv.tv_sec  = 2;
    tv.tv_usec = 0;

    r = select(cam_fd[id] + 1, &fds, NULL, NULL, &tv);
    if (r == -1)
    {
        printf("select err   %d, %s, %d\n", id, strerror(errno), errno);
        return -1;
    }
    else if (r == 0)
    {
        printf("select timeout  %d, %s, %d\n", id, strerror(errno), errno);
        return -2;
    }
#else
    int ret;
    struct pollfd pfd;
    pfd.fd = cam_fd[id];
    pfd.events = POLLIN | POLLERR;
    ret = poll(&pfd, 1, timeout);
    if (ret < 0) {
      //printf("%s: polling error (error %d)\n", __func__, ret);
      return ret;
    } else if (!ret) {
      //printf("%s: no data in %ld millisecs\n", __func__, (long)timeout);
      return -ETIMEDOUT;
    }
#endif
    return 0;
}

int camera_init(int id, int width, int height, int fps, int frameNums, int rotate, int isp_no_use)
{
    char dev_name[32];
    int i, ret;
    int fd = -1;
    struct v4l2_capability cap;
    struct v4l2_format fmt;
    struct v4l2_pix_format subch_fmt;
    struct v4l2_requestbuffers req;
    struct v4l2_buffer v4l2Buf;
    enum v4l2_buf_type type;

    if (cam_fd[id] > -1)
        return 0;

    if(id == CLR_CAM)
    {
        int iCamIdx = 5;
        for(i = MAX_VIDEO_NUM - 1; i >= 0 ; i --)
        {
            sprintf(dev_name, "/dev/video%d", i);
            int fd = open(dev_name, O_RDONLY);
            if(fd == -1)
                continue;

            struct v4l2_capability capability;
            memset(&capability, 0, sizeof(struct v4l2_capability));
            if (ioctl(fd, VIDIOC_QUERYCAP, &capability) < 0)
            {
//                printf("Video device(%s): query capability not supported.\n", dev_name);
                close(fd);
                continue;
            }

            close(fd);

            if(!strcmp((char*)(capability.driver), "cif_cif10") &&
                    !strcmp((char*)(capability.card), "rv1108"))
            {
                iCamIdx = i;
                break;
            }
        }

        sprintf(dev_name, "/dev/video%d", iCamIdx);
    }
    else if(id == IR_CAM)
    {
        int iCamIdx = 2;
        for(i = 0; i < MAX_VIDEO_NUM; i ++)
        {
            sprintf(dev_name, "/dev/video%d", i);
            int fd = open(dev_name, O_RDONLY);
            if(fd == -1)
                continue;

            struct v4l2_capability capability;
            memset(&capability, 0, sizeof(struct v4l2_capability));
            if (ioctl(fd, VIDIOC_QUERYCAP, &capability) < 0)
            {
//                printf("Video device(%s): query capability not supported.\n", dev_name);
                close(fd);
                continue;
            }

            close(fd);
            if(!strcmp((char*)(capability.driver), "rkisp11") &&
                    !strcmp((char*)(capability.card), "rkisp11_mainpath"))
            {
                iCamIdx = i;
                break;
            }
        }
        sprintf(dev_name, "/dev/video%d", iCamIdx);
    }

    printf("open %s\n", dev_name);
    if ((fd = open(dev_name, O_RDWR /* required */ /*| O_NONBLOCK*/, 0)) < 0) {
        printf("can't open %s(%s)\n", dev_name, strerror(errno));
        goto open_err;
    }

    if (-1 == xioctl(fd, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            printf("%s is no V4L2 device\n", dev_name);
            goto err;
        } else {
            printf("%s errir %d, %s\n", "VIDIOC_QUERYCAP", errno, strerror(errno));
            goto err;
        }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        printf("%s is no video capture device\n", dev_name);
        goto err;
    }

    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        printf("%s does not support streaming i/o\n", dev_name);
        goto err;
    }

    /* set image format */
    CLEAR(fmt);
    CLEAR(subch_fmt);

    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
#if (SEND_CAM_VIA_CVBS == 0 || 1)
    if(id == IR_CAM)
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_SBGGR10;
    else
#endif        
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV12;

    fmt.fmt.pix.sizeimage = (width * height * 3 / 2);
#if (SEND_CAM_VIA_CVBS == 0 || 1)
    if(id == IR_CAM)
        fmt.fmt.pix.sizeimage = (width * height * 2);
#endif

    if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
    {
        printf("%s errir %d, %s\n", "VIDIOC_S_FMT", errno, strerror(errno));
        goto err;
    }

#if (SEND_CAM_VIA_CVBS == 0 || 1)
    if(id == IR_CAM)
    {

        struct v4l2_input inp;
        inp.index = 0;
        inp.type = V4L2_INPUT_TYPE_CAMERA;

        /* set input input index */
        if (xioctl(fd, VIDIOC_S_INPUT, &inp) == -1) {
            printf("VIDIOC_S_INPUT1 error\n");
            goto err;
        }

        CLEAR(fmt);
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = width;
        fmt.fmt.pix.height = height;
        fmt.fmt.pix.bytesperline = width;
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_SBGGR10;
        fmt.fmt.pix.colorspace = V4L2_COLORSPACE_JPEG;
        fmt.fmt.pix.sizeimage = (width * height * 2);

        if (-1 == xioctl(fd, VIDIOC_S_FMT, &fmt))
        {
            printf("%s errir %d, %s\n", "VIDIOC_S_FMT1", errno, strerror(errno));
            goto err;
        }
    }

#endif

    CLEAR(req);
    req.count = frameNums;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_USERPTR;

    if (-1 == xioctl(fd, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            printf("%s does not support memory mapping\n", dev_name);
            goto err;
        } else {
            printf("%s errir %d, %s\n", "VIDIOC_REQBUFS", errno, strerror(errno));
            goto err;
        }
    }

    if (req.count < 2) {
        printf("Insufficient buffer memory on %s\n", dev_name);
        goto err;
    }

    buffers[id] = (buffer*)calloc(req.count, sizeof(struct buffer));
    for (i = 0; i < req.count; ++i) {
        buffers[id][i].start = MAP_FAILED;
    }
    if (!buffers[id]) {
        printf("Out of memory\n");
        goto buffer_rel;
    }

    for (i = 0; i < req.count; ++i) {
        Buffer* buffer = CmaAlloc(fmt.fmt.pix.sizeimage);

        buffers[id][i].length = buffer->size();
        buffers[id][i].fd = buffer->fd();
        buffers[id][i].buf_org = (void*)buffer;
        buffers[id][i].start =
                mmap(NULL /* start anywhere */,
                      buffers[id][i].length,
                      PROT_READ | PROT_WRITE /* required */,
                      MAP_SHARED /* recommended */,
                      buffer->fd(), 0);

        if(buffers[id][i].start == MAP_FAILED)
        {
            printf("%s errir %d, %s\n", "VIDIOC_REQBUFS", errno, strerror(errno));
            goto unmap;
        }
    }

    for (i = 0; i < req.count; ++i) {
        memset(&v4l2Buf, 0, sizeof(v4l2Buf));

        v4l2Buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        v4l2Buf.memory = V4L2_MEMORY_USERPTR;
        v4l2Buf.index = i;
        v4l2Buf.m.userptr = (unsigned long)buffers[id][i].start;
        v4l2Buf.length = buffers[id][i].length;

        ret = ioctl(fd, VIDIOC_QBUF, &v4l2Buf);
        if (ret < 0) {
            printf("%s: QBUF failed; index %d type %d memory %d (error %d)", __func__, v4l2Buf.index, v4l2Buf.type, v4l2Buf.memory, ret);
            goto unmap;
        }
    }

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(fd, VIDIOC_STREAMON, &type)) {
        printf("%s errir %d, %s\n", "VIDIOC_STREAMON", errno, strerror(errno));
        goto unmap;
    }

    cam_fd[id] = fd;    
    nbuffers[id] = req.count;

    return 0;

unmap:
    for (i = 0; i < req.count; i++) {
        if (buffers[id][i].start != MAP_FAILED)
            munmap(buffers[id][i].start, buffers[id][i].length);
        if(buffers[id][i].buf_org)
            CmaFree((Buffer*)buffers[id][i].buf_org);
    }
buffer_rel:
    free(buffers[id]);
err:
    close(fd);
open_err:
    cam_fd[id] = -1;
    return -1;
}

int camera_release (int id)
{
    int i;
    int ret;
    enum v4l2_buf_type type;
    struct v4l2_buffer buf;

    if (cam_fd[id] == -1)
        return -1;

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (-1 == xioctl(cam_fd[id], VIDIOC_STREAMOFF, &type)) {
        printf("%s errir %d, %s\n", "VIDIOC_STREAMOFF", errno, strerror(errno));
        return -1;
    }

    for (i = 0; i < nbuffers[id]; i++)
    {
        if (buffers[id][i].start != MAP_FAILED)
            munmap(buffers[id][i].start, buffers[id][i].length);
    }

    for(i = 0; i < nbuffers[id]; i ++)
    {
        if(buffers[id][i].buf_org)
            CmaFree((Buffer*)buffers[id][i].buf_org);
    }

    free(buffers[id]);

    if (close (cam_fd[id]) < 0)
        perror ("close");

    cam_fd[id] = -1;

    return 0;
}

#define I2C_ADDR_JX_H62 0x36
static int g_iIRCam0 = -1;
static int g_iIRCam1 = -1;

int camera_ir0_i2c_open()
{
    if(g_iIRCam0 < 0)
    {
        if ((g_iIRCam0 = open("/dev/i2c-1", O_RDWR)) < 0)
        {
            printf("IR0: Failed to open the bus: Addr = %x\n", I2C_ADDR_JX_H62);
            return -1;
        }

        if (ioctl(g_iIRCam0, I2C_SLAVE_FORCE, I2C_ADDR_JX_H62) < 0)
        {
            printf("IR0: Failed to acquire bus access and/or talk to slave.\n");
            g_iIRCam0 = -1;
            return -1;
        }
    }

    return g_iIRCam0;
}


int camera_ir0_set_regval(unsigned char regaddr, unsigned char regval)
{
    int ret = 0;
    char szBuf[2] = { 0 };
    if (camera_ir0_i2c_open() < 0)
        return -1;

    szBuf[0] = regaddr;
    szBuf[1] = regval;

//    printf("IR0: %x, %x\n", regaddr, regval);
    ret = write(g_iIRCam0, szBuf, 2);
    if (ret != 2)
    {
        printf("IR0: Failed to write ir0 camera regs. %d\n", ret);
        return -2;
    }

    return 0;
}


int camera_ir0_get_regval(unsigned char regaddr, unsigned char* regval)
{
    int ret = 0;
    char szBuf[2] = { 0 };
    if(camera_ir0_i2c_open() < 0)
        return -1;

    szBuf[0] = regaddr;
    ret = write(g_iIRCam0, szBuf, 1);
    if (ret != 1)
    {
        printf("IR0-1: Failed to write ir0 camera regs. %d\n", ret);
        return -2;
    }

    ret = read(g_iIRCam0, szBuf, 1);
    if (ret != 1)
    {
        printf("IR0-2: Failed to read ir0 camera regs. %d\n", ret);
        return -2;
    }

    if(regval)
        *regval = szBuf[0];

    return 0;
}

int camera_ir1_i2c_open()
{
    if(g_iIRCam1 < 0)
    {
        if ((g_iIRCam1 = open("/dev/i2c-3", O_RDWR)) < 0)
        {
            printf("IR1: Failed to open the bus: Addr = %x\n", I2C_ADDR_JX_H62);
            return -1;
        }

        if (ioctl(g_iIRCam1, I2C_SLAVE_FORCE, I2C_ADDR_JX_H62) < 0)
        {
            printf("IR1: Failed to acquire bus access and/or talk to slave.\n");
            g_iIRCam1 = -1;
            return -1;
        }
    }

    return g_iIRCam1;
}

int camera_ir1_set_regval(unsigned char regaddr, unsigned char regval)
{
    int ret = 0;
    char szBuf[2] = { 0 };
    if(camera_ir1_i2c_open() < 0)
        return -1;

    szBuf[0] = regaddr;
    szBuf[1] = regval;

//    printf("IR1: %x, %x\n", regaddr, regval);

    ret = write(g_iIRCam1, szBuf, 2);
    if (ret != 2)
    {
        printf("IR1: Failed to write ir1 camera regs. %d\n", ret);
        return -2;
    }

    return 0;
}

int camera_ir1_get_regval(unsigned char regaddr, unsigned char* regval)
{
    int ret = 0;
    char szBuf[2] = { 0 };
    if(camera_ir1_i2c_open() < 0)
        return -1;

    szBuf[0] = regaddr;
    ret = write(g_iIRCam1, szBuf, 1);
    if (ret != 1) {
        printf("IR1-1: Failed to write ir1 camera regs. %d\n", ret);
        return -2;
    }
    ret = read(g_iIRCam1, szBuf, 1);
    if (ret != 1) {
        printf("IR1-2: Failed to read ir1 camera regs. %d\n", ret);
        return -2;
    }

    if(regval)
        *regval = szBuf[0];

    return 0;
}



#define V4L2_CID_S_G_REG        (V4L2_CID_BASE+52)
#define V4L2_CID_IRLED_CTRL     (V4L2_CID_BASE+53)
#define V4L2_CID_IRLED_ON       (V4L2_CID_BASE+54)

//return 0 : OK, -1 : Err
/* ex:
    reg_val = 0x40;
    if (set_camera_regval(0x10	, &reg_val) == 0)
        printf("reg write ok\n");
    else
        printf("reg write error\n");
*/
int camera_set_regval(int id, unsigned char regaddr, unsigned char regval)
{
    int ret = -1;    

    if(id == IR_CAM)
        return camera_ir0_set_regval(regaddr, regval);
    else if(id == IR_CAM1)
        return camera_ir1_set_regval(regaddr, regval);
    else
    {
        struct v4l2_streamparm parms;

        parms.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        parms.parm.capture.extendedmode = V4L2_CID_S_G_REG;
        parms.parm.capture.capturemode = 4; // set video capture mode

        parms.parm.capture.timeperframe.denominator = regaddr;
        parms.parm.capture.timeperframe.numerator = regval;

        ret = ioctl(cam_fd[id], VIDIOC_S_PARM, &parms);
    }

    return ret;
}

//return 0 : OK, -1 : Err
/* ex:
    if (get_camera_regval(0x0B	, &reg_val) == 0)
        printf("Product ID LSB = 0x%02x\n", reg_val);
    else
        printf("reg read error\n");
*/

int camera_get_regval(int id, unsigned char regaddr, unsigned char *regval)
{
    int ret = -1;
    if(id == IR_CAM)
        return camera_ir0_get_regval(regaddr, regval);
    else if(id == IR_CAM1)
        return camera_ir1_get_regval(regaddr, regval);
    else
    {
        struct v4l2_streamparm parms;

        parms.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

        parms.parm.capture.extendedmode = V4L2_CID_BASE + 52;

        parms.parm.capture.timeperframe.denominator = 0xED;
        parms.parm.capture.timeperframe.numerator = regaddr;

        ioctl(cam_fd[id], VIDIOC_S_PARM, &parms);

        ret = ioctl(cam_fd[id], VIDIOC_G_PARM, &parms);

        if (regval != NULL)
            *regval = (unsigned char)(parms.parm.raw_data[0]);

        return (unsigned char)(parms.parm.raw_data[0]);
    }
}

int camera_clr_set_exp(int value)
{
    //value = value * 1 / 3;
    camera_set_regval(CLR_CAM, 0xfe, 0); //page select
    camera_set_regval(CLR_CAM, 0x04, (unsigned char)value);
    camera_set_regval(CLR_CAM, 0x03, (unsigned char)((value >> 8) & 0xFF));
    return 0;
}

int camera_clr_get_exp()
{
    camera_set_regval(CLR_CAM, 0xfe, 0); //page select
    return camera_get_regval(CLR_CAM, 0x04, NULL) | (camera_get_regval(CLR_CAM, 0x03, NULL) << 8);
}

int camera_clr_set_gain(int value)
{
    camera_set_regval(CLR_CAM, 0xfe, 0); //page select
    camera_set_regval(CLR_CAM, 0xb0, (unsigned char)value);
    return 0;
}

int camera_clr_get_gain()
{
    camera_set_regval(CLR_CAM, 0xfe, 0); //page select
    return camera_get_regval(CLR_CAM, 0xb0, NULL);
}

int camera_set_irled(int id, int enable, int count)
{
    int ret = -1;
//    printf("camera_set_irled: %d\n", enable);
#if 1
    struct v4l2_streamparm parms;

    parms.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parms.parm.capture.extendedmode = V4L2_CID_IRLED_CTRL;

    parms.parm.capture.timeperframe.numerator = enable;
    parms.parm.capture.timeperframe.denominator = count;

    ret = ioctl(cam_fd[id], VIDIOC_S_PARM, &parms);
#endif
    return ret;
}

int camera_set_irled_on(int id, int on)
{
    int ret = -1;
//    printf("camera_set_irled_on: %d\n", on);
#if 1
    struct v4l2_streamparm parms;

    parms.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parms.parm.capture.extendedmode = V4L2_CID_IRLED_ON;
    parms.parm.capture.timeperframe.numerator = on;

    ret = ioctl(cam_fd[id], VIDIOC_S_PARM, &parms);
#endif
    return ret;
}

int camera_set_exp_byreg(int id, int value)
{
    camera_set_regval(id, 0x01, (unsigned char)(value & 0xFF));
    camera_set_regval(id, 0x02, (unsigned char)((value >> 8) & 0x0F));
    usleep(1000);

//    printf("Exp: %d\n", value);
    return 0;
}

int camera_set_gain_byreg(int id, int value)
{
    camera_set_regval(id, 0x00, value);
//    printf("Gain: %d\n", value);
    return 0;
}



int getExp(int id)
{
    unsigned char pageset = 0x00;
    unsigned char b0x10Value, b0xFValue;
    camera_set_regval(id, 0xfe, pageset);
    camera_get_regval(id, 0x03, &b0xFValue);
    camera_get_regval(id, 0x04, &b0x10Value);
//    printf("    exp reg value = %02x : %02x\n", b0x10Value, b0xFValue);
    int nValue = ((int)b0xFValue << 8 | b0x10Value);

    return nValue;

}

int getGain(int id)
{
    unsigned char pageset = 0x00;
    unsigned char b0xFValue;
    camera_set_regval(id, 0xfe, pageset);
    camera_get_regval(id, 0xB0, &b0xFValue);
    return b0xFValue;
}

int getYAVGValue(int id)
{
    unsigned char b0x2FValue;
    camera_get_regval(id, 0x2F, &b0x2FValue);

    int nValue = b0x2FValue;

    return nValue;
}


int create_buffer(struct buffer* buf, int size)
{
    if(buf->start != NULL && buf->start != MAP_FAILED)
        return 0;

    Buffer* buffer = CmaAlloc(size);

    buf->length = buffer->size();
    buf->fd = buffer->fd();
    buf->buf_org = (void*)buffer;
    buf->start =
            mmap(NULL /* start anywhere */,
                  buf->length,
                  PROT_READ | PROT_WRITE /* required */,
                  MAP_SHARED /* recommended */,
                  buffer->fd(), 0);

    if(buf->start == MAP_FAILED)
    {
        printf("%s errir %d, %s\n", "VIDIOC_REQBUFS", errno, strerror(errno));
        return  -1;
    }

    return 0;
}

int delete_buffer(struct buffer* buf)
{
    if(buf == NULL)
        return -1;

    if(buf->start == NULL)
        return -1;

    if (buf->start != MAP_FAILED)
        munmap(buf->start, buf->length);
    if(buf->buf_org)
        CmaFree((Buffer*)buf->buf_org);

    memset(buf, 0, sizeof(buffer));

    return 0;
}


int camera_switch(int id, int camid)
{
    int ret = -1;
    struct v4l2_streamparm parms;

//    printf("camera_switch: %d\n", camid);

    parms.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parms.parm.capture.extendedmode = V4L2_CID_BASE + 55;

    parms.parm.capture.timeperframe.numerator = camid;

    ret = ioctl(cam_fd[id], VIDIOC_S_PARM, &parms);

    return ret;
}
