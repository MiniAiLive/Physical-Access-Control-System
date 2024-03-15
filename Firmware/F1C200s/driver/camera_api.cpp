
#include "camera_api.h"

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
#include <linux/fb.h>
#include <sys/poll.h>
#include <linux/i2c-dev.h>
#include <linux/videodev2.h>
#include <time.h>

int cam_fd = -1;
struct buffer *buffers;
static int nbuffers;

int wait_camera_ready(int id)
{
    fd_set fds;
    struct timeval tv;
    int r;

    FD_ZERO(&fds);
    FD_SET(cam_fd, &fds);

    /* Timeout */
    tv.tv_sec  = 2;
    tv.tv_usec = 0;

    r = select(cam_fd + 1, &fds, NULL, NULL, &tv);
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

    return 0;
}

int camera_init(int id, int width, int height, int fps, int frameNums)
{
    char dev_name[32];
    int i;
    int fd = -1;
    int iReadyCount = 5;
    struct v4l2_input inp;           /* select the current video input */
    struct v4l2_format fmt;
    struct v4l2_streamparm parms;    /* set streaming parameters */
    struct v4l2_requestbuffers req;
    enum v4l2_buf_type type;
    if (cam_fd > -1)
        return 0;

    sprintf(dev_name, "/dev/video%d", id);
    fd = open(dev_name, O_RDWR);
    if(fd == -1)
    {
       printf("can't open %s(%s)\n", dev_name, strerror(errno));
       goto open_err;
    }

    inp.index = 0;
    inp.type = V4L2_INPUT_TYPE_CAMERA;
    if(ioctl(fd, VIDIOC_S_INPUT, &inp) < 0)
    {
        printf("VIDIOC_S_INPUT error\n");
        goto err;
    }

    memset(&parms, 0, sizeof(struct v4l2_streamparm));
    parms.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parms.parm.capture.timeperframe.numerator = 1;
    parms.parm.capture.timeperframe.denominator = fps;
    if(ioctl(fd,VIDIOC_S_PARM,&parms) < 0)
    {
        printf("VIDIOC_S_PARM error\n");
        goto err;
    }

    /* set image format */
    CLEAR (fmt);
    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = width;
    fmt.fmt.pix.height      = height;
    //fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_SBGGR8;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_NV21;
    fmt.fmt.pix.field       = V4L2_FIELD_NONE;

    if (ioctl(fd, VIDIOC_S_FMT, &fmt) < 0)
    {
        printf("set image format failed\n");
        goto err;
    }

    while(ioctl(fd, VIDIOC_G_FMT, &fmt) && iReadyCount){
        iReadyCount --;
        if(iReadyCount == 0)
        {
            printf("get image format failed\n");
            goto err;
        }
        usleep(5000);
    }

    memset(&req, 0, sizeof(struct v4l2_requestbuffers));
    req.count = frameNums;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if(ioctl(fd, VIDIOC_REQBUFS, &req) < 0)
    {
        printf(" VIDIOC_REQBUFS failed\n");
        goto err;
    }

    buffers = (struct buffer*)calloc(req.count, sizeof(struct buffer));
    for (nbuffers = 0; nbuffers < req.count; nbuffers++)
    {
        struct v4l2_buffer buf;

        memset(&buf, 0, sizeof(struct v4l2_buffer));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = nbuffers;

        if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1)
        {
            printf("VIDIOC_QUERYBUF error\n");
            goto buffer_rel;
        }

        buffers[nbuffers].start  = mmap(NULL, buf.length,
                PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
        buffers[nbuffers].length = buf.length;
        if (buffers[nbuffers].start == MAP_FAILED)
        {
            printf("mmap failed\n");
            goto buffer_rel;
        }
    }

    for (i = 0; i < nbuffers; i++)
    {
        struct v4l2_buffer buf;

        memset(&buf, 0, sizeof(struct v4l2_buffer));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        if (ioctl(fd, VIDIOC_QBUF, &buf) == -1)
        {
            printf("VIDIOC_QBUF error\n");
            goto unmap;
        }
    }

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) == -1)
    {
        printf("VIDIOC_STREAMON error\n");
        goto unmap;
    }

    cam_fd = fd;
    return 0;
unmap:
    for (i = 0; i < nbuffers; i++)
        munmap(buffers[i].start, buffers[i].length);
buffer_rel:
    free(buffers);
err:
    close(fd);
open_err:
    cam_fd = -1;
    return -1;
}

int camera_release (int id)
{
    int i;
    enum v4l2_buf_type type;

    if (cam_fd == -1)
        return -1;

    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(cam_fd, VIDIOC_STREAMOFF, &type);

    for (i = 0; i < nbuffers; i++)
        munmap(buffers[i].start, buffers[i].length);

    free(buffers);

    if (close (cam_fd) < 0)
        perror ("close");

    cam_fd = -1;
    return 0;
}
