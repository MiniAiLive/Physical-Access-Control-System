#ifndef CAMERA_API_H
#define CAMERA_API_H

#include <sys/types.h>

#define IR_CAM 0
#define CLR_CAM 1
#define IR_CAM1 2
#define IR_CAM_SUB0     1
#define IR_CAM_SUB1     0

#define WIDTH_1280 1280
#define WIDTH_1080 1080
#define WIDTH_960 960
#define WIDTH_800 800
#define WIDTH_720 720
#define WIDTH_640 640
#define WIDTH_320 320

#define HEIGHT_720 720
#define HEIGHT_600 600
#define HEIGHT_480 480
#define HEIGHT_240 240


#define FRAME_NUM 4
#define MAX_VIDEO_NUM   6

#define NV12_BUF_SIZE(w, h) (((w) * (h) * 3 ) >> 1) // w * h * 3 / 2

struct buffer
{
    void   *start;
    size_t length;
    int     fd;
    void*   buf_org;
};

struct size
{
    int width;
    int height;
};

struct regval_list {
    unsigned char reg_num;
    unsigned char value;
};

#ifndef CLEAR
#define CLEAR(x) memset (&(x), 0, sizeof (x))
#endif

#define ALIGN_4K(x) (((x) + (4095)) & ~(4095))
#define ALIGN_32B(x) (((x) + (31)) & ~(31))
#define ALIGN_16B(x) (((x) + (15)) & ~(15))


#ifdef __cplusplus
extern	"C"
{
#endif

int wait_camera_ready(int id);
int wait_camera_ready_ext(int id, int timeout);
int camera_init(int id, int width, int height, int fps, int frameNum, int rotate, int isp_no_use);
int camera_release (int id);

int camera_set_exp_byreg(int id, int value);
int camera_set_gain_byreg(int id, int value);
int camera_set_regval(int id, unsigned char regaddr, unsigned char regval);
int camera_get_regval(int id, unsigned char regaddr, unsigned char *regval);
int camera_set_irled(int id, int enable, int count);
int camera_set_irled_on(int id, int on);
int camera_clr_set_exp(int value);
int camera_clr_set_gain(int value);
int camera_clr_get_exp();
int camera_clr_get_gain();

//int getExp(int id);
//int getGain(int id);
//int getYAVGValue(int id);

int create_buffer(struct buffer* buf, int size);
int delete_buffer(struct buffer* buf);

int camera_switch(int id, int camid);

extern struct buffer *buffers[2];
extern int cam_fd[2];

#ifdef __cplusplus
}
#endif


#endif // CAMERA_API_H
