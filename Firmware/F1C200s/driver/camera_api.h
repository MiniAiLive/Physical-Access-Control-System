#ifndef CAMERA_API_H
#define CAMERA_API_H

#include <sys/types.h>

#define CAM_ID      4

#define WIDTH_640   640
#define WIDTH_720   720
#define HEIGHT_480  480

#define FRAME_NUM   4
#define FPS         30

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

#define CLEAR(x) memset (&(x), 0, sizeof (x))

#define ALIGN_4K(x) (((x) + (4095)) & ~(4095))
#define ALIGN_32B(x) (((x) + (31)) & ~(31))
#define ALIGN_16B(x) (((x) + (15)) & ~(15))


#ifdef __cplusplus
extern	"C"
{
#endif

int wait_camera_ready(int id);
int camera_init(int id, int width, int height, int fps, int frameNum);
int camera_release (int id);

extern struct buffer *buffers;
extern int cam_fd;

#ifdef __cplusplus
}
#endif


#endif // CAMERA_API_H
