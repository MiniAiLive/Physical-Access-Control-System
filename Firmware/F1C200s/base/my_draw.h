#ifndef DRAW_H
#define DRAW_H

#include "appdef.h"
#include "mutex.h"
#include "thread.h"
#include "my_list.h"

#ifdef MY_LINUX
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <fcntl.h>

#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <linux/fb.h>
#endif /* MY_LINUX */

void SetScreenInfo(void*);

class Pen
{
public:
    enum PenStyle
    {
        PS_None,
        PS_Solid,
        PS_Dash,
        PS_Dot,
        PS_DashDot,
        PS_End
    };
    Pen(): mColor(0), mWidth(1), mStyle(PS_Solid){}
    Pen(uint color, int width = 1) :mWidth(width), mStyle(PS_Solid) { mColor = color; }
    void SetColor(uint color);
    void SetColor(int r, int g, int b, int a = 0xff);
    uint GetColor();
    void SetStyle(PenStyle ps);
    PenStyle GetStyle();
    void SetWidth(int w);
    int GetWidth();

private:
    uint mColor;
    int mWidth;
    PenStyle mStyle;
};

class Font;

class Canvas: public Thread, public Mutex
{
public:
    enum DeviceColorMode{
        DCM_None,
        DCM_ARGB,
        DCM_RGB565,
        DCM_End
    };

    enum C_Text_Align{
        C_TA_Left = 0x01,
        C_TA_Center = 0x02,
        C_TA_Right = 0x03,
        C_TA_HoriMask = 0x0f,
        C_TA_Top = 0x10,
        C_TA_Middle = 0x20,
        C_TA_Bottom = 0x30,
        C_TA_VertMask = 0xf0
    };

    Canvas(int width, int height, const char* fb_path = 0);
    Canvas(int width, int height, DeviceColorMode mode, unsigned char* frame_buffer);
    //Canvas(Window* wndPainter);
    ~Canvas();

    void DrawStart();
    void DrawEnd();
    void Start();
    void Stop();
    inline void DrawPixel(int x, int y, uint color);
    inline uint ColorAt(int x, int y);
    void DrawFillRect(int x, int y, int width, int height);
    void DrawRect(int x, int y, int w, int h);
    void DrawRectLine(int x, int y, int w, int h);
    void DrawLine(int x1, int y1, int x2, int y2, int border = 1);
    void DrawImage(int x, int y, void* _img,
                   int sx = 0, int sy = 0, int sw = -1, int sh = -1);
    void DrawImage(int destx, int desty, int destw, int desth, void* _img,
        int sx = 0, int sy = 0, int sw = -1, int sh = -1);
    void DrawTextOut(int x, int y, const MY_WCHAR* _text,
        int limit_width = -1,
        int text_align = C_TA_Left | C_TA_Top,
        int limit_height = -1,
        int length = -1, int start = 0);
    void SetForeColor(uint color);
    uint GetForeColor();
    void SetBackColor(uint color);
    uint GetBackColor();
    int DrawTextWidth(const MY_WCHAR* _text, int length = -1, int start = 0);
    int DrawTextHeight(const MY_WCHAR* _text, int length = -1, int start = 0);
    int GetTextHeight(const MY_WCHAR* _text, int length = -1, int start = 0);
//    void SetValidRect(Rect rt);
//    void SetValidRect(int x, int y, int w, int h);

    void* DrawGetImage();
    static Canvas* GetInstance();
    void Sync();
    Pen GetPen();
    void SetPen(Pen p);
    void DrawClear(); //clear buffer

    void SetDeviceColorMode(DeviceColorMode m);
    DeviceColorMode GetDeviceColorMode();
    void DrawFaceRect(int x_pos, int y_pos, int x_size, int y_size, uint color);
    void DrawMemFullImg(unsigned char* pbImg32);
    void DrawMemImgRGB888(unsigned char* pbImg32, int width, int height);
#ifdef MY_LINUX
    int  DispOpen();
    void DispClose();

    int VideoStart();
    void VideoStop();
    int  VideoMap(int iWidth, int iHeight, unsigned int* piAddr);
#endif
protected:
    int DrawInit(const char* fb_path);
    int DrawInit(unsigned char* frame_buffer, DeviceColorMode mode);
    void DrawDestroy();
    void Sync_RGB565();
    void Sync_ARGB();
    void DrawRectLineSub(int x, int y, int w, int h);

    void SyncYuv(unsigned char* data, int width, int height);
    void DrawTextOutSub(int x, int y, const MY_WCHAR* _text,
        int limit_width = -1,
        int text_align = C_TA_Left | C_TA_Top,
        int limit_height = -1,
        int length = -1, int start = 0);

    void    run();
private:
    void* mFrameBuffer; //temporary buffer.
    void* mDeviceBuffer; //real framebuffer.
    uint mForeColor;
    uint mBackColor;

    // canvas font {{
    int mFontSize;
    //Font* mFontEng;
    Font* mFontOther;
    // }}
    //user font
    //Font* mFontUser;

    static Canvas* mInstance;
//    Rect mValidRect; //the area on which we can draw.
//    Rect mCanvasRect; //Canvas's drawing area.
    Pen mPen;
    DeviceColorMode mDevColorMode;
    
    //Window* mCanvasWindow;

    //framebuffer data
#ifdef MY_LINUX
    int fb_init(const char* dev_path);
    void* fb_map(int fd);
    int fb_unmap();
    int draw_rectangle(int x_pos, int y_pos, int x_size, int y_size);
    int fb_fd;
    struct fb_var_screeninfo fb_vinfo;
    struct fb_fix_screeninfo fb_finfo;
    int fb_screensize;
    int disp_fd;
#endif /* MY_LINUX */
public:
    void SetFontSize(int n_size);
    int GetFontSize();
//    void SetEngFont(Font* font);
    void SetOtherFont(Font* font);
//    void SetUserFont(Font* font);
//    Font* GetUserFont();
//    Font* GetEngFont();
    Font* GetOtherFont();
    void MyDrawText(int x, int y,
        const MY_WCHAR* _text, Size *limit_size = 0, Size* boundingSize = 0,
        int length = -1, int start = 0);
    void MyGetTextMetrics(const MY_WCHAR* _text,
        Size* sz, int length = -1, int start = 0);

    void SetScreenSize(int width, int height);
    static void conv422to420Rotate270(unsigned char* src, int iWidth, int iHeight, unsigned char* pbDst);
private:
    int mScreenWidth;
    int mScreenHeight;
public:
    int GetScreenWidth();
    int GetScreenHeight();
};

#endif // DRAW_H
