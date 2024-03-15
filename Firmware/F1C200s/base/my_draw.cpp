#include "my_draw.h"
//#include "lcdtask.h"

#include "my_font.h"
#include "my_image.h"

#ifdef MY_LINUX
#include <linux/fb.h>
//#include "videodev2.h"
//#include "sun8iw8p_display.h"
//#include "lcdmanager.h"
#include <errno.h>
#endif /* MY_LINUX */

#ifndef MAX
#define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif  //	MAX

#include <math.h>

#define COLOR_TRANSPARENT 0x00
#define COLOR_OPAQUE 0xff
#define TO_COLOR(r, g, b, a) ((b) + ((g) << 8) + ((r) << 16) + ((a) << 24))
#define COLOR_R(c) (((c) & 0x00ff0000) >> 16)
#define COLOR_G(c) (((c) & 0x0000ff00) >> 8)
#define COLOR_B(c) ((c) & 0x000000ff)
#define COLOR_A(c) (((c) & 0xff000000) >> 24)

#define C_ALPHA(a) ((a >> 24) & 0xFF)
#define C_RED(a) ((a >> 16) & 0xFF)
#define C_GREEN(a) ((a >> 8) & 0xFF)
#define C_BLUE(a) ((a) & 0xFF)
#define MAKE_COLOR(a, r, g, b) ((a & 0xFF) << 24 | (r & 0xFF) << 16 | (g & 0xFF) << 8 | (b & 0xFF))

#define TO_RGB565(r,g,b) ((((unsigned char)(r) >> 3) << 11) | (((unsigned char)(g) >> 2) << 5) | ((unsigned char)(b) >> 3))
#define RGB565_R(rgb) (((rgb) & 0xf800) >> 8)
#define RGB565_G(rgb) (((rgb) & 0x07e0) >> 3)
#define RGB565_B(rgb) (((rgb) & 0x001f) << 3)

void* glScreenInfo;

Canvas* Canvas::mInstance = 0;

Canvas* Canvas::GetInstance()
{
    if (!mInstance) {
#ifdef MY_LINUX
        mInstance = new Canvas(0, 0, "/dev/fb0");
#else /* MY_LINUX */
        mInstance = new Canvas();
#endif /* MY_LINUX */
#ifdef BV_FLM_1V7
        mInstance->SetDeviceColorMode(DCM_RGB565);
#else /* BV_FLM_1V7 */
        mInstance->SetDeviceColorMode(DCM_ARGB);
#endif /* BV_FLM_1V7 */
    }
    return mInstance;
}

Canvas::Canvas(int width, int height, const char* fb_path)
    : mScreenWidth(width)
    , mScreenHeight(height)
{
//    mCanvasWindow = NULL;
    mForeColor = TO_COLOR(0, 0, 0, COLOR_OPAQUE);
    mBackColor = TO_COLOR(0, 0, 0, COLOR_OPAQUE);
    mFrameBuffer = 0;
    mDeviceBuffer = 0;
    mDevColorMode = DCM_None;

    mFontSize = 12;
    //mFontUser = NULL;
    //mFontEng = NULL;
    mFontOther = NULL;

//    mValidRect.SetPos(0, 0);
//    mValidRect.SetSize(0, 0);

#ifdef MY_LINUX
    fb_fd = -1;
    memset(&fb_vinfo, 0, sizeof(struct fb_var_screeninfo));
    memset(&fb_finfo, 0, sizeof(struct fb_fix_screeninfo));
    fb_screensize = 0;
#endif /* MY_LINUX */

    //init draw object
    DrawInit(fb_path);
}

Canvas::Canvas(int width, int height, DeviceColorMode mode, unsigned char *frame_buffer)
    : mScreenWidth(width)
    , mScreenHeight(height)
{
//    mCanvasWindow = NULL;
    mForeColor = TO_COLOR(0, 0, 0, COLOR_OPAQUE);
    mBackColor = TO_COLOR(0, 0, 0, COLOR_OPAQUE);
    mFrameBuffer = 0;
    mDeviceBuffer = 0;
    mDevColorMode = DCM_None;

    mFontSize = 12;
    mFontOther = NULL;

#ifdef MY_LINUX
    fb_fd = -1;
    memset(&fb_vinfo, 0, sizeof(struct fb_var_screeninfo));
    memset(&fb_finfo, 0, sizeof(struct fb_fix_screeninfo));
    fb_screensize = 0;
#endif /* MY_LINUX */

    //init draw object
    DrawInit(frame_buffer, mode);
}

Canvas::~Canvas()
{
    //destroy draw object
    DrawDestroy();
}

int Canvas::DrawInit(const char* fb_path)
{
//    mCanvasRect.SetRect(0, 0, mScreenWidth, mScreenHeight);
    if (fb_path)
    {
        fb_fd = fb_init(fb_path);
        if (fb_fd >= 0)
        {
#ifdef USE_BUFFER2
            mDeviceBuffer = fb_map(fb_fd);
            mFrameBuffer = (void*)new uint[mScreenWidth*mScreenHeight];//[fb_vinfo.xres*fb_vinfo.yres];
#else /* USE_BUFFER2 */
            mFrameBuffer = fb_map(fb_fd);
            mDeviceBuffer = mFrameBuffer;
#endif /* USE_BUFFER2 */
        }
    }
    memset(mFrameBuffer, 0xff, mScreenWidth* mScreenHeight * sizeof(uint));
    return 0;
}

int Canvas::DrawInit(unsigned char* frame_buffer, DeviceColorMode mode)
{
//    mCanvasRect.SetRect(0, 0, mScreenWidth, mScreenHeight);
    SetDeviceColorMode(mode);
    if (frame_buffer)
    {
        mFrameBuffer = (void*)frame_buffer;
        mDeviceBuffer = mFrameBuffer;
    }
    //memset(mFrameBuffer, 0xff, mScreenWidth* mScreenHeight * sizeof(uint));
    return 0;
}

void Canvas::DrawDestroy()
{
    fb_unmap();
#ifdef USE_BUFFER2
    if (mDeviceBuffer != mFrameBuffer)
        delete[](uint*)mFrameBuffer;
#endif //USE_BUFFER2
}

/*
* before draw, you must call this function.
*/

void Canvas::DrawStart()
{
    Lock();
    //mValidRect = mCanvasRect;
}

/*
* after completion of draw, you must call this function.
*/

void Canvas::DrawEnd()
{
    Unlock();
}

void Canvas::Start()
{
    Thread::Start();
}

void Canvas::Stop()
{
    Thread::Wait();
}

uint Canvas::ColorAt(int x, int y)
{
//    if (!mCanvasRect.Contains(MyPoint(x, y)) ||
//        !mValidRect.Contains(MyPoint(x, y)))
//    {
//        return 0;
//    }
    if (x < 0 || x > mScreenWidth)
        return 0;
    if (y < 0 || y > mScreenHeight)
        return 0;
    uint *ptr = (uint*)mFrameBuffer;
    int location = y * mScreenWidth + x;
    return ptr[location];
}

/*
* set color of pixel(x,y).
*/
void Canvas::DrawPixel(int x, int y, uint color)
{
#ifdef USE_DEBUG
    //printf("[%s] (%d,%d)=%08x,\n", __FUNCTION__, x, y, color);
#endif //USE_DEBUG
    if (x < 0 || x > mScreenWidth)
        return;
    if (y < 0 || y > mScreenHeight)
        return;
    uint *ptr = (uint*)mFrameBuffer;
    int location = y * mScreenWidth + x;
    unsigned char r, g, b, a, r1, g1, b1, a1;
    unsigned int o_r, og, ob, oa;
    r1 = COLOR_R(ptr[location]);
    g1 = COLOR_G(ptr[location]);
    b1 = COLOR_B(ptr[location]);
    a1 = COLOR_A(ptr[location]);

    r = COLOR_R(color);
    g = COLOR_G(color);
    b = COLOR_B(color);
    a = COLOR_A(color);

    
    oa = (a * 255 + a1 *(255 - a)) / 255;
    if (oa == 0)
    {
        o_r = og = ob = 0;
    }
    else
    {
        o_r = (r * a + r1 * a1 * (255 - a)/255) / oa;
        og = (g * a + g1 * a1 * (255 - a)/255) / oa;
        ob = (b * a + b1 * a1 * (255 - a)/255) / oa;
    }
    
    /*
    oa = a;
    if (a == 0)
    {
        or = r1;
        og = g1;
        ob = b1;
    }
    else
    {
        or = (r*a + r1*(255 - a) + 128) / 255;
        og = (g*a + g1*(255 - a) + 128) / 255;
        ob = (b*a + b1*(255 - a) + 128) / 255;
    }
    */
    
    ptr[location] = TO_COLOR(o_r,og,ob,oa);
}

void Canvas::DrawFillRect(int x, int y, int width, int height)
{
    int i, j;
    
#if 0
    if (COLOR_A(mBackColor) == 0xff)
    {
        for (i = 0; i < width; i++)
            *((uint*)mFrameBuffer + y * mScreenWidth + i + x) = mBackColor;
        for (j = 1; j < height; j++)
        {
            memcpy((void*)((uint*)mFrameBuffer + (j + y) * mScreenWidth + x),
                (uint*)mFrameBuffer + y * mScreenWidth + x, width * sizeof(uint));
        }
    }
    else
#endif 
    {
        for (i = x; i < x + width; i++)
        {
            for (j = y; j < y + height; j++)
            {
                DrawPixel(i, j, mBackColor);
            }
        }
    }
}

void Canvas::DrawRectLine(int /*x*/, int /*y*/, int /*w*/, int /*h*/)
{
}

void Canvas::DrawRectLineSub(int /*x*/, int /*y*/, int /*w*/, int /*h*/)
{
}


void Canvas::DrawRect(int x, int y, int w, int h)
{
    int w1 = w, h1 = h;
    if (w > 2)
        w1 = w - 2;
    if (h > 2)
        h1 = h - 2;
    DrawFillRect(x+1, y+1, w1, h1);
    int i,j;
    for (j = 0; j < mPen.GetWidth(); j ++)
    {
        for (i = 0; i < w; i ++)
        {
            //top border
            DrawPixel(x+i, y+j, mPen.GetColor());
            //bottom border
            DrawPixel(x+i, y+h-j-1, mPen.GetColor());
        }
        for (i = 0; i < h; i ++)
        {
            //top border
            DrawPixel(x+j, y+i, mPen.GetColor());
            //bottom border
            DrawPixel(x+w-j-1, y+i, mPen.GetColor());
        }
    }
}

void Canvas::DrawLine(int x1, int y1, int x2, int y2, int /*border*/)
{
    int style_step = 1;
    if (mPen.GetStyle() == Pen::PS_Dot)
        style_step = 2;
    if (y1 == y2)
    {
        for (; x1 <= x2; x1+= style_step)
            DrawPixel(x1, y1, mPen.GetColor());
    }
    else if (x2 == x1)
    {
        if (y1 > y2)
        {
            //swap x1 and x2.
            y2 = y2 + y1;
            y1 = y2 - y1;
        }
        for (; y1 <= y2; y1+= style_step)
            DrawPixel(x1, y1, mPen.GetColor());
    }
    else
    {
        float a;
        float step = 0;
        float i;
        a = 1.0*(y2- y1)/(x2 - x1);
        step = fabs(a);
        if (step >1)
            step = 1.0/step;
        if (x1 > x2)
        {
            //swap x1 and x2.
            x2 = x2 + x1;
            x1 = x2 - x1;
        }
        if (mPen.GetStyle() == Pen::PS_Dot)
        {
            for (i = x1; i <= x2; i += step*2)
            {
                DrawPixel(i, i*a, mPen.GetColor());
            }
        }
        else
        {
            for (i = x1; i <= x2; i += step)
            {
                DrawPixel(i, i*a, mPen.GetColor());
            }
        }
    }
}

void Canvas::DrawImage(int destx, int desty, int destw, int desth, void* _img,
    int sx, int sy, int sw, int sh)
{
    int i, j;
    Image* img = (Image*)_img;
    float rw = 1, rh = 1;
    if (sh == -1)
        sh = img->height();
    if (sw == -1)
        sw = img->width();
    rw = 1.0 * sw / destw;
    rh = 1.0 * sh / desth;

    if (desth < sh && destw < sw) //zoom out
    {
        for (i = destx; i < destx + destw; i++)
        {
            for (j = desty; j < desty + desth; j++)
            {
                int k, p;
                unsigned int r = 0, g = 0, b = 0, a = 0;
                for (k = 0; k < rw; k++)
                {
                    for (p = 0; p < rh; p++)
                    {
                        unsigned int color = img->pixel(sx + (i - destx)*rw + k, sy + (j - desty)*rh + p);
                        r += COLOR_R(color);
                        g += COLOR_G(color);
                        b += COLOR_B(color);
                        a += COLOR_A(color);
                    }
                }
                r = r / (rw * rh);
                g = g / (rw * rh);
                b = b / (rw * rh);
                a = a / (rw * rh);
                DrawPixel(i, j, TO_COLOR(r, b, g, a));
            }
        }
    }
    else //zoom in
    {
        for (i = destx; i < destx + destw; i++)
        {
            for (j = desty; j < desty + desth; j++)
            {
                unsigned int color = img->pixel(sx + (i - destx) * rw, sy + (j - desty)*rh);
                DrawPixel(i, j, color);
            }
        }
    }
}

///*
//* draw image data on canvas.
//*/
void Canvas::DrawImage(int x, int y, void* _img,
                       int sx, int sy, int sw, int sh)
{
    int i, j;
    Image* img = (Image*)_img;
    if (sh == -1)
        sh = img->height();
    if (sw == -1)
        sw = img->width();
    printf("[DrawImage] sh : %d   sw: %d\n", sh, sw);
    //paint.drawImage(x, y, *img, sx, sy, sw, sh);
    for (i = sx; i < sx + sw; i ++)
    {
        for (j = sy; j < sy + sh; j ++)
        {
            DrawPixel(x + i - sx, y + j - sy, img->pixel(i-sx, j-sy));
        }
    }
}

void Canvas::DrawTextOutSub(int x, int y,
    const MY_WCHAR* _text, int limit_width,
    int text_align,
    int limit_height,
    int length, int start)
{
    int ww = 0, hh = 0;
    //printf("[%s] x=%d,y=%d, len=%d,\n", __FUNCTION__, x, y, wcslen(_text));
    if (limit_width > 0 || limit_height > 0)
    {
        int w = DrawTextWidth(_text, length, start);
        ww = limit_width;
        switch (text_align & C_TA_HoriMask)
        {
        case C_TA_Left:
            break;
        case C_TA_Center:
            if (w < limit_width)
                x += (limit_width - w) / 2;
            break;
        case C_TA_Right:
            if (w < limit_width)
                x += limit_width - w;
            break;
        }

        int h = DrawTextHeight(_text, length, start);
        if (limit_height > 0)
        {
            hh = limit_height;
            switch (text_align & C_TA_VertMask)
            {
            case C_TA_Top:
                break;
            case C_TA_Middle:
                if (h < limit_height)
                    y += (limit_height - h) / 2;
                break;
            case C_TA_Bottom:
                if (h < limit_height)
                    y += limit_height - h;
                break;
            }
        }

        Size sz(ww, hh);
        MyDrawText(x, y, _text, &sz, 0, length, start);
    }
    else
        MyDrawText(x, y, _text, 0, 0, length, start);
}

void Canvas::DrawTextOut(int x, int y,
    const MY_WCHAR* _text, int limit_width,
    int text_align,
    int limit_height,
    int length, int start)
{
    int pos = 0;
    int i;
    int len = wcslen(_text);
    if (length < 0)
        length = len;
    if (start < 0)
        start = 0;
    length = start + length > len ? len - start: length;
    MY_WCHAR swTemp[STR_MAX_LEN];
    for (i = start; i < start + length; i++)
    {
        if (_text[i] == 0x0d || _text[i] == 0x0a)
        {
            swTemp[pos] = 0;
            DrawTextOutSub(x, y, swTemp, limit_width, text_align, limit_height);
            y += DrawTextHeight(swTemp);
            pos = 0;
        }
        else
            swTemp[pos++] = _text[i];
    }
    swTemp[pos] = 0;
    DrawTextOutSub(x, y, swTemp, limit_width, text_align, limit_height);
}

int Canvas::GetTextHeight(const MY_WCHAR* _text, int length, int start)
{
    int pos = 0;
    int i;
    int len = wcslen(_text);
    int iTextHeight = 0;

    if (length < 0)
        length = len;
    if (start < 0)
        start = 0;

    length = start + length > len ? len - start: length;
    MY_WCHAR swTemp[STR_MAX_LEN] = { 0 };

    iTextHeight += DrawTextHeight(swTemp);
    return iTextHeight;
}

int Canvas::DrawTextWidth(const MY_WCHAR* _text,
    int length, int start)
{
    Size sz;
    MyGetTextMetrics(_text, &sz, length, start);
    return sz.GetWidth();
}

int Canvas::DrawTextHeight(const MY_WCHAR* _text,
    int length, int start)
{
    Size sz;
    MyGetTextMetrics(_text, &sz, length, start);
    return sz.GetHeight();
}

#ifdef USE_DEBUG_TRACE
static int dchar_count = 0;
#endif /* USE_DEBUG_TRACE */
void Canvas::MyDrawText( int x, int y,
    const MY_WCHAR *_text,
    Size* limit_size,
    Size* boundingSize,
    int length, int start)
{
    int i;
    int len = (int)wcslen(_text);
    int w, h;
    int old_width = 0;
    int destx,desty;
    int max_width = 0;
    int cur_height;
    int max_height = 0;
    int max_h = 0;

    Font* font;
    uint color = GetForeColor();
    unsigned char r, g, b, a;
    r = COLOR_R(color);
    g = COLOR_G(color);
    b = COLOR_B(color);
    FT_GlyphSlot* glyph;

    if (start < 0)
        start = 0;
    if (start > len - 1)
        start = len -1;
    if (length > 0 && start + length <= len)
        len = length;
    else
        len = len - start;
    for (i = start; i < start + len; i ++)
    {
        uint idx;
        font = mFontOther;
        idx = _text[i];
        //new line character
        if (idx == 0x0D || idx == 0x0A)
        {
            if (max_width < old_width)
                max_width = old_width;
            old_width = 0;
            if(max_h > 0)
            {
                y = y + max_h;
                max_height += max_h;
            }
            else
            {
                y = y + GetFontSize()*2;
                max_height += GetFontSize()*2;
            }
            continue;
        }
        if (idx == 32) //space
        {
            old_width += GetFontSize() / 3;
            continue;
        }
        //draw a character.
//        if (idx < 255) //ascii character?
//        {
//            font = mFontEng;
//        }

        glyph = font->RenderGlyph(idx);
        if (!glyph) //failed to load glyph bitmap, skip this character.
        {
            old_width += GetFontSize();
            continue;
        }

        {
#ifdef USE_DEBUG_TRACE
            my_debug("[%s]going to draw char:%08X, (*glyph)->bitmap.rows=%d, (*glyph)->bitmap.width,\n",
                     __FUNCTION__, _text[i], (int)(*glyph)->bitmap.rows, (int)(*glyph)->bitmap.width);
#endif /* USE_DEBUG_TRACE */
            for (h = 0; h < (int)(*glyph)->bitmap.rows; h++)
            {
                for (w = 0; w < (int)(*glyph)->bitmap.width; w++)
                {
                    a = (*glyph)->bitmap.buffer[h*(*glyph)->bitmap.pitch + w];
                    destx = x + old_width + TRUNC((*glyph)->metrics.horiBearingX) + w;
                    if (max_h < GetFontSize() + TRUNC((*glyph)->metrics.height) - TRUNC((*glyph)->metrics.horiBearingY))
                        max_h = GetFontSize() + TRUNC((*glyph)->metrics.height) - TRUNC((*glyph)->metrics.horiBearingY);
                    desty = y + GetFontSize() - TRUNC((*glyph)->metrics.horiBearingY) + h;
                    if (!boundingSize)
                        DrawPixel(destx, desty, TO_COLOR(r, g, b, a));
                }
            }
        }
        old_width += TRUNC((*glyph)->metrics.horiAdvance);
        cur_height = max_h;
    }
    max_height += max_h;
    //printf("[%s] h=%d, w=%d,\n", __FUNCTION__, max_height, max_width);
    if (boundingSize)
    {
        boundingSize->SetWidth(max_width > old_width? max_width : old_width);
        boundingSize->SetHeight(max_height);
    }
}

void Canvas::MyGetTextMetrics(const MY_WCHAR *_text,
    Size* sz, int length, int start)
{
    MyDrawText(0, 0, _text, 0, sz, length, start);
}


void Canvas::SetForeColor(uint color)
{
    mForeColor = color;
}

uint Canvas::GetForeColor()
{
    return mForeColor;
}

void Canvas::SetBackColor(uint color)
{
    mBackColor = color;
}

uint Canvas::GetBackColor()
{
    return mBackColor;
}

//void Canvas::SetValidRect(int x, int y, int w, int h)
//{
//    mValidRect.SetPos(x, y);
//    mValidRect.SetSize(w, h);
//}

//void Canvas::SetValidRect(Rect rt)
//{
//    mValidRect = rt;
//}

void Canvas::SetDeviceColorMode(DeviceColorMode m)
{
    mDevColorMode = m;
}

Canvas::DeviceColorMode Canvas::GetDeviceColorMode()
{
    return mDevColorMode;
}

void Canvas::Sync_ARGB()
{
#ifdef MY_LINUX
#ifdef USE_BUFFER2

    int size = sizeof(uint) * mScreenWidth * mScreenHeight;

    memcpy(mDeviceBuffer, mFrameBuffer, size);
#endif /* USE_BUFFER2 */
#endif /* MY_LINUX */
}

void Canvas::Sync_RGB565()
{
#ifdef MY_LINUX
#ifdef USE_BUFFER2

    uint *src = (uint*)mFrameBuffer;
    unsigned short *dest = (unsigned short*)mDeviceBuffer;
    unsigned char r,g,b;
    int location;
    for (int x = 0; x < mScreenWidth; x++)
        for (int y = 0; y < mScreenHeight; y++)
        {
            location = y * fb_vinfo.xres + x;
            r = COLOR_R(src[y * mScreenWidth + x]);
            g = COLOR_G(src[y * mScreenWidth + x]);
            b = COLOR_B(src[y * mScreenWidth + x]);

            dest[location] = TO_RGB565(r,g,b);
        }

#endif /* USE_BUFFER2 */
#endif /* MY_LINUX */
}

//pass buffer to device.
void Canvas::Sync()
{
//    my_debug("[%s] rect(%d,%d,%d,%d)\n", __FUNCTION__,
//             evt->GetRect()->GetX(), evt->GetRect()->GetY(),
//             evt->GetRect()->GetWidth(), evt->GetRect()->GetHeight());
#if defined(USE_QT) && !defined(MY_LINUX)
    uint *src = (uint*)mFrameBuffer;
    QImage *dest = (QImage*)mDeviceBuffer;
    for (int x = 0; x < mScreenWidth; x++)
        for (int y = 0; y < mScreenHeight; y++)
        {
            dest->setPixel(x, y, src[y * mScreenWidth + x]);
            //dest->setPixel(x, y, 0xff0000ff);
        }
    if (glScreenInfo)
    {
        Widget* w = (Widget*)glScreenInfo;
        //qDebug()<<"emit paint event";
        w->emitRepaint(evt);
    }
#elif defined(MY_LINUX)

    if (mDeviceBuffer == mFrameBuffer)
        return;
    if (mDevColorMode == DCM_RGB565)
        Sync_RGB565();
    else if (mDevColorMode == DCM_ARGB)
        Sync_ARGB();

#endif /* USE_QT */
}

void Canvas::SetPen(Pen p)
{
    mPen = p;
}

Pen Canvas::GetPen()
{
    return mPen;
}

#ifdef MY_LINUX

int Canvas::fb_init(const char *dev_path)
{
    int fd;
    fd = open(dev_path, O_RDWR);
    if(fd < 0) {
        printf("failed to open %s\n", dev_path);
    }
    else
    {
        my_debug("%s open ok.\n", dev_path);
    }
    return fd;
}

void* Canvas::fb_map(int fd)
{
    char* fbp;
    // Get fixed screen information
    if (ioctl(fd, FBIOGET_FSCREENINFO, &fb_finfo) == -1) {
        my_debug("%s", "Error reading fixed information.\n");
    }

    // Get variable screen information
    if (ioctl(fd, FBIOGET_VSCREENINFO, &fb_vinfo) == -1) {
        my_debug("%s", "Error reading variable information.\n");
    }

    fb_screensize = fb_vinfo.xres * fb_vinfo.yres * fb_vinfo.bits_per_pixel / 8;

    fbp = (char*)mmap(NULL, fb_screensize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

//    my_debug("clear buffer.\n");
    memset(fbp, 0, fb_screensize); //modify
    //memset(fbp, 0xff000000, fb_screensize); //modify

    my_debug("fb map ok=%p, screensize=%d, xres=%d, yres=%d"
             ", activate=%08X, sync=%08X, vmode=%08X\n", fbp,
             fb_screensize, fb_vinfo.xres, fb_vinfo.yres,
             fb_vinfo.activate, fb_vinfo.sync, fb_vinfo.vmode);
    mScreenWidth = fb_vinfo.xres;
    mScreenHeight = fb_vinfo.yres;

    return fbp;
}

int Canvas::fb_unmap()
{
//    my_debug("\t\tfb_unmap\n");

    if(fb_fd >= 0)
    {
        if(mDeviceBuffer)
        {
    //        my_debug("\t\tfb_unmap1\n");
            munmap(mDeviceBuffer, fb_screensize);
            mDeviceBuffer = NULL;
        }
//        my_debug("\t\tfb_close\n");
        close(fb_fd);
    }
    return 0;
}

int Canvas::draw_rectangle(int x_pos, int y_pos, int x_size, int y_size)
{
    int i, j;
    long location;
    uint* lcdfbptr = (uint*)mFrameBuffer;

    for (i = 0; i <= x_size; i++) {

        for (j = 0; j < 8; j++) {
            //left top corner
            location = (y_pos + j) * fb_vinfo.xres + (x_pos + i);
            lcdfbptr[location] = 0xffffff00;

            location = (y_pos + i) * fb_vinfo.xres + (x_pos + j);
            lcdfbptr[location] = 0xffffff00;

            //right bottom corner
            location = (y_pos + y_size - j) * fb_vinfo.xres + (x_pos + x_size - i);
            lcdfbptr[location] = 0xffffff00;

            location = (y_pos + y_size - i) * fb_vinfo.xres + (x_pos + x_size - j);
            lcdfbptr[location] = 0xffffff00;
        }
    }
    return 0;
}

#endif /* MY_LINUX */

void Canvas::SetFontSize(int n_size)
{
    if (n_size > 0)
    {
        mFontSize = n_size;
//        if (mFontEng)
//            mFontEng->SetPointSize(n_size);
        if (mFontOther)
            mFontOther->SetPointSize(n_size);
    }
}


int Canvas::GetFontSize()
{
    return mFontSize;
}

//void Canvas::SetEngFont(Font* font)
//{
//    mFontEng = font;
//}


void Canvas::SetOtherFont(Font* font)
{
    mFontOther = font;
}


//void Canvas::SetUserFont(Font* font)
//{
//    mFontUser = font;
//}

//Font* Canvas::GetUserFont()
//{
//    // TODO: insert return statement here
//    return mFontUser;
//}


//Font* Canvas::GetEngFont()
//{
//    // TODO: insert return statement here
//    return mFontEng;
//}

Font* Canvas::GetOtherFont()
{
    // TODO: insert return statement here
    return mFontOther;
}


void Canvas::SetScreenSize(int width, int height)
{
    if (width > 0)
        mScreenWidth = width;
    if (height > 0)
        mScreenHeight = height;
}


int Canvas::GetScreenWidth()
{
    return mScreenWidth;
}


int Canvas::GetScreenHeight()
{
    return mScreenHeight;
}

void Canvas::DrawMemImgRGB888(unsigned char* pbImg32, int width, int height)
{
    int i,j;
    int k = 0;
    unsigned char r,g,b;
#if 0
    FILE* fp = fopen("/db/888.bin", "wb");
    if(fp)
    {
        fwrite(pbImg32, 1, width*height*3, fp);
        fflush(fp);
        fclose(fp);
    }
#endif
    for (i = 0; i < width; i ++)
    {
        for (j =0; j < height; j ++)
        {
            r = pbImg32[j*width*3 + i*3];
            g = pbImg32[j*width*3 + i*3 + 1];
            b = pbImg32[j*width*3 + i*3 + 2];
            DrawPixel(i, j, TO_COLOR(r, g, b, 0xff));
#if 0
            if (k ++ < 16)
            {
                my_debug("%08x ", TO_COLOR(r, g, b, 0xff));
            }
#endif
        }
    }
    //my_debug("[%s] ended.\n", __FUNCTION__);
}

#ifdef MY_LINUX
//typedef struct _tagtest_layer_info
//{
//    int screen_id;
//    int layer_id;
//    int mem_id;
//    disp_layer_config layer_config;
//    int addr_map;
//    int width, height; //screen size
//    int dispfh; //device node handle
//    int fh; //picture resource file handle
//    int mem;
//    int clear; //is clear layer
//    char filename[32];
//} test_layer_info;

//test_layer_info g_xDispInfo = {0};

int Canvas::DispOpen()
{
    if(disp_fd > 0)
        return -1;

    unsigned int args[6];
    //memset(&g_xDispInfo, 0, sizeof(g_xDispInfo));

    if ((disp_fd = open("/dev/disp", O_RDWR)) == -1)
    {
        printf("can't open /dev/disp(%s)\n", strerror(errno));
        return -1;
    }

//    args[0] = 0;
//    ioctl(disp_fd, DISP_LCD_ENABLE, (void *)args);

    printf("Canvas: Disp = %d\n", disp_fd);

    return disp_fd;
}

void Canvas::DispClose()
{
//    int ret = 0;
//    unsigned int args[6];
//    args[0] = 0;
    //ioctl(disp_fd, DISP_LCD_DISABLE, (void*)args);

    if (disp_fd > 0)
        close(disp_fd);
    disp_fd = -1;
}

int Canvas::VideoMap(int iWidth, int iHeight, unsigned int* piAddr)
{
    unsigned int args[6];
    int ret;

    if(disp_fd == -1)
        return -1;

//    //source frame size
//    g_xDispInfo.layer_config.info.fb.size[0].width = iWidth;
//    g_xDispInfo.layer_config.info.fb.size[0].height = iHeight;
//    g_xDispInfo.layer_config.info.fb.size[1].width = iWidth / 2;
//    g_xDispInfo.layer_config.info.fb.size[1].height = iHeight / 2;

//    // src
//    //test_info.layer_config.info.fb.crop.x     = 0;
//    //test_info.layer_config.info.fb.crop.y     = 0;
//    g_xDispInfo.layer_config.info.fb.crop.width = (unsigned long long)iWidth << 32;
//    g_xDispInfo.layer_config.info.fb.crop.height= (unsigned long long)iHeight << 32;

//    g_xDispInfo.layer_config.info.fb.addr[0] = (*piAddr);
//    g_xDispInfo.layer_config.info.fb.addr[1] = (g_xDispInfo.layer_config.info.fb.addr[0] + iWidth * iHeight);
//    //test_info.layer_config.info.fb.addr[2] = (int)(test_info.layer_config.info.fb.addr[0] + width*height*5/4);
//    g_xDispInfo.layer_config.enable = 1;
//    args[0] = g_xDispInfo.screen_id;
//    args[1] = (int)&g_xDispInfo.layer_config;
//    args[2] = 1;
//    ret = ioctl(disp_fd, DISP_LAYER_SET_CONFIG, (void*)args);

//    if (ret != 0) {
//        printf("disp_set_addr fail to set layer info\n");
//    }
    //printf("[Camera Test] VideoMap is successed\n");
    return ret;
}

int Canvas::VideoStart()
{
//    unsigned int args[6] = { 0 };
//    int iDispWidth = ioctl(disp_fd, DISP_GET_SCN_WIDTH, (void*)args);
//    int iDispHeight = ioctl(disp_fd, DISP_GET_SCN_HEIGHT, (void*)args);

//    g_xDispInfo.screen_id = 0; //0 for lcd ,1 for hdmi, 2 for edp
//    g_xDispInfo.layer_config.channel = 0;
//    g_xDispInfo.layer_config.layer_id = 0;
//    g_xDispInfo.layer_config.info.zorder = 1;
//    //test_info.layer_config.info.ck_enable        = 0;
//    g_xDispInfo.layer_config.info.alpha_mode       = 1; //global alpha
//    g_xDispInfo.layer_config.info.alpha_value      = 0xff;

//    //display window of the screen
//    g_xDispInfo.layer_config.info.screen_win.x = 0;
//    g_xDispInfo.layer_config.info.screen_win.y = 0;
//    g_xDispInfo.layer_config.info.screen_win.width    = iDispWidth;
//    g_xDispInfo.layer_config.info.screen_win.height   = iDispHeight;

//    //mode
//    g_xDispInfo.layer_config.info.mode = LAYER_MODE_BUFFER;

//    //data format
//    g_xDispInfo.layer_config.info.fb.format = DISP_FORMAT_YUV420_SP_UVUV;
//    printf("[LCD Manager] Video start\n");
    return 1;
}

void Canvas::VideoStop()
{
//    unsigned int args[6] = { 0 };

//    g_xDispInfo.layer_config.enable = 0;
//    args[0] = g_xDispInfo.screen_id;
//    args[1] = (int)&g_xDispInfo.layer_config;
//    args[2] = 1;
//    int ret = ioctl(disp_fd, DISP_LAYER_SET_CONFIG, (void*)args);
//    if (ret != 0) {
//        printf("fail to set layer info\n");
//    }

}

void Canvas::DrawFaceRect(int x_pos, int y_pos, int x_size, int y_size, uint color)
{
    int i, j;
    long location;

    x_size -= 1;
    y_size -= 1;
    Pen old_pen;
    old_pen = GetPen();
    mPen.SetColor(color);
    mPen.SetWidth(4);

    for (i = 0; i < x_size / 6; i++)
    {
        for (j = 0; j < 4; j++)
        {
            //left top corner
            if((y_pos + j) < mScreenHeight && (x_pos + i) < mScreenWidth && (x_pos + i) >= 0 && (y_pos + j) >= 0)
            {
                DrawPixel(x_pos + i, y_pos + j, color);
            }

            if((y_pos + i) < mScreenHeight && (x_pos + j) < mScreenWidth && (y_pos + i) >=0 && (x_pos + j) >=0)
            {
                DrawPixel(x_pos + j, y_pos + i, color);
            }

            //rigth top corner
            if((y_pos + j) < mScreenHeight && (x_pos + x_size - i) < mScreenWidth && (y_pos + j) >= 0 && (x_pos + x_size - i) >= 0)
            {
                DrawPixel(x_pos + x_size - i, y_pos + j, color);
            }

            if((y_pos + i) < mScreenHeight && (x_pos + x_size - j) < mScreenWidth && (y_pos + i) >= 0 && (x_pos + x_size - j) >= 0)
            {
                DrawPixel(x_pos + x_size - j, y_pos + i, color);
            }

            //right bottom corner
            if((y_pos + y_size - j) < mScreenHeight && (x_pos + x_size - i) < mScreenWidth && (y_pos + y_size - j) >= 0 && (x_pos + x_size - i) >= 0)
            {
                DrawPixel(x_pos + x_size - i, y_pos + y_size -j, color);
            }

            if((y_pos + y_size - i) < mScreenHeight && (x_pos + x_size - j) < mScreenWidth && (y_pos + y_size - i) >= 0 && (x_pos + x_size - j) >= 0)
            {
                DrawPixel(x_pos + x_size - j, y_pos + y_size - i, color);
            }

            //left bottom corner
            if((y_pos + y_size - j) < mScreenHeight && (x_pos + i) < mScreenWidth && (y_pos + y_size - j) >= 0 && (x_pos + i) >= 0)
            {
                DrawPixel(x_pos + i, y_pos + y_size - j, color);
            }

            if((y_pos + y_size - i) < mScreenHeight && (x_pos + j) < mScreenWidth && (y_pos + y_size - i) >= 0 && (x_pos + j) >= 0)
            {
                DrawPixel(x_pos + j, y_pos + y_size -i, color);
            }
        }
    }
    mPen = old_pen;
}

void Canvas::conv422to420Rotate270(unsigned char* src, int iWidth, int iHeight, unsigned char* pbDst)
{
    int nNewWidth = iHeight;
    int nNewHeight = iWidth;

    unsigned char* dstY1 = pbDst;
    unsigned char* dstY2 = pbDst + 1;
    unsigned char* dstY3 = pbDst + nNewWidth;
    unsigned char* dstY4 = pbDst + nNewWidth + 1;
    unsigned char* dstUV = pbDst + nNewWidth * nNewHeight;

    int h, w, ph1, pw1, ph2, pw2, ph3, pw3, ph4, pw4;
    unsigned char Y1, Y2, Y3, Y4, U, V;

    for (h = 0; h < nNewHeight; h += 2) {

            pw1 = h;
            pw2 = h;
            pw3 = h + 1;
            pw4 = h + 1;

            for (w = 0; w < nNewWidth; w += 2) {

                    ph1 = w;
                    ph2 = w + 1;
                    ph3 = w;
                    ph4 = w + 1;

                    Y1 = src[(pw1 + ph1 * iWidth) * 2];
                    Y2 = src[(pw2 + ph2 * iWidth) * 2];
                    Y3 = src[(pw3 + ph3 * iWidth) * 2];
                    Y4 = src[(pw4 + ph4 * iWidth) * 2];

                    U = src[(pw1 + ph1 * iWidth) * 2 + 1];
                    V = src[(pw1 + ph1 * iWidth) * 2 + 3];

                    *dstY1 = Y1;
                    *dstY2 = Y2;
                    *dstY3 = Y3;
                    *dstY4 = Y4;
                    *dstUV = U;
                    dstUV[1] = V;

                    dstY1 += 2;
                    dstY2 += 2;
                    dstY3 += 2;
                    dstY4 += 2;

                    dstUV += 2;
            }
            dstY1 += nNewWidth;
            dstY2 += nNewWidth;
            dstY3 += nNewWidth;
            dstY4 += nNewWidth;
    }
}

void Pen::SetColor(uint color)
{
    mColor = color;
}
void Pen::SetColor(int r, int g, int b, int a)
{
    mColor = TO_COLOR(r, g, b, a);
}
uint Pen::GetColor()
{
    return mColor;
}
void Pen::SetStyle(PenStyle ps)
{
    mStyle = ps;
}

Pen::PenStyle Pen::GetStyle()
{
    return mStyle;
}
void Pen::SetWidth(int w)
{
    if (w >= 1)
        mWidth = w;
}

int Pen::GetWidth()
{
    return mWidth;
}

unsigned short g_abTmpData[240 * 320] = { 0 };
inline int ConvertYUVtoRGB565(int y, int u, int v, unsigned short* dstData, int index)
{
    y = MAX(0, y - 16);

    int r = (y * 1192 + v * 1634) >> 10;
    int g = (y * 1192 - v * 834 - 400 * u) >> 10;
    int b = (y * 1192 + u * 2066) >> 10;

    r = r > 255? 255 : r < 0 ? 0 : r;
    g = g > 255? 255 : g < 0 ? 0 : g;
    b = b > 255? 255 : b < 0 ? 0 : b;

    dstData[index] = TO_RGB565(r,g,b);
    return 0;
}

void Canvas::SyncYuv(unsigned char* data, int width, int height)
{
    if(mDeviceBuffer == NULL)
        return;

    int size = width * height;
    int offset = size;
    int u, v, y1, y2, y3, y4;

    for(int i = 0, k = offset; i < size; i += 2, k += 2)
    {
        y1 = data[i];
        y2 = data[i + 1];
        y3 = data[width + i];
        y4 = data[width + i + 1];

        v = data[k];
        u = data[k + 1];
        v = v - 128;
        u = u - 128;

        ConvertYUVtoRGB565(y1, u, v, g_abTmpData, i);
        ConvertYUVtoRGB565(y2, u, v, g_abTmpData, i + 1);
        ConvertYUVtoRGB565(y3, u, v, g_abTmpData, width + i);
        ConvertYUVtoRGB565(y4, u, v, g_abTmpData, width + i + 1);

        if (i != 0 && (i + 2) % width == 0)
            i += width;
    }

    memcpy(mDeviceBuffer, g_abTmpData, fb_screensize);
}

void Canvas::run()
{
//    if(mDeviceBuffer == NULL)
//        return;

//    float rOld = Now();
//    int iFlag = 0;
//    while(g_xSS.iRunningCamSurface)
//    {
//        pthread_cond_wait(&g_xBackLCDCond, &g_xBackLCDLock);

//        if(g_xSS.iBackState == 1)
//        {
//            iFlag ++;
//            if(g_xSS.iShowIrCamera == 0)
//            {
//#if 0
//                ConvertYUV420_NV21toRGB888(g_clrYuvData, HEIGHT_240, WIDTH_320, g_clrRgbData1);

//                DrawMemImgRGB888(g_clrRgbData1, HEIGHT_240, WIDTH_320);
//                Sync();
//#else
//                SyncYuv(g_clrYuvData, HEIGHT_240, WIDTH_320);
//#endif
//            }
//            else
//            {
//                ConvertY_NV21toRGB888_Half(g_irOnData, HEIGHT_480, WIDTH_640, g_clrRgbData1);

//                DrawMemImgRGB888(g_clrRgbData1, HEIGHT_240, WIDTH_320);
//                Sync();
//            }

//            if(iFlag > 5)
//                LCDTask::BackDispOn();
//        }
//        else
//        {
//            if(iFlag > 0)
//                memset(mDeviceBuffer, 0, fb_screensize);

//            LCDTask::BackDispOff();
//            iFlag = 0;
//        }

//        rOld = Now();

//        usleep(1 * 1000);
//    }

//    memset(mDeviceBuffer, 0, fb_screensize);
}

#endif // MY_LINUX
