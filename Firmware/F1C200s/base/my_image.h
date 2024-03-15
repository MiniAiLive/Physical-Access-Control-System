#ifndef MY_IMAGE_H
#define MY_IMAGE_H

#include "png/pngimage.h"

#include <stdio.h>


#define COLOR_TRANSPARENT 0x00
#define COLOR_OPAQUE 0xff
#define TO_COLOR(r, g, b, a) ((b) + ((g) << 8) + ((r) << 16) + ((a) << 24))
#define COLOR_R(c) (((c) & 0x00ff0000) >> 16)
#define COLOR_G(c) (((c) & 0x0000ff00) >> 8)
#define COLOR_B(c) ((c) & 0x000000ff)
#define COLOR_A(c) (((c) & 0xff000000) >> 24)


class Image
{
public:
    enum{
        FMT_BIN,
        FMT_PNG,
        FMT_END
    };
    Image(const char* path = 0);
    ~Image();
    int load(const char* path);
    int height();
    int width();

    void setSize(int width, int height);
    unsigned int pixel(int x, int y);
    int valid();
    unsigned char *bits();
    Image scaled(int width, int height);
    static void smoothImage1(unsigned char* buffer, int width, int height);
    static void smoothImageRGBA(unsigned char* buffer, int width, int height);

private:
    char* pbData;
    int mBpp;//Bits per pixel
    int mWidth;
    int mHeight;
    int mFmt; //Format of Image: FMT_BMP, ...
    PngImage mPng;
};


#endif // MY_IMAGE_H

