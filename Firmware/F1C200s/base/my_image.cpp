#include "my_image.h"

Image::Image(const char *path)
{
    mWidth = mHeight = 0;
    mBpp = 0;
    pbData = 0;
    mFmt = FMT_END;
    if(path)
        load(path);
}

void Image::setSize(int width, int height)
{
    if(pbData)
        delete[] pbData;
    pbData = (char*)new unsigned int[width * height];
    mWidth = width;
    mHeight = height;
    mBpp = 32;
    this->mFmt = FMT_BIN;
}

Image Image::scaled(int width, int height)
{
    int i, j;
    Image img;
    img.setSize(width, height);
    float rw = 1, rh = 1;
    rw = 1.0 * mWidth / width;
    rh = 1.0 * mHeight / height;

    if(height < mHeight && width < mWidth) // zoom out
    {
        for (i = 0; i < width; i++)
        {
            for (j = 0; j < height; j++)
            {
                int k, p;
                unsigned int r = 0, g = 0, b = 0, a = 0;
                for (k = 0; k < rw; k++)
                {
                    for (p = 0; p < rh; p++)
                    {
                        unsigned int color = pixel(i*rw + k, j*rh + p);
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
                ((unsigned int*)img.pbData)[j*width + i] = TO_COLOR(r, b, g, a);
            }
        }
    }
    else //zoom in
    {
        for (i = 0; i < width; i++)
        {
            for (j = 0; j < height; j++)
            {
                ((unsigned int*)img.pbData)[j*width + i] = pixel(i* rw, j*rh);
            }
        }
    }

    if (mWidth != width || mHeight != height)
    {
        smoothImageRGBA((unsigned char*)img.pbData, width, height);
    }
    return img;
}

Image::~Image()
{
    if(pbData)
        delete[] pbData;
}

int Image::valid()
{
    return mFmt < FMT_END && mFmt >= FMT_BIN;
}

int Image::load(const char *path)
{
    int ret;
    ret = mPng.load(path);
    if(ret)
    {
        mFmt = FMT_PNG;
        mWidth = mPng.getWidth();
        mHeight = mPng.getHeight();
        mBpp = 32;
        return ret;
    }
    return ret;
}

void Image::smoothImage1(unsigned char *buffer, int width, int height)
{
    int i, j;
    for(i = 0; i < height - 1; i++)
    {
        for(j = 0; j < width - 1; j++)
        {
            buffer[i* width + j] = (buffer[i*width + j] + buffer[(i + 1)*width + j] + buffer[(i - 1)*width + j] + buffer[i*width + j + 1] + buffer[i*width + (j - 1)]) / 5;
        }
    }
}

void Image::smoothImageRGBA(unsigned char* buffer, int width, int height)
{
    int i, j;
    unsigned int color[5];
    unsigned int r, g, b, a;
    for (i = 0; i < height - 1; i++)
    {
        for (j = 0; j < width - 1; j++)
        {
            color[0] = buffer[i*width + j];
            color[1] = buffer[(i + 1)*width + j];
            color[2] = buffer[(i - 1)*width + j];
            color[3] = buffer[i*width + j + 1];
            color[4] = buffer[i*width + (j - 1)];
            r = g = b = a = 0;
            for (int k = 0; k < 5; k++)
            {
                r += COLOR_R(color[k]);
                g += COLOR_G(color[k]);
                b += COLOR_B(color[k]);
                a += COLOR_A(color[k]);
            }
            r /= 5;
            g /= 5;
            b /= 5;
            a /= 5;
            buffer[i* width + j] = TO_COLOR(r, g, b, a);
        }
    }
}

int Image::width()
{
    return mWidth;
}

int Image::height()
{
    return mHeight;
}

unsigned char* Image::bits()
{
    if(mFmt == FMT_PNG)
    {
        return mPng.bits();
    }
    return NULL;
}

unsigned int Image::pixel(int x, int y)
{
    unsigned int color = 0;
    if(mFmt == FMT_PNG)
    {
        color = mPng.colorAt(x, y);
    }
    else if(mFmt == FMT_BIN)
    {
        if(pbData)
            color = ((unsigned int*)pbData)[y * mWidth + x];
    }

    return color;
}

