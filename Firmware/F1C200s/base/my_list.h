#ifndef LIST_H
#define LIST_H

#include "appdef.h"
#include "shared.h"

using namespace std;
#include <list>

class MyObject
{
public:
    MyObject(MyObject* parent = 0);
    virtual ~MyObject() {}
    virtual MY_WCHAR* TR(const char* _text, MY_WCHAR* out);
};

class MyPoint;

/*
* use this class for Rectangle
*/
class Rect
{
public:
    Rect();
    Rect(int x, int y, int w, int h);
    void SetPos(int x, int y);
    void SetSize(int w, int h);
    void SetRect(int x, int y, int w, int h);
    void Move(int xoff, int yoff);
    int GetWidth();
    int GetHeight();
    int GetX();
    int GetY();
    int Contains(MyPoint pt);
    int IntersectWithRect(Rect rt);
private:
    int mX;
    int mY;
    int mWidth;
    int mHeight;
};

class Margin
{
public:
    Margin(): mLeft(0), mRight(0), mTop(0), mBottom(0) {}
    void SetLeft(int _left) { mLeft = _left; }
    void SetRight(int _right) { mRight = _right; }
    void SetTop(int _top) { mTop = _top; }
    void SetBottom(int _bottom) { mBottom = _bottom; }
    int GetLeft() { return mLeft; }
    int GetRight() { return mRight; }
    int GetTop() { return mTop; }
    int GetBottom() { return mBottom; }
    void SetMargin(int n) { mLeft = mRight = mTop = mBottom = n; }
    void SetMargins(int _left, int _right, int _top, int _bottom)
    {
        mLeft = _left;
        mRight = _right;
        mTop = _top;
        mBottom = _bottom;
    }

private:
    int mLeft;
    int mRight;
    int mTop;
    int mBottom;
};

class MyPoint
{
public:
    MyPoint();
    MyPoint(int x, int y);
    void SetX(int x);
    void SetY(int y);
    int GetX();
    int GetY();
    void SetPoint(int x, int y);
    void SetWidth(int w) { SetX(w); }
    void SetHeight(int h) { SetY(h); }
    int GetWidth() { return mX; }
    int GetHeight() { return mY; }
    void Move(int xoff, int yoff);
private:
    int mX;
    int mY;
};

typedef MyPoint Size;

#if 0
class List
{
public:
    List();
    int Append(void*);
    void RemoveTail();
    void RemoveAt(int);
    void Remove(void* a);
    int Count();
    void Clear(int _free = 0);
    int IsEmpty();
    List *GetNext();
    List *GetPrev();
    List *GetHead();
    List *GetTail();
    void RemoveHead();
    void* GetData();
private:
    List *mHead;
    List *mTail;
    List *mNext;
    List *mPrev;
    int mCount;
    void* mData;
};
#endif 

#endif // LIST_H
