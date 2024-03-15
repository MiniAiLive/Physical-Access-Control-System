#include "my_list.h"
#include "my_lang.h"

#include <string.h>
#include <ctype.h>
#include <stdio.h>

using namespace std;
#include <typeinfo>


Rect::Rect()
{
    mX = 0;
    mY = 0;
    mWidth = 0;
    mHeight = 0;
}

Rect::Rect(int x, int y, int w, int h)
{
    SetRect(x, y, w, h);
}

void Rect::SetPos(int x, int y)
{
    mX = x;
    mY = y;
}

void Rect::SetSize(int w, int h)
{
    mWidth = w;
    mHeight = h;
}

void Rect::SetRect(int x, int y, int w, int h)
{
    mWidth = w;
    mHeight = h;
    mX = x;
    mY = y;
}

void Rect::Move(int xoff, int yoff)
{
    mX += xoff;
    mY += yoff;
}

int Rect::GetHeight()
{
    return mHeight;
}

int Rect::GetWidth()
{
    return mWidth;
}

int Rect::GetX()
{
    return mX;
}

int Rect::GetY()
{
    return mY;
}

/*
* returns true if given pt is in rect area.
*/

int Rect::Contains(MyPoint pt)
{
    if (pt.GetX() < mX || pt.GetY() < mY)
        return 0;
    if (mX + mWidth <= pt.GetX() || mY + mHeight <= pt.GetY())
        return 0;
    return 1;
}

/*
* returns true if given rt intersects with the rect.
*/

int Rect::IntersectWithRect(Rect rt)
{
    int minx, minxw, maxx;
    int miny, maxy, minyh;
    if (mX > rt.mX)
    {
        minx = rt.mX;
        minxw = rt.mWidth;
        maxx = mX;
    }
    else
    {
        minx = mX;
        minxw = mWidth;
        maxx = rt.mX;
    }
    if (mY > rt.mY)
    {
        miny = rt.mY;
        minyh = rt.mHeight;
        maxy = mY;
    }
    else
    {
        miny = mY;
        minyh = mHeight;
        maxy = rt.mY;
    }
    if ( (maxx - minx < minxw) && (maxy - miny < minyh) )
        return 1;
    return 0;
}

MyPoint::MyPoint()
{
    mX = 0;
    mY = 0;
}

MyPoint::MyPoint(int x, int y)
{
    mX = x;
    mY = y;
}

int MyPoint::GetX()
{
    return mX;
}

int MyPoint::GetY()
{
    return mY;
}

void MyPoint::SetX(int x)
{
    mX = x;
}

void MyPoint::SetY(int y)
{
    mY = y;
}

void MyPoint::SetPoint(int x, int y)
{
    mX = x;
    mY = y;
}

void MyPoint::Move(int xoff, int yoff)
{
    mX += xoff;
    mY += yoff;
}

////////////////////////////////////////////////////
/// \brief List::List
///
////////////////////////////////////////////////////

#if 0
List::List()
{
    mHead = 0;
    mTail = 0;
    mNext = 0;
    mPrev = 0;
    mCount = 0;
}

int List::Append(void* a)
{
    List *tmp;
    tmp = new List();
    if (!tmp)
        return 1;
    if (mTail)
    {
        mTail->mNext = tmp;
        tmp->mPrev = mTail;
        mTail = tmp;
    }
    else
    {
        mTail = mHead = tmp;
    }
    tmp->mData = a;
    mCount ++;
    return 0;
}

int List::Count()
{
    return mCount;
}

/*
* ret:
    0: ok
    1: fail, not found
*/
void List::RemoveAt(int idx)
{
    int i;
    List *itr = mHead;
    List *prev = 0;
    if (idx < 0)
        return;
    for (i = 0; i < idx; i ++)
    {
        if (!itr)
            return;
        prev = itr;
        itr = itr->mNext;
    }
    if (prev)
    {
        prev->mNext = itr->mNext;
        itr->mNext->mPrev = prev;
    }
    else
    {
        if (mHead->mNext)
        {
            mHead->mNext->mPrev = 0;
            mHead = mHead->mNext;
        }
        else
            mHead = mTail = mPrev = 0;
    }
    delete itr;
    mCount --;
}

void List::RemoveTail()
{
    if (mCount == 0)
        return;
    RemoveAt(mCount - 1);
}

void List::Remove(void* a)
{
    List *itr = mHead;
    List *prev = 0;
    while (itr)
    {
        if (itr->GetData() == a)
            break;
        prev = itr;
        itr = itr->mNext;
    }
    if (!itr)
    {
        //not found
        return;
    }
    if (prev)
    {
        prev->mNext = itr->mNext;
        
        if (itr == mTail)
        {
            mTail = prev;
        }
        else
            itr->mNext->mPrev = prev;
            
    }
    else
    {
        if (mHead->mNext)
        {
            mHead->mNext->mPrev = 0;
            mHead = mHead->mNext;
        }
            
        else
            mHead = mTail = mPrev = 0;
    }

    delete itr;
    mCount --;
}

void List::Clear(int _free)
{
    List *itr = mHead;
    List *prev = 0;
    while (itr)
    {
        prev = itr;
        itr = itr->mNext;
        if (prev)
        {
            if (_free)
                delete prev->GetData();
            delete prev;
        }
    }
    mHead = 0;
    mTail = 0;
    mCount = 0;
    mNext = 0;
    mPrev = 0;
}

List* List::GetHead()
{
    return mHead;
}

List* List::GetNext()
{
    return mNext;
}

List* List::GetPrev()
{
    return mPrev;
}

List* List::GetTail()
{
    return mTail;
}

void List::RemoveHead()
{
    RemoveAt(0);
}

int List::IsEmpty()
{
    return (mHead == 0);
}

void* List::GetData()
{
    return mData;
}

#endif /* 0 */

MY_WCHAR* MyObject::TR(const char* _text, MY_WCHAR* out)
{
    char str1[STR_MAX_LEN];
    char *str = str1;
    int i;
    sprintf(str, "%s_", typeid(*this).name());
    if (*str >= '0' && *str <= '9')
    {
        while(*str && (*str >= '0' && *str <= '9'))
        {
            str++;
        }
    }
    else
    {
        for (i = 0; i <= (int)strlen(str) - 6; i++)
        {
            str[i] = str[i + 6];
        }
    }
    strcat(str, _text);
    MY_TR(str, out);
    for (i = 0; i < (int)strlen(str); i++)
    {
        if ( ((MY_WCHAR)str[i]) != out[i])
            break;
    }
    if (i >= (int)strlen(str))
        MY_TR(_text, out);
    return out;

}

MyObject::MyObject(MyObject* /*parent*/)
{

}
