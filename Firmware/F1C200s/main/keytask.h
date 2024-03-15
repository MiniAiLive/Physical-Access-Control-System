#ifndef KEYTASK_H
#define KEYTASK_H

#include "thread.h"
#include "mutex.h"
#include "settings.h"

#define MAX_KEY_NUM 20

#define KEY_CLICKED 0
#define KEY_LONG_PRESS 1
#define KEY_LONG_PRESS_MAX 2

class KeyTask : public Thread
{
public:
    KeyTask();
    ~KeyTask();

    void    Start();
    void    Stop();
    void    Exit();

    void    ResetKey();
    void    AddKey(int iKeyID, int iLongFlag, int iLongTime = 9 * 1000);
    void    RemoveKey(int iKeyID);

    int     GetCounter() {return m_iCounter;}
protected:
    void    run();

    int     m_iRunning;
    int     m_iCounter;

    Mutex   m_xMutex;
    int     m_aiKey[MAX_KEY_NUM];
    int     m_aiLongFlag[MAX_KEY_NUM];
    int     m_aiLongTime[MAX_KEY_NUM];
    int     m_iKeyCount;

    Mutex   m_xBackKeyMutex;
    int     m_aiBackKeyState[3];
};

#endif // KEYTASK_H
