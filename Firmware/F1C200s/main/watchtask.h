#ifndef WATCHTASK_H
#define WATCHTASK_H

#include "thread.h"
#include "mutex.h"

enum
{
    WATCH_TYPE_LOW_BATT,
    WATCH_TYPE_TIMEOUT,
    WATCH_TYPE_TIMER,
    WATCH_TYPE_SIREN,
};

#define MAX_TIMER_COUNT 20

class WatchTask : public Thread
{
public:
    WatchTask();
    ~WatchTask();

    void    Start();
    void    Stop();
    int     AddTimer(float iMsec);
    void    RemoveTimer(int iTimerID);
    void    ResetTimer(int iTimerID);
    int     GetCounter(int iTimerID);

protected:
    void    run();

    int     m_iRunning;
    int     m_iIDCounter;

    Mutex   m_xTimerMutex;
    int     m_iTimerCount;
    int     m_aiTimerIDs[MAX_TIMER_COUNT];
    int     m_aiTimerCounter[MAX_TIMER_COUNT];
    float   m_aiTimerMsec[MAX_TIMER_COUNT];
    float   m_arTimerTick[MAX_TIMER_COUNT];
};

#endif // WATCHTASK_H
