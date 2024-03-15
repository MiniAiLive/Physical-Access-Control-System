#ifndef CAMERASURFACE_H
#define CAMERASURFACE_H

#include "thread.h"

class CameraSurface : public Thread
{
public:
    CameraSurface();
    virtual ~CameraSurface();

    void    Start();
    void    Stop();

protected:
    void    run();

private:
    int     m_iRunning;
};

#endif // BASE_H
