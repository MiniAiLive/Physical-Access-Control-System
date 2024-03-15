#ifndef THREAD_H
#define THREAD_H

#ifndef USE_QT

#include <pthread.h>

class Thread
{
public:
    Thread();

    void Start();
    void Wait();
    void Exit();

    void ThreadProc();

protected:
    virtual void run() = 0;

    pthread_t   m_thread;
};

#else

#include <QThread>

class Thread : public QThread
{
    Q_OBJECT
public:
    Thread(QObject* parent = 0);

    void Start();
    void Wait();
    void Exit();

protected:
};

#endif

#endif // THREAD_H
