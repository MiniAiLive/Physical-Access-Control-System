#include "thread.h"
#include <stdio.h>
#include <stdlib.h>

#ifndef USE_QT

void* ThreadProc1(void*);

Thread::Thread()
{
    m_thread = 0;
}

void Thread::Start()
{
    if(pthread_create(&m_thread, NULL, ThreadProc1, this) != 0)
        perror ("can't create thread\n");
}

void Thread::Wait()
{
    if(m_thread != 0)
    {
        pthread_join(m_thread, NULL);
        m_thread = 0;
    }
}

void Thread::Exit()
{
    if(m_thread != 0)
    {
        pthread_cancel(m_thread);
        m_thread = 0;
    }
}

void Thread::ThreadProc()
{
    run();
}

void* ThreadProc1(void* param)
{
    Thread* pThread = (Thread*)(param);
    pThread->ThreadProc();

    pthread_exit(NULL);
    return NULL;
}

#else

Thread::Thread(QObject* parent)
    : QThread(parent)
{

}

void Thread::Start()
{
    QThread::start();
}

void Thread::Wait()
{
    QThread::wait();
}

void Thread::Exit()
{
    QThread::terminate();
}

#endif
