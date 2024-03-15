#ifndef ENROLLFACEFORM_H
#define ENROLLFACEFORM_H

#include "formbase.h"
#include <QRunnable>
#include <QThread>
#include <QMutex>
#include <QTime>

class QLabel;
class EnrollThread : public QThread
{
    Q_OBJECT
public:
    explicit EnrollThread();
    ~EnrollThread();

protected:
    void    run();
};

class EnrollFaceForm : public FormBase, public QRunnable
{
    Q_OBJECT
public:
    explicit EnrollFaceForm(QGraphicsView *pView, FormBase* pParentForm);
    ~EnrollFaceForm();

    enum {EF_FINISH, EF_START, EF_LOOP};


    void    StartEnroll();
    void    run();
signals:
    void    SigEnrollFinished(int iEnrollResult);

public slots:
    void    OnPause();

protected:
    void    mousePressEvent(QMouseEvent* e);
    void    RetranslateUI();

private:
    EnrollThread*   m_pEnrollThread;
};


#endif // ENROLLFACEFORM_H
