#ifndef CAMERACALIBFORM_H
#define CAMERACALIBFORM_H

#include "formbase.h"
#include <QRunnable>
#include <QMutex>
#include <QThread>

class CalibThread : public QThread
{
    Q_OBJECT
public:
    explicit CalibThread();
    ~CalibThread();

protected:
    void    run();
};


class CamCalibForm : public FormBase, public QRunnable
{
    Q_OBJECT
public:
    explicit CamCalibForm(QGraphicsView *pView, FormBase* pParentForm);
    ~CamCalibForm();

    enum {CF_FINISH, CF_START, CF_LOOP};

    void    StartCalib();
    void    run();

public slots:
    void    OnPause();
    
signals:
    void    sigCalibFinished(int iResult, int iCamX, int iCamY);
    
public slots:

protected:
    void    mousePressEvent(QMouseEvent* e);
    
private:
    int     m_iRunning;
    QImage  m_xGuideImage;

    CalibThread*    m_pCalibThread;
};

#endif // CAMERACALIBFORM_H
