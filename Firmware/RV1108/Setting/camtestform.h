#ifndef CAMTESTFORM_H
#define CAMTESTFORM_H

#include "formbase.h"
#include "touchthread.h"
#include <QRunnable>
#include <QMutex>

#define DEV_CLR_CAM 1000
#define DEV_IR_CAM1 1001

class LCDTask;
class CamTestForm : public FormBase, public QRunnable
{
    Q_OBJECT
public:
    explicit CamTestForm(QGraphicsView* pxView, FormBase* pxParentForm);
    ~CamTestForm();

    void    StartTest(int iCamID, int iTimeout = -1, int iTestVol = 0, int iSetCam = 0);
    void    run();

signals:
    void    SigBack(int iCamError);

public slots:
    void    OnPause();

protected:
    void    mousePressEvent(QMouseEvent* e);
    void    mouseMoveEvent(QMouseEvent* e);
    void    mouseReleaseEvent(QMouseEvent* e);
    bool    event(QEvent* e);

    void    InitSettings();
    void    ResetButtons();
    void    AddButton(int iID, int iX1, int iY1, int iX2, int iY2, const char* szNormal, const char* szPress, unsigned int iNormalColor, int iPressColor, int iState = BTN_STATE_NONE);
    int     CheckBtnState(QPoint pos, int mode);
    void    UpdateLCD();

private:
    int     m_ID;

    QMutex  m_xMutex;
    int     m_fRunning;

    int     m_fTestVol;
    int     m_iTimeOut;
    bool    m_bIsDevTest;
    
    int     m_iSetClrCam;
    int     m_iImageProcess[3];
    QRect   m_axProcessSlider[3];
    int     m_iSelectSlider;

    int     m_iSettings;
    int     m_iBtnCount;
    BUTTON  m_axBtns[MAX_BUTTON_CNT];
};

#endif // CAMTESTFORM_H
