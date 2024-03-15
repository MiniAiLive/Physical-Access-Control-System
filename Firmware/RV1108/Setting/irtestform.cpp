#include "irtestform.h"
#include "base.h"
#include "shared.h"
#include "uitheme.h"
#include "stringtable.h"
#include "engineparam.h"
#include "camerasurface.h"
#include "camera_api.h"
#include "mainwindow.h"
#include "rokthread.h"
#include "alertdlg.h"
#include "lcdtask.h"
#include "drv_gpio.h"
#include "mainbackproc.h"
#include "faceengine.h"

#include <QtGui>
#include <unistd.h>

static int g_iEnrolledCount = 0;
static int g_iEnrollResult = 0;
static SRect g_xEnrollRect = { 0 };

extern unsigned char g_pbDiffIrImage[N_D_ENGINE_SIZE];

IRTestForm::IRTestForm(QGraphicsView *pView, FormBase* pParentForm)
                               : FormBase(pView, pParentForm)
{
    setAutoDelete(false);

    m_iRunning = 0;

    SetBGColor(QColor(Qt::black));
}

IRTestForm::~IRTestForm()
{
}

void IRTestForm::StartTest()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;

#endif
    m_iRunning = 1;

    g_iEnrolledCount = 0;
    g_iEnrollResult = 0;
    memset(&g_xEnrollRect, 0, sizeof(g_xEnrollRect));

    FormBase::OnStart(0);

    QThreadPool::globalInstance()->start(this);
}

void IRTestForm::OnPause()
{
    FormBase::OnPause();

    m_iRunning = 0;
    QThreadPool::globalInstance()->waitForDone();
}

void IRTestForm::run()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    int iOldCam = g_xCS.x.bShowCam;
    g_xCS.x.bShowCam = 1;

    float aEngineResult[10];
    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* pRokThread = w->GetROK();

    LCDTask::FB_Init();
    LCDTask::LCD_MemClear(0xFF000000);
    LCDTask::LCD_Update();

    usleep(500 * 1000);
    memset(&g_show_buf, 0, sizeof(g_show_buf));
    StartCamSurface(0);

    int iCamError = 0;
    while(m_iRunning)
    {
        if(g_iIRCamInited < 0)
        {
            iCamError = 1;
            usleep(1000 * 1000);
            break;
        }

        pRokThread->InitTime();

        camera_set_irled_on(IR_CAM, 1);
        WaitIRTimeout(500);

        FaceEngine::ExtractFace(NULL, g_irOnBayerL, g_irOnBayerR, aEngineResult);

        int fIsDetected = aEngineResult[1] == 1 ? 1 : 0;
        if(fIsDetected)
        {
            g_xEnrollRect.x = (int)aEngineResult[2];
            g_xEnrollRect.y = (int)aEngineResult[3];
            g_xEnrollRect.width = (int)aEngineResult[4];
            g_xEnrollRect.height = (int)aEngineResult[5];
        }
        else
            memset(&g_xEnrollRect, 0, sizeof(g_xEnrollRect));

        g_nFaceRectValid = fIsDetected;
        CalcNextExposure();

        LCDTask::LCD_MemClear(0);
        LCDTask::DrawMemFace(g_xEnrollRect.x, g_xEnrollRect.y, g_xEnrollRect.width, g_xEnrollRect.height, COLOR_GREEN);
        LCDTask::LCD_Update();

        usleep(300 * 1000);
    }

    StopCamSurface();

    LCDTask::LCD_MemClear(0xFF000000);
    LCDTask::LCD_Update();
    LCDTask::FB_Release();

    g_xCS.x.bShowCam = iOldCam;

    if(!FormBase::QuitFlag)
        emit SigBack(iCamError);
}

void IRTestForm::RetranslateUI()
{
}

bool IRTestForm::event(QEvent* e)
{
    if(e->type() == EV_KEY_EVENT)
    {
        KeyEvent* pEvent = static_cast<KeyEvent*>(e);
        if (pEvent->m_iKeyID == E_BTN_FUNC)
        {
            switch(pEvent->m_iEvType)
            {
            case KeyEvent::EV_CLICKED:
                m_iRunning = 0;
                break;
            case KeyEvent::EV_DOUBLE_CLICKED:
                break;
            case KeyEvent::EV_LONG_PRESSED:
                qApp->exit(QAPP_RET_OK);
                break;
            }
        }
    }

    return QWidget::event(e);
}
