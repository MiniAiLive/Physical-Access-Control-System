#include "camcalibform.h"
#include "base.h"
#include "mainwindow.h"
#include "camera_api.h"
#include "shared.h"
#include "engineparam.h"
#include "camerasurface.h"
#include "camera_api.h"
#include "rokthread.h"
#include "alertdlg.h"
#include "lcdtask.h"
#include "faceengine.h"
#include "FaceRetrievalSystem.h"
#include "DBManager.h"
#include "my_lang.h"
#include "msg.h"

#include <QtGui>
#include <unistd.h>
#include <linux/videodev2.h>
#include <linux/fb.h>
#include <sys/ioctl.h>

int g_iCalibCount = 0;
SRect g_xCalibRect = { 0 };
int   g_aCalibResult[3] = { 0 };

CamCalibForm::CamCalibForm(QGraphicsView *pView, FormBase* pParentForm) :
    FormBase(pView, pParentForm)
{
    SetBGColor(Qt::black);
    setAutoDelete(false);

    m_xGuideImage = QImage("/usr/share/movie/00.png");
    unsigned char* pbData = m_xGuideImage.bits();
    for(int i = 0; i < m_xGuideImage.width() * m_xGuideImage.height(); i ++)
    {
        pbData[i * 4] = pbData[i * 4 + 3];
        pbData[i * 4 + 1] = pbData[i * 4 + 3];
        pbData[i * 4 + 2] = pbData[i * 4 + 3];
    }

    m_pCalibThread = NULL;
}

CamCalibForm::~CamCalibForm()
{

}

void CamCalibForm::StartCalib()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    FormBase::OnStart();
    m_iRunning = 1;

    g_iCalibCount = 0;
    memset(&g_xCalibRect, 0, sizeof(g_xCalibRect));
    memset(g_aCalibResult, 0, sizeof(g_aCalibResult));

    QThreadPool::globalInstance()->start(this);
}

void CamCalibForm::OnPause()
{
    FormBase::OnPause();

    m_iRunning = 0;
    QThreadPool::globalInstance()->waitForDone();
}

void CamCalibForm::run()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    message_queue_init(&g_worker, sizeof(MSG), MAX_MSG_NUM);
    SendGlobalMsg(MSG_CALIB_FACE, CF_START, 0, 0);

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* pRokThread = w->GetROK();

    unsigned int* piMemBuf = LCDTask::GetMemBuf();
#if (SCREEN_MODE == 0)
    QImage xScreenImg((unsigned char*)piMemBuf, 480, 360, QImage::Format_ARGB32);
#else
    QImage xScreenImg((unsigned char*)piMemBuf, MAX_X, MAX_Y, QImage::Format_ARGB32);
#endif
    QImage xGuideImage(":/icons/00.png");
    QImage xProgressImg(":/icons/progress_bar.png");

    QPainter painter;
    painter.begin(&xScreenImg);

    MSG* pMsg = NULL;
    while(1)
    {
        pMsg = (MSG*)message_queue_read(&g_worker);
        if(pMsg->type == MSG_CALIB_FACE)
        {
            if(pMsg->data1 == CF_FINISH)
            {
                LCDTask::LCD_MemClear(0);
                painter.fillRect(xScreenImg.rect(), Qt::black);
                LCDTask::LCD_Update();

                StopCamSurface();

                if(m_pCalibThread)
                {
                    m_pCalibThread->wait();
                    delete m_pCalibThread;
                    m_pCalibThread = NULL;
                }

                LCDTask::FB_Release();
                break;
            }
            else if(pMsg->data1 == CF_START)
            {
                LCDTask::FB_Init();
                LCDTask::VideoStart();
                LCDTask::DispOn();

                memset(&g_show_buf, 0, sizeof(g_show_buf));
                StartCamSurface(1);

                camera_set_irled(IR_CAM, g_xEP.iLedStatus, 0);

                m_pCalibThread = new CalibThread;
                m_pCalibThread->start();
            }
            else if(pMsg->data1 == CF_LOOP)
            {
                pRokThread->InitTime();
            }
        }
        else if(pMsg->type == MSG_CAMERA)
        {
            if(pMsg->data1 == EC_CAM_FIRST_RECV)
            {
                SendGlobalMsg(MSG_DRAW_LCD, 0, 0, 0);
            }
        }
        else if(pMsg->type == MSG_DRAW_LCD)
        {
            LCDTask::LCD_MemClear(0);
            painter.fillRect(xScreenImg.rect(), Qt::NoBrush);
            if(g_aCalibResult[0] == 0)
            {
                if(g_xCalibRect.width == 0)
                    painter.drawImage(QRect(0, 0, xScreenImg.width(), xScreenImg.height()), xGuideImage);

                LCDTask::DrawMemFace(g_xCalibRect.x, g_xCalibRect.y, g_xCalibRect.width, g_xCalibRect.height, COLOR_RED, 1, 0, 0);
            }
            else if(g_aCalibResult[0] == 1)
            {
                LCDTask::DrawMemFace(g_xCalibRect.x, g_xCalibRect.y, g_xCalibRect.width, g_xCalibRect.height, COLOR_GREEN, 1, g_aCalibResult[1], g_aCalibResult[2]);
            }

#if 0
            int iProgressWid = (xProgressImg.width() - 8) * g_iCalibCount / 5;
            painter.fillRect(QRect((xScreenImg.width() - xProgressImg.width()) / 2 + 5, xScreenImg.height() - xProgressImg.height(),
                                   iProgressWid, xProgressImg.height()), QColor(0x21, 0x7b, 0xc2));

            painter.drawImage(QRect((xScreenImg.width() - xProgressImg.width()) / 2, xScreenImg.height() - xProgressImg.height(),
                                    xProgressImg.width(), xProgressImg.height()),
                              xProgressImg);
#else
            int iProgressWid = xScreenImg.width() * g_iCalibCount / 5;
            painter.fillRect(QRect(0, xScreenImg.height() - PROGRESSBAR_HEIGHT, iProgressWid, PROGRESSBAR_HEIGHT), QColor(0x21, 0x7b, 0xc2));
#endif

            painter.drawImage(QRect(0, 0, xScreenImg.width(), xScreenImg.height()), xGuideImage);
            LCDTask::LCD_Update();
        }

        message_queue_message_free(&g_worker, (void*)pMsg);
    }

    painter.end();

    if(pMsg != NULL)
        message_queue_message_free(&g_worker, (void*)pMsg);

    message_queue_destroy(&g_worker);

    AlertDlg::Locked = 1;

    if(!FormBase::QuitFlag)
        emit sigCalibFinished(g_aCalibResult[0], g_aCalibResult[1], g_aCalibResult[2]);
}


void CamCalibForm::mousePressEvent(QMouseEvent* e)
{
    QWidget::mousePressEvent(e);

    g_xSS.iRunningCamSurface = 0;
}

CalibThread::CalibThread()
{

}

CalibThread::~CalibThread()
{

}

void CalibThread::run()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    int iValidClr = 0;
    float aEngineResult[10];

    float rOldTime = Now();

    FaceEngine::InitCalibOffset();

    //waiting for camera switch
    usleep(300 * 1000);
    while(g_xSS.iRunningCamSurface)
    {
        if(g_iIRCamInited < 0 || g_iClrCamInited == -1)
        {
            g_aCalibResult[0] = 3;
            break;
        }

        camera_set_irled_on(IR_CAM, 1);
        WaitIRTimeout(500);

        if(g_xSS.iRunningCamSurface == 0)
            break;

        if(g_iClrCamInited != -1 && g_xSS.iShowIrCamera == 0)
        {
            if(g_xSS.iRunningCamSurface == 0)
                break;
        }

#if (AUTO_TEST == 1)
        char szFilePath[256] = { 0 };
        if(rand() % 2 == 0)
            sprintf(szFilePath, "/test/480_640_success.bin");
        else
            sprintf(szFilePath, "/test/480_640_failed.bin");
        FILE* fp = fopen(szFilePath, "rb");
        if(fp)
        {
            fread(g_irOnData, sizeof(g_irOnData), 1, fp);
            fclose(fp);
        }
#endif

        if(g_iClrCamInited != -1 && g_xSS.iShowIrCamera == 0)
        {
            pthread_mutex_lock(&g_clrWriteLock);
            memcpy(g_clrRgbData, g_clrTmpRgbData, sizeof(g_clrTmpRgbData));
            pthread_mutex_unlock(&g_clrWriteLock);

            iValidClr = 1;
        }

        FaceEngine::ExtractFace(iValidClr == 1 ? g_clrRgbData : NULL, g_irOnBayerL, g_irOnBayerR, aEngineResult);

        int iIsDetected = aEngineResult[1] == 1 ? 1 : 0;
        if(iIsDetected)
        {
            g_xCalibRect.x = (int)aEngineResult[2];
            g_xCalibRect.y = (int)aEngineResult[3];
            g_xCalibRect.width = (int)aEngineResult[4];
            g_xCalibRect.height = (int)aEngineResult[5];

//            FaceRectFit2ClrCamera(g_xCalibRect.x, g_xCalibRect.y, g_xCalibRect.width, g_xCalibRect.height);

//            if (g_xCalibRect.width < 130)
//                iIsDetected = 0;

            rOldTime = Now();
        }
        else
            memset(&g_xCalibRect, 0, sizeof(g_xCalibRect));

        g_nFaceRectValid = iIsDetected;

        if(Now() - rOldTime > SETTING_TIMEOUT * 1000)
            break;

        fr_calc_Off((unsigned short*)g_irOffData);
        CalcNextExposure();

        if(aEngineResult[0] == ES_INVALID)
        {
            g_aCalibResult[0] = 2;
            break;
        }
        if (g_aCalibResult[0] == 1)
        {
            break;
        }

        if (g_iCalibCount > 1/* && (g_iCalibCount % 2) == 0*/)
        {
            int nResOffsetX = 0, nResOffsetY = 0;
            if (fr_GetOffsetIr2Clr(&nResOffsetX, &nResOffsetY) == ES_SUCCESS)
            {
                g_aCalibResult[0] = 1;
                g_aCalibResult[1] = nResOffsetX;
                g_aCalibResult[2] = nResOffsetY;

                LOG_PRINT("[Calib Cam] cam calibration offset = %d, %d\n", g_aCalibResult[1], g_aCalibResult[2]);
                break;
            }
            else if (g_iCalibCount == MAX_CALCOFFSETPROCESS_COUNT)
            {
                g_aCalibResult[0] = 4;
                break;
            }

        }

        float rCurTime = Now();
        if(rCurTime - rOldTime < 500)
            usleep((500 - (rCurTime - rOldTime)) * 1000);

        if (iIsDetected)
            g_iCalibCount ++;

        SendGlobalMsg(MSG_DRAW_LCD, 0, 0, 0);
        SendGlobalMsg(MSG_CALIB_FACE, CamCalibForm::CF_LOOP, 0, 0);
    }

    SendGlobalMsg(MSG_CALIB_FACE, CamCalibForm::CF_FINISH, 0, 0);
}

