#include "enrollfaceform.h"
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
#include "faceengine.h"
#include "soundbase.h"
#include "playthread.h"
#include "my_lang.h"
#include "FaceRetrievalSystem.h"
#include "msg.h"
#include "rk_fb.h"

#include <QtGui>
#include <QLabel>
#include <unistd.h>

int g_iEnrolledCount = 0;
int g_iEnrollResult = 0;
SRect g_xEnrollRect = { 0 };

EnrollFaceForm::EnrollFaceForm(QGraphicsView *pView, FormBase* pParentForm)
                               : FormBase(pView, pParentForm)
{
    SetBGColor(Qt::black);
    m_pEnrollThread = NULL;

    setAutoDelete(false);
}

EnrollFaceForm::~EnrollFaceForm()
{
}

void EnrollFaceForm::StartEnroll()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    g_iEnrolledCount = 0;
    g_iEnrollResult = 0;
    memset(&g_xEnrollRect, 0, sizeof(g_xEnrollRect));

    update();
    FormBase::OnStart(0);

    message_queue_init(&g_worker, sizeof(MSG), MAX_MSG_NUM);
    QThreadPool::globalInstance()->start(this);
}

void EnrollFaceForm::OnPause()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif
    FormBase::OnPause();

    SendGlobalMsg(MSG_ENROLL_FACE, EF_FINISH, 0, 0);
    QThreadPool::globalInstance()->waitForDone();

    message_queue_destroy(&g_worker);
}

void EnrollFaceForm::run()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    SendGlobalMsg(MSG_ENROLL_FACE, EF_START, 0, 0);

    PlayEnrollFaceSound(0);

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
        if(pMsg->type == MSG_ENROLL_FACE)
        {
            if(pMsg->data1 == EF_FINISH)
            {
                LCDTask::LCD_MemClear(0);
                painter.fillRect(xScreenImg.rect(), Qt::black);
                LCDTask::LCD_Update();

                StopCamSurface();

                if(m_pEnrollThread)
                {
                    m_pEnrollThread->wait();
                    delete m_pEnrollThread;
                    m_pEnrollThread = NULL;
                }

                LCDTask::FB_Release();
                break;
            }
            else if(pMsg->data1 == EF_START)
            {
                LCDTask::FB_Init();
                LCDTask::VideoStart();
                LCDTask::DispOn();

                memset(&g_show_buf, 0, sizeof(g_show_buf));
                StartCamSurface(0);

                camera_set_irled(IR_CAM, g_xEP.iLedStatus, 0);
                m_pEnrollThread = new EnrollThread();
                m_pEnrollThread->start();
            }
            else if(pMsg->data1 == EF_LOOP)
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
            painter.drawImage(QRect(0, 0, xScreenImg.width(), xScreenImg.height()), xGuideImage);

#if 0
            int iProgressWid = (xProgressImg.width() - 8) * g_iEnrolledCount / 5;
            painter.fillRect(QRect((xScreenImg.width() - xProgressImg.width()) / 2 + 5, xScreenImg.height() - xProgressImg.height(),
                                   iProgressWid, xProgressImg.height()), QColor(0x21, 0x7b, 0xc2));

            painter.drawImage(QRect((xScreenImg.width() - xProgressImg.width()) / 2, xScreenImg.height() - xProgressImg.height(),
                                    xProgressImg.width(), xProgressImg.height()),
                              xProgressImg);
#else
            int iProgressWid = xScreenImg.width() * g_iEnrolledCount / 5;
            painter.fillRect(QRect(0, xScreenImg.height() - PROGRESSBAR_HEIGHT, iProgressWid, PROGRESSBAR_HEIGHT), QColor(0x21, 0x7b, 0xc2));
#endif

            painter.save();
            painter.setPen(Qt::white);
#if (SCREEN_MODE)
            painter.setFont(g_UITheme->SubTextFont);
#else
            QFont f = painter.font();
            f.setPointSize(11);
            painter.setFont(f);
#endif

            switch(g_iEnrolledCount)
            {
            case 0:
                painter.drawText(QRect(10, xScreenImg.height() - LCD_FOOTER_HEIGHT, xScreenImg.width() - 20, LCD_FOOTER_HEIGHT),
                            Qt::AlignCenter, StringTable::Str_Enroll_face);
                break;
            case 1:
            case 2:
                painter.drawText(QRect(10, xScreenImg.height() - LCD_FOOTER_HEIGHT, xScreenImg.width() - 20, LCD_FOOTER_HEIGHT),
                            Qt::AlignCenter, StringTable::Str_Enroll_face_up);
                break;
            case 3:
            case 4:
                painter.drawText(QRect(10, xScreenImg.height() - LCD_FOOTER_HEIGHT, xScreenImg.width() - 20, LCD_FOOTER_HEIGHT),
                            Qt::AlignCenter, StringTable::Str_Enroll_face_down);
                break;
            }
            painter.restore();

            if(g_xSS.iShowIrCamera == 0)
            {
                LCDTask::DrawMemFace(g_xEnrollRect.x, g_xEnrollRect.y, g_xEnrollRect.width, g_xEnrollRect.height, COLOR_GREEN, 1, g_xPS.x.iCamOffX, g_xPS.x.iCamOffY);
            }
            else
            {
                LCDTask::DrawMemFace(g_xEnrollRect.x, g_xEnrollRect.y, g_xEnrollRect.width, g_xEnrollRect.height, COLOR_GREEN);
            }

            LCDTask::LCD_Update();
        }

        message_queue_message_free(&g_worker, (void*)pMsg);
    }

    painter.end();

    if(pMsg != NULL)
        message_queue_message_free(&g_worker, (void*)pMsg);

    if(!FormBase::QuitFlag)
    {
        if(g_iEnrollResult >= 1)
        {
            emit SigEnrollFinished(g_iEnrollResult);
            return;
        }

        SigBack();
    }
}

void EnrollFaceForm::mousePressEvent(QMouseEvent* e)
{
    SendGlobalMsg(MSG_ENROLL_FACE, EF_FINISH, 0, 0);
}

void EnrollFaceForm::RetranslateUI()
{
}


EnrollThread::EnrollThread()
{

}

EnrollThread::~EnrollThread()
{

}

void EnrollThread::run()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif
    //waiting for camera switch
    usleep(300 * 1000);

    float aEngineResult[10];
    float rOldTime = Now();
    g_iHandControlFlag = 0; //changed by KSB 20180711

    while(g_xSS.iRunningCamSurface)
    {
        SendGlobalMsg(MSG_ENROLL_FACE, EnrollFaceForm::EF_LOOP, 0, 0);

        int iValidClr = 0;
        if(g_iIRCamInited < 0)
        {
            g_iEnrollResult = 4;
            break;
        }

        if(g_xSS.iShowIrCamera == 0)
        {
            if(g_xSS.iRunningCamSurface == 0)
                break;
        }

        float rStart = Now();
        if(g_iClrCamInited != -1 && g_xSS.iShowIrCamera == 0)
        {
            pthread_mutex_lock(&g_clrWriteLock);
            memcpy(g_clrRgbData, g_clrTmpRgbData, sizeof(g_clrTmpRgbData));
            pthread_mutex_unlock(&g_clrWriteLock);

            iValidClr = 1;
        }
        printf("===========pre cam capture1: %f\n", Now() - rStart);

        rStart = Now();
        camera_set_irled_on(IR_CAM, 1);
        WaitIRTimeout(500);

        printf("===============pre cam capture2: %f\n", Now() - rStart);
        if(g_xSS.iRunningCamSurface == 0)
            break;

#if (AUTO_TEST == 1)
        memset(g_irOffData, 0, sizeof(g_irOffData));

        char szFilePath[256] = { 0 };
        if(rand() % 2 == 0)
            sprintf(szFilePath, "/test/480_640_success.bin");
        else
            sprintf(szFilePath, "/test/480_640_failed.bin");
        printf("==========read file = %s\n", szFilePath);

        FILE* fp = fopen(szFilePath, "rb");
        if(fp)
        {
            fread(g_irOnData, sizeof(g_irOnData), 1, fp);
            fclose(fp);
        }
#endif

        rStart= Now();
        FaceEngine::ExtractFace(iValidClr == 1 ? g_clrRgbData : NULL, g_irOnBayerL, g_irOnBayerR, aEngineResult);        

        int fIsDetected = aEngineResult[1] == 1 ? 1 : 0;
        if(fIsDetected)
        {
            g_xEnrollRect.x = (int)aEngineResult[2];
            g_xEnrollRect.y = (int)aEngineResult[3];
            g_xEnrollRect.width = (int)aEngineResult[4];
            g_xEnrollRect.height = (int)aEngineResult[5];

//            if(iValidClr == 1)
//            {
//                FaceRectFit2ClrCamera(g_xEnrollRect.x, g_xEnrollRect.y, g_xEnrollRect.width, g_xEnrollRect.height);
//            }

            rOldTime = Now();

            printf("==============extract face time: %f\n", Now() - rStart);
        }
        else
            memset(&g_xEnrollRect, 0, sizeof(g_xEnrollRect));

        if(Now() - rOldTime > SETTING_TIMEOUT * 1000)
            break;

        g_nFaceRectValid = fIsDetected;

        fr_calc_Off((unsigned short*)g_irOffData);
        CalcNextExposure();

        SendGlobalMsg(MSG_DRAW_LCD, 0, 0, 0);

        rStart = Now();
        FaceEngine::RegisterFace(aEngineResult);
#if (AUTO_CAMERA_CALIB == 1)
        if (aEngineResult[0] != ES_FAILED && g_xPS.x.bCamAutoAdjust != END_CAM_CALIB)
        {
            int nResOffsetX = 0, nResOffsetY = 0;
            if (fr_GetOffsetIr2Clr(&nResOffsetX, &nResOffsetY) == ES_SUCCESS)
            {
                g_xPS.x.iCamOffX = nResOffsetX;
                g_xPS.x.iCamOffY = nResOffsetY;
                g_xPS.x.bCamAutoAdjust = END_CAM_CALIB;
                UpdatePermanenceSettings();
            }
        }
#endif

        if(aEngineResult[0] != ES_PROCESS)
        {
            printf("==============register face time: %f\n", Now() - rStart);

            if(aEngineResult[0] == ES_ENEXT || aEngineResult[0] == ES_SUCCESS)
            {
                g_iEnrolledCount ++;
                PlaySoundTop();

                printf("------------------%d\n", g_iEnrolledCount);
                if(g_iEnrolledCount == 1)
                    PlayEnrollFaceUpSound(0);
                else if(g_iEnrolledCount == 3)
                {
                    printf("------------------111  %d\n", g_iEnrolledCount);
                    PlayEnrollFaceDownSound(0);
                }
            }
            else if(aEngineResult[0] == ES_EPREV)
                g_iEnrolledCount --;
            else if(aEngineResult[0] == ES_SPOOF_FACE)
            {
                g_iEnrollResult = 3;
                break;
            }
            if(aEngineResult[0] == ES_SUCCESS)
            {
                g_iEnrollResult = 1;
                break;
            }
            else if(aEngineResult[0] == ES_DUPLICATED)
            {
                g_iEnrollResult = 2;
                break;
            }

            SendGlobalMsg(MSG_DRAW_LCD, 0, 0, 0);
        }

        if(fIsDetected == 0)
            usleep(300 * 1000);
    }

    SendGlobalMsg(MSG_ENROLL_FACE, EnrollFaceForm::EF_FINISH, 0, 0);
}
