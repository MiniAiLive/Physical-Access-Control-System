#include "verifyfaceform.h"
#include "ui_passcodeform.h"
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
#include "facerecogtask.h"
#include "DBManager.h"
#include "menuform.h"
#include "cardrecogtask.h"
#include "i2cbase.h"
#include "uarttask.h"
#include "settings.h"

#include <QtGui>
#include <QLabel>
#include <unistd.h>

#define SCENE_SHOW_TIME    1000

extern int g_iMinSize;

VerifyFaceForm::VerifyFaceForm(QGraphicsView *pView, FormBase* pParentForm)
                               : FormBase(pView, pParentForm)
{
    SetupUI();

    m_iUIWaiting = 0;
    m_iOldMsg = -1;
    m_rResetTime = 0;
    m_rClickTime = 0;
    m_iCounter = 0;
    m_pFaceRecogTask = NULL;
    m_pCardRecogTask = NULL;
    m_iCurScene = SCENE_NONE;
    ResetButtons();

    setAutoDelete(false);

    connect(this, SIGNAL(SigBack(int, int)), this, SLOT(GotoBack(int, int)));
}

VerifyFaceForm::~VerifyFaceForm()
{
    SendGlobalMsg(MSG_VERIFY, VF_END, 0, 0);
    OnStop();
    delete ui;
    ui = NULL;
}

void VerifyFaceForm::StartVerify()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    FormBase::OnStart(0);

    g_xSS.iVerifyFlag = 1;
    g_xSS.rVerifyLoopTime = Now();
    m_iTimer = startTimer(500);
    QThreadPool::globalInstance()->start(this);
}

void VerifyFaceForm::OnResume()
{
    FormBase::OnResume();
    RetranslateUI();
    printf("VerifyFaceForm::OnResume\n");

    if(g_xSS.iVDBStart == 1)
        SetUI(UI_VDB);

    g_xSS.iVerifyFlag = 1;
    g_xSS.rVerifyLoopTime = Now();
    m_iTimer = startTimer(500);
    QThreadPool::globalInstance()->start(this);
}

void VerifyFaceForm::OnPause()
{
    FormBase::OnPause();

    if(m_iTimer != -1)
        killTimer(m_iTimer);
    m_iTimer = -1;

    QThreadPool::globalInstance()->waitForDone();
}

void VerifyFaceForm::OnStop()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif
    if(m_iTimer != -1)
        killTimer(m_iTimer);
    m_iTimer = -1;

    QThreadPool::globalInstance()->waitForDone();
}

void VerifyFaceForm::GotoBack(int iUI, int iParam)
{
    printf("GotoBack %d, %d\n", iUI, iParam);
    if(iUI == BACK_MENU)
    {
        QThreadPool::globalInstance()->waitForDone();
        SetUI(UI_NONE);

        g_xSS.iNoSoundPlayFlag = 1;

        MenuForm* pMainForm = new MenuForm(m_pParentView, this);
        connect(pMainForm, SIGNAL(SigBack()), this, SLOT(OnResume()));
        pMainForm->OnStart();
    }
    else if(iUI == BACK_PASSWORD)
    {
        if(g_xSS.iVDBMode == 0)
        {
            m_rClickTime = Now();
            SetScene(SCENE_PASSWORD);

            ConnectButtons(0);
            SendGlobalMsg(MSG_VERIFY, VF_PASSWORD_STARTED, iParam, 0);
            printf("BACK_PASSWORD %f\n", Now());
        }
    }
    else if(iUI == BACK_SCREEN_SAVER)
    {
        printf("BACK_SCREEN_SAVER\n");
        SetScene(SCENE_SCREEN_SAVER);
        SendGlobalMsg(MSG_VERIFY, VF_SCREEN_SAVER_STARTED, iParam, 0);
    }
    else if(iUI == BACK_MAIN)
    {
        PlaySoundLeft();

        SetUI(UI_NONE);

        if(g_xSS.iRunningCamSurface)
            SendGlobalMsg(MSG_VERIFY, VF_VIDEO_START, 0, 0);
        else
            SendGlobalMsg(MSG_VERIFY, VF_START, 0, 0);
    }
    else if(iUI == BACK_VDB)
    {
        SetUI(UI_VDB);
        //SendGlobalMsg(MSG_VERIFY, VF_VDB_START, 0, 0);
        if(g_xSS.iRunningCamSurface)
            SendGlobalMsg(MSG_VERIFY, VF_VDB_START, 0, 0);
        else
            SendGlobalMsg(MSG_VERIFY, VF_START, 1, 0);
    }
}

void VerifyFaceForm::run()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    m_iUIWaiting = 0;
    g_xSS.iAdminMode = VERIFY_USER;
    m_pFaceRecogTask = new FaceRecogTask();
    m_pCardRecogTask = new CardRecogTask();

    ResetMultiVerify();

    message_queue_init(&g_worker, sizeof(MSG), MAX_MSG_NUM);
    printf("[VeruftFace] VDBStart = %d\n", g_xSS.iVDBStart);

    if(g_xSS.iVDBStart == 1)
    {
        m_iUIWaiting = 1;
        g_xSS.iVDBStart = 0;
        g_xSS.iVDBMode = 1;
        SendGlobalMsg(MSG_VERIFY, VF_START, 1, 0);
    }
    else
    {
        m_iUIWaiting = 1;
        g_xSS.iVDBMode = 0;
        SendGlobalMsg(MSG_VERIFY, VF_START, 0, 0);
    }

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* pRokThread = w->GetROK();
    pRokThread->SetKeyScanMode(this);

    UARTTask* pBackUART = w->GetUART(BACK_BOARD_PORT);
    UARTTask* pPWRUART = w->GetUART(PWR_BOARD_PORT);

    unsigned int* piMemBuf = LCDTask::GetMemBuf();
#if (SCREEN_MODE == 0)
    QImage xScreenImg((unsigned char*)piMemBuf, 480, 360, QImage::Format_ARGB32);
#else
    QImage xScreenImg((unsigned char*)piMemBuf, MAX_X, MAX_Y, QImage::Format_ARGB32);
#endif

    QPainter painter;
    painter.begin(&xScreenImg);

    MSG* pMsg = NULL;
    while(1)
    {
        pMsg = (MSG*)message_queue_read(&g_worker);
        g_xSS.rVerifyLoopTime = Now();

        if(pMsg->type == MSG_VERIFY)
        {
            if(pMsg->data1 == VF_END)
            {
                painter.end();

                if(pMsg != NULL)
                    message_queue_message_free(&g_worker, (void*)pMsg);

                message_queue_destroy(&g_worker);

                m_pFaceRecogTask->Stop();
                delete m_pFaceRecogTask;
                m_pFaceRecogTask = NULL;

                m_pCardRecogTask->Stop();
                delete m_pCardRecogTask;
                m_pCardRecogTask = NULL;

                pRokThread->SetKeyScanMode(NULL);
                return;
            }
            if(pMsg->data1 == VF_START)
            {
                printf("pMsg->data1 == VF_START %d\n", m_iUIWaiting);
                m_iUIWaiting = 1;

                memset(&g_show_buf, 0, sizeof(g_show_buf));
                StartCamSurface(0);

                if(pMsg->data2 == 0)
                    SendGlobalMsg(MSG_VERIFY, VF_VIDEO_START, 0, 0);
                else
                    SendGlobalMsg(MSG_VERIFY, VF_VDB_START, 0, 0);
            }
            else if(pMsg->data1 == VF_VIDEO_START)
            {
                printf("VF_VIDEO_START\n");
                AudioCommThread::Release();

                camera_set_regval(CLR_CAM, 0x17, 0x15);

                my_system("echo 0 > /sys/class/display/TV/enable");
                LCDTask::DispOn();
                LCDTask::FB_Init();
                LCDTask::VideoStart();

                SetScene(SCENE_MAIN);
                SendGlobalMsg(MSG_VERIFY, VF_MAIN_STARTED, 0, 0);
            }
            else if(pMsg->data1 == VF_VDB_START)
            {
                printf("pMsg->data1 == VF_VDB_START %d\n", m_iUIWaiting);

                camera_set_regval(CLR_CAM, 0x17, 0x14);

                usleep(300 * 1000);

                m_iUIWaiting = 1;
                LCDTask::DispOff();
                my_system("echo 1 > /sys/class/display/TV/enable");

                LCDTask::FB_Init(1);
                LCDTask::VideoStart();

                SetScene(SCENE_VDB);
                SendGlobalMsg(MSG_VERIFY, VF_VDB_STARTED, 0, 0);

                AudioCommThread::Create();
            }
            else if(pMsg->data1 == VF_FINISH)
            {
                qDebug() << "VF_FINISH" << m_iCounter << pMsg->data3 << m_iUIWaiting;

                if(g_xSS.iRunningCamSurface)
                {
                    printf("Camera Close!\n");
                    LCDTask::LCD_MemClear(0);
                    painter.fillRect(xScreenImg.rect(), Qt::black);
                    LCDTask::LCD_Update();

                    StopCamSurface();
                }

                SendGlobalMsg(MSG_VERIFY, VF_VIDEO_STOP, pMsg->data2, pMsg->data3);
            }
            else if(pMsg->data1 == VF_VIDEO_STOP)
            {
                printf("VF_VIDEO_STOP\n");
                m_pFaceRecogTask->Stop();
                m_pCardRecogTask->Stop();

                LCDTask::VideoStop();
                LCDTask::FB_Release();

                AudioCommThread::Release();

                camera_set_irled(IR_CAM, 0, 0);
                GPIO_fast_setvalue(IR_LED, OFF);

                if(pMsg->data2 == BACK_PASSWORD)
                {
                    emit SigBack(BACK_PASSWORD, 0);
                }
                else if(pMsg->data2 == BACK_SCREEN_SAVER)
                {
                    printf("SigBack Screen Saver!\n");
                    emit SigBack(BACK_SCREEN_SAVER, 0);
                }
                else if(pMsg->data2 == BACK_MAIN)
                {
                    printf("pMsg->data1 == BACK_MAIN\n");
                    ResetMultiVerify();

                    g_xSS.iNoSoundPlayFlag = 1;
                    g_xSS.iAdminMode = VERIFY_USER;

                    emit SigBack(BACK_MAIN, 0);
                }
                else if(pMsg->data2 == BACK_VDB)
                {
                    emit SigBack(BACK_VDB, 0);
                }
                else
                {
                    PlaySoundLeft();
                    break;
                }
            }
            else if(pMsg->data1 == VF_PASSWORD_STARTED)
            {
                update();
                printf("VF_PASSWORD_STARTED %f\n", Now());
                if(pMsg->data2 == 1)
                    PlayRetypePasscodeSound(1);
                else if(pMsg->data2 == 0)
                {
                    PlayTypePasscodeSound(1);
                }
                else if(pMsg->data2 == 2)
                    PlayManagerSound();

                ConnectButtons(1);
                m_iCounter ++;
                m_iCurScene = SCENE_PASSWORD;
                m_iUIWaiting = 0;
            }
            else if(pMsg->data1 == VF_SCREEN_SAVER_STARTED)
            {
                m_iCounter ++;
                m_iCurScene = SCENE_SCREEN_SAVER;
                m_iUIWaiting = 0;

                if(g_iMinSize < 15 * 1000)
                    qApp->exit(QAPP_RET_OK);
            }
            else if(pMsg->data1 == VF_MAIN_STARTED)
            {
                m_iCounter ++;
                m_iCurScene = SCENE_MAIN;
                m_iUIWaiting = 0;
            }
            else if(pMsg->data1 == VF_VDB_STARTED)
            {
                m_iCounter ++;
                g_xSS.iVDBError = 0;
                m_iCurScene = SCENE_VDB;
            }
            else if(pMsg->data1 == VF_PASSWORD_RESULT)
            {
                int iFindID = pMsg->data2;
                if(iFindID >= 0)//succe
                {
                    if(g_xCS.x.bAuthMode == AUTH_MODE_SINGLE || g_xSS.iAdminMode == VERIFY_MANAGER)
                    {
                        SaveLog(LOG_TYPE_PASSCODE, LOG_INFO_SUCCESS, iFindID, -1);
                    }
                    else if(g_xCS.x.bAuthMode == AUTH_MODE_MULTI)
                    {
                        if(g_xSS.iFirstSuccIdx > 0)
                        {
                            SaveLog(LOG_TYPE_PASSCODE, LOG_INFO_SUCCESS, iFindID, g_xSS.iFirstSuccType);
                        }
                        else
                        {
                            SaveLog(LOG_TYPE_PASSCODE, LOG_INFO_SUCCESS, iFindID, -1);
                        }
                    }

                    if(pMsg->data2 != -1 && g_xSS.iAdminMode == VERIFY_MANAGER)
                    {
                        PlayWelcomeSound();
                        SendGlobalMsg(MSG_VERIFY, VF_FINISH, BACK_MENU, 0);
                        break;
                    }
                    else
                    {
                        MainBackProc::MotorOpenClose(pPWRUART, 0, 0, 0, g_xCS.x.bUnlockTime, 0);
                        PlayEffSuccess();
                        usleep(SCENE_SHOW_TIME * 1000);

                        m_iUIWaiting = 1;
                        SigBack(BACK_PASSWORD, 0);

                        ResetMultiVerify();
                    }
                }
                else //failed
                {
                    if(g_xCS.x.bAuthMode == AUTH_MODE_SINGLE || g_xSS.iAdminMode == VERIFY_MANAGER)
                    {
                        SaveLog(LOG_TYPE_PASSCODE, LOG_INFO_FAILED, -1, -1);
                    }
                    else
                    {
                        int iFirstSuccType = g_xSS.iFirstSuccType;
                        if (iFirstSuccType == LOG_TYPE_PASSCODE)
                            iFirstSuccType = -1;

                        int iFirstUserIdx = g_xSS.iFirstSuccIdx - 1;
                        PSMetaInfo pxMetaInfo = dbm_GetPersonMetaInfoByID(iFirstUserIdx);
                        if (pxMetaInfo)
                            SaveLog(LOG_TYPE_PASSCODE, LOG_INFO_FAILED, pxMetaInfo->nID, iFirstSuccType);
                        else
                            SaveLog(LOG_TYPE_PASSCODE, LOG_INFO_FAILED, -1, iFirstSuccType);
                    }

                    ResetMultiVerify();

                    PlayEffFailed();
                    usleep(SCENE_SHOW_TIME * 1000);

                    m_iUIWaiting = 1;
                    SigBack(BACK_PASSWORD, 0);
                }
                ResetMultiVerify();
            }
            else if(pMsg->data1 == VF_EVENT)
            {
                printf("pMsg->data1 == VF_EVENT  %d, %d, %d\n", pMsg->data2, m_iCurScene, SCENE_SCREEN_SAVER);
                if(pMsg->data2 == HM_BELL_KEY)
                {
                    printf("pMsg->data2 == HM_BELL_KEY\n");
                    int iState = MainBackProc::ST_RECV_ACK;
                    if(pMsg->data3 == 0)
                        iState = MainBackProc::MotorOpen(pBackUART, 0, 0, 0, 1);

                    if(g_xSS.iVDBMode == 0)
                    {
                        m_iUIWaiting = 1;
                        printf("===========iState = %d\n", iState);
                        if(iState != MainBackProc::ST_ERR)
                        {
                            g_xSS.iVDBMode = 1;
                            SendGlobalMsg(MSG_VERIFY, VF_VIDEO_STOP, BACK_VDB, 0);
                        }
                    }
                }
                else if(pMsg->data2 == E_VDB_END)
                {
                    printf("pMsg->data2 == E_VDB_END  %d\n", g_xSS.iVDBMode);
                    if(g_xSS.iVDBMode == 1)
                    {
                        m_iUIWaiting = 0;
                        g_xSS.iVDBMode = 0;
                        SendGlobalMsg(MSG_VERIFY, VF_VIDEO_STOP, BACK_MAIN, 0);
                    }
                }
                else if(pMsg->data2 == HM_DETECTED)
                {
                    printf("pMsg->data2 == HM_DETECTED\n");
                    if(m_iCurScene == SCENE_SCREEN_SAVER)
                    {
                        ResetMultiVerify();

                        g_xSS.iNoSoundPlayFlag = 1;
                        g_xSS.iAdminMode = VERIFY_USER;
                        SendGlobalMsg(MSG_VERIFY, VF_FINISH, BACK_MAIN, 0);
                    }
                }
                else if(pMsg->data2 == E_MIC_START)
                {
                    printf("pMsg->data2 == E_MIC_START\n");
//                    if(g_xSS.iVDBMode == 1)
//                        AudioCommThread::Create();
                }
                else if(pMsg->data2 == E_MIC_END)
                {
                    printf("pMsg->data2 == E_MIC_END\n");
//                    AudioCommThread::Release();
                }

                pRokThread->SetKeyScanMode(this);
            }
        }
        else if(pMsg->type == MSG_RECOG_FACE && pMsg->data3 == m_pFaceRecogTask->GetCounter())
        {
            if(pMsg->data1 == FACE_TASK_FINISHED)
            {
                printf("FACE_TASK_FINISHED\n");
                int iResult = m_pFaceRecogTask->GetResult();
                if(iResult == FACE_RESULT_SUCCESS)
                {
                    printf("FACE_RESULT_SUCCESS\n");
                    fr_VerifyInit();
                    m_pFaceRecogTask->Pause();

                    int iFindIndex = m_pFaceRecogTask->GetRecogIndex();
                    int iFindID = -1;
                    PSMetaInfo pxMetaInfo = dbm_GetPersonMetaInfoByIndex(iFindIndex);
                    if(pxMetaInfo)
                        iFindID = pxMetaInfo->nID;

                    if(g_xSS.iAdminMode == VERIFY_MANAGER)
                    {
                        SaveLog(LOG_TYPE_FACE, LOG_INFO_SUCCESS, iFindID, -1);
                        PlayWelcomeSound();
                        SendGlobalMsg(MSG_VERIFY, VF_FINISH, BACK_MENU, 0);
                    }
                    else if(g_xCS.x.bAuthMode == AUTH_MODE_SINGLE)
                    {
                        m_iUIWaiting = 1;

                        PlayEffSuccess();
                        DrawVerifyScene(painter, xScreenImg.rect(), SCENE_SUCC);

                        MainBackProc::MotorOpenClose(pPWRUART, 0, 0, 0, g_xCS.x.bUnlockTime, 0);

                        SaveLog(LOG_TYPE_FACE, LOG_INFO_SUCCESS, iFindID, -1);
                        usleep(SCENE_SHOW_TIME * 1000);
                        m_pFaceRecogTask->Stop();
                        m_pCardRecogTask->Stop();

                        SetScene(SCENE_MAIN);
                        SendGlobalMsg(MSG_VERIFY, VF_MAIN_STARTED, 0, 0);
                    }
                    else if(g_xCS.x.bAuthMode == AUTH_MODE_MULTI)
                    {
                        if ((g_xSS.iFirstSuccIdx == 0 && g_xSS.iFirstPwdSuccCount == 0) || (g_xSS.iFirstSuccType == LOG_TYPE_FACE))
                        {
                            m_iUIWaiting = 1;

                            m_pFaceRecogTask->Stop();
                            m_pCardRecogTask->Stop();

                            {
                                FaceEngine::GetLastFaceImage(g_xSS.abFirstFaceData, &g_xSS.iFirstJpgLen);
                                PlayPickSound(0, 1);
                                ResetDetectTimeout();
                            }

                            g_xSS.iFirstSuccIdx = iFindID + 1;
                            g_xSS.iFirstSuccType = LOG_TYPE_FACE;

                            iResult = FACE_RESULT_NONE;
                            usleep(500 * 1000);

                            SetScene(SCENE_MAIN);
                            SendGlobalMsg(MSG_VERIFY, VF_MAIN_STARTED, 0, 0);
                        }
                        else
                        {
                            if (g_xSS.iFirstSuccIdx > 0 && g_xSS.iFirstSuccIdx - 1 != iFindID)
                            {
                                iResult = FACE_RESULT_FAILED;
                            }
                            else if (g_xSS.iFirstPwdSuccCount > 0)
                            {
                                int iIdx = 0;
                                for ( ; iIdx < g_xSS.iFirstPwdSuccCount ; iIdx ++)
                                {
                                    if (g_xSS.aiFirstPwdSuccIdArray[iIdx] == iFindID)
                                        break;
                                }
                                if (iIdx >= g_xSS.iFirstPwdSuccCount)
                                    iResult = FACE_RESULT_FAILED;
                            }

                            if (iResult != FACE_RESULT_FAILED)
                            {
                                m_iUIWaiting = 1;

                                PlayEffSuccess();
                                DrawVerifyScene(painter, xScreenImg.rect(), SCENE_SUCC);

                                MainBackProc::MotorOpenClose(pPWRUART, 0, 0, 0, g_xCS.x.bUnlockTime, 0);

                                SaveLog(LOG_TYPE_FACE, LOG_INFO_SUCCESS, iFindID, g_xSS.iFirstSuccType);
                                usleep(SCENE_SHOW_TIME * 1000);
                                m_pFaceRecogTask->Stop();
                                m_pCardRecogTask->Stop();

                                SetScene(SCENE_MAIN);
                                ResetMultiVerify();
                                SendGlobalMsg(MSG_VERIFY, VF_MAIN_STARTED, 0, 0);
                            }
                        }
                    }
                }

                if(iResult == FACE_RESULT_FAILED)
                {
                    printf("FACE_RESULT_FAILED\n");
                    m_iUIWaiting = 1;
                    m_pFaceRecogTask->Pause();
                    PlayEffFailed();
                    DrawVerifyScene(painter, xScreenImg.rect(), SCENE_FAILED);

                    if(g_xSS.iAdminMode == VERIFY_MANAGER || g_xCS.x.bAuthMode == AUTH_MODE_SINGLE)
                    {
                        SaveLog(LOG_TYPE_FACE, LOG_INFO_FAILED, -1, -1);
                    }
                    else
                    {
                        int iFirstSuccType = g_xSS.iFirstSuccType;
                        if (iFirstSuccType == LOG_TYPE_FACE)
                        {
                            iFirstSuccType = -1;
                            g_xSS.iFirstSuccType = LOG_TYPE_NONE;
                            g_xSS.iFirstSuccIdx = 0;
                        }

                        int iFirstUserIdx = g_xSS.iFirstSuccIdx - 1;
                        if (iFirstSuccType == LOG_TYPE_PASSCODE && g_xSS.iFirstPwdSuccCount > 0)
                            iFirstUserIdx = g_xSS.aiFirstPwdSuccIdArray[0];

                        SaveLog(LOG_TYPE_FACE, LOG_INFO_FAILED, iFirstUserIdx, iFirstSuccType);
                    }

                    usleep(SCENE_SHOW_TIME * 1000);
                    m_pFaceRecogTask->Stop();
                    m_pCardRecogTask->Stop();

                    SetScene(SCENE_MAIN);
                    ResetMultiVerify();
                    SendGlobalMsg(MSG_VERIFY, VF_MAIN_STARTED, 0, 0);
                }
                else if(iResult == FACE_RESULT_TIMEOUT)
                {
                    printf("FACE_RESULT_TIMEOUT\n");
                    m_iUIWaiting = 1;
                    SendGlobalMsg(MSG_VERIFY, VF_FINISH, BACK_SCREEN_SAVER, 0);
                }
            }
            else if(pMsg->data1 == FACE_TASK_DETECTED)
            {
                SendGlobalMsg(MSG_DRAW_LCD, 0, 0, 0);
            }
        }
        else if(pMsg->type == MSG_RECOG_CARD && pMsg->data3 == m_pCardRecogTask->GetCounter())
        {
            if(pMsg->data1 == CARD_TASK_FINISHED)
            {
                int iResult = m_pCardRecogTask->GetResult();
                if(iResult == CARD_RESULT_SUCCESS)
                {
                    int iFindID = m_pCardRecogTask->GetRecogID();
                    if(g_xSS.iAdminMode == VERIFY_MANAGER)
                    {
                        SaveLog(LOG_TYPE_CARD, LOG_INFO_SUCCESS, iFindID, -1);
                        PlayWelcomeSound();
                        SendGlobalMsg(MSG_VERIFY, VF_FINISH, BACK_MENU, 0);
                    }
                    else if(g_xCS.x.bAuthMode == AUTH_MODE_SINGLE)
                    {
                        m_iUIWaiting = 1;

                        PlayEffSuccess();
                        DrawVerifyScene(painter, xScreenImg.rect(), SCENE_SUCC);

                        MainBackProc::MotorOpenClose(pPWRUART, 0, 0, 0, g_xCS.x.bUnlockTime, 0);


                        SaveLog(LOG_TYPE_CARD, LOG_INFO_SUCCESS, iFindID, -1);
                        usleep(SCENE_SHOW_TIME * 1000);
                        m_pFaceRecogTask->Stop();
                        m_pCardRecogTask->Stop();

                        SetScene(SCENE_MAIN);
                        SendGlobalMsg(MSG_VERIFY, VF_MAIN_STARTED, 0, 0);
                    }
                    else if(g_xCS.x.bAuthMode == AUTH_MODE_MULTI)
                    {
                        if ((g_xSS.iFirstSuccIdx == 0 && g_xSS.iFirstPwdSuccCount == 0) || (g_xSS.iFirstSuccType == LOG_TYPE_CARD))
                        {
                            m_pFaceRecogTask->Stop();
                            m_pCardRecogTask->Stop();

                            {
                                PlayPickSound(0, 1);
                                ResetDetectTimeout();
                            }

                            g_xSS.iFirstSuccIdx = iFindID + 1;
                            g_xSS.iFirstSuccType = LOG_TYPE_CARD;

                            iResult = CARD_RESULT_NONE;
                            usleep(500 * 1000);

                            m_pFaceRecogTask->Start();
                            m_pCardRecogTask->Start();
                        }
                        else
                        {
                            if (g_xSS.iFirstSuccIdx > 0 && g_xSS.iFirstSuccIdx - 1 != iFindID)
                            {
                                iResult = CARD_RESULT_FAILED;
                            }
                            else if (g_xSS.iFirstPwdSuccCount > 0)
                            {
                                int iIdx = 0;
                                for ( ; iIdx < g_xSS.iFirstPwdSuccCount ; iIdx ++)
                                {
                                    if (g_xSS.aiFirstPwdSuccIdArray[iIdx] == iFindID)
                                        break;
                                }
                                if (iIdx >= g_xSS.iFirstPwdSuccCount)
                                    iResult = CARD_RESULT_FAILED;
                            }

                            if (iResult != CARD_RESULT_FAILED)
                            {
                                m_iUIWaiting = 1;
                                PlayEffSuccess();
                                DrawVerifyScene(painter, xScreenImg.rect(), SCENE_SUCC);

                                MainBackProc::MotorOpenClose(pPWRUART, 0, 0, 0, g_xCS.x.bUnlockTime, 0);

                                SaveLog(LOG_TYPE_CARD, LOG_INFO_SUCCESS, iFindID, g_xSS.iFirstSuccType);
                                usleep(SCENE_SHOW_TIME * 1000);
                                m_pFaceRecogTask->Stop();
                                m_pCardRecogTask->Stop();

                                SetScene(SCENE_MAIN);
                                ResetMultiVerify();
                                SendGlobalMsg(MSG_VERIFY, VF_MAIN_STARTED, 0, 0);
                            }
                        }
                    }
                    m_iCounter ++;
                }

                if(iResult == CARD_RESULT_FAILED)
                {
                    printf("CARD_RESULT_FAILED\n");
                    m_iUIWaiting = 1;
                    m_pFaceRecogTask->Pause();
                    PlayEffFailed();
                    DrawVerifyScene(painter, xScreenImg.rect(), SCENE_FAILED);

                    if(g_xSS.iAdminMode == VERIFY_MANAGER || g_xCS.x.bAuthMode == AUTH_MODE_SINGLE)
                    {
                        SaveLog(LOG_TYPE_CARD, LOG_INFO_FAILED, -1, -1);
                    }
                    else
                    {
                        int iFirstSuccType = g_xSS.iFirstSuccType;
                        if (iFirstSuccType == LOG_TYPE_CARD)
                        {
                            iFirstSuccType = -1;
                            g_xSS.iFirstSuccType = LOG_TYPE_NONE;
                            g_xSS.iFirstSuccIdx = 0;
                        }

                        int iFirstUserIdx = g_xSS.iFirstSuccIdx - 1;
                        if (iFirstSuccType == LOG_TYPE_PASSCODE && g_xSS.iFirstPwdSuccCount > 0)
                            iFirstUserIdx = g_xSS.aiFirstPwdSuccIdArray[0];

                        SaveLog(LOG_TYPE_CARD, LOG_INFO_FAILED, iFirstUserIdx, iFirstSuccType);
                    }

                    usleep(SCENE_SHOW_TIME * 1000);
                    m_pFaceRecogTask->Stop();
                    m_pCardRecogTask->Stop();

                    SetScene(SCENE_MAIN);
                    ResetMultiVerify();
                    SendGlobalMsg(MSG_VERIFY, VF_MAIN_STARTED, 0, 0);
                }
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
            if(m_iCurScene == SCENE_MAIN)
                DrawVerifyScene(painter, xScreenImg.rect(), SCENE_MAIN);
        }
        else if(pMsg->type == MSG_BUTTON_CLICKED)
        {            
            qDebug() << "MSG_BUTTON_CLICKED" << m_iCurScene << m_iCounter;

            if(pMsg->data1 == BTN_ID_ANY)
            {
                if(m_iCurScene == SCENE_SCREEN_SAVER)
                {
                    SendGlobalMsg(MSG_VERIFY, VF_FINISH, BACK_MAIN, 0);
                }
                if(m_iCurScene == SCENE_MAIN)
                {
                    
                    {
                        if(m_iUIWaiting == 0)
                        {
                            m_iUIWaiting = 1;
                            SendGlobalMsg(MSG_VERIFY, VF_FINISH, BACK_MENU, 0);
                        }
                    }
                }
            }
            else if(pMsg->data1 == BTN_ID_SETTINGS)
            {
                if(dbm_GetManagerCount() == 0)
                {
                    SendGlobalMsg(MSG_VERIFY, VF_FINISH, BACK_MENU, 0);
                }
                else
                {
                    if(m_iUIWaiting == 0)
                    {
                        m_iUIWaiting = 1;

                        m_pFaceRecogTask->Stop();
                        m_pCardRecogTask->Stop();

                        if(g_xSS.iAdminMode == VERIFY_USER)
                        {
                            g_xSS.iAdminMode = VERIFY_MANAGER;
                            DrawVerifyScene(painter, xScreenImg.rect(), SCENE_MAIN);

                            PlayManagerSound();
                        }
                        else
                            g_xSS.iAdminMode = VERIFY_USER;

                        SetScene(SCENE_MAIN);
                        ResetMultiVerify();
                        SendGlobalMsg(MSG_VERIFY, VF_MAIN_STARTED, 0, 0);
                    }
                }
            }
        }
        else if(pMsg->type == MSG_BUTTON_UPDATE)
        {

        }
        message_queue_message_free(&g_worker, (void*)pMsg);
    }

    painter.end();

    if(pMsg != NULL)
        message_queue_message_free(&g_worker, (void*)pMsg);

    message_queue_destroy(&g_worker);

    m_pFaceRecogTask->Stop();
    delete m_pFaceRecogTask;
    m_pFaceRecogTask = NULL;

    m_pCardRecogTask->Stop();
    delete m_pCardRecogTask;
    m_pCardRecogTask = NULL;

    pRokThread->SetKeyScanMode(NULL);

    SigBack(BACK_MENU, 0);    
    m_iCounter ++;
    g_xSS.iVDBStart = 0;
}

void VerifyFaceForm::mousePressEvent(QMouseEvent* e)
{
    QWidget::mousePressEvent(e);
    printf("\t\t\t\t\tmousePressEvent11 %f\n", Now());

    int iPressed = CheckBtnState(e->pos(), TOUCH_PRESS);
    if(iPressed == -1)
        return;

    printf("SendGlobal: %d\n", m_iCounter);
    SendGlobalMsg(MSG_BUTTON_CLICKED, iPressed, BTN_STATE_NONE, m_iCounter);
}

void VerifyFaceForm::timerEvent(QTimerEvent* e)
{
    if(m_iCurScene == SCENE_PASSWORD && Now() - m_rClickTime > g_xCS.x.bScreenSaverTime * 60 * 1000)
    {
        if(m_iUIWaiting == 0)
        {
            m_iUIWaiting = 1;
            SendGlobalMsg(MSG_VERIFY, VF_FINISH, BACK_SCREEN_SAVER, 0);
        }
    }
    else if(m_iCurScene == SCENE_VDB)
    {
       static int iKeyHaltCount = 0;
       if(iKeyHaltCount % 5 == 0)
       {
            MainWindow* w = (MainWindow*)m_pParentView;
            UARTTask* pBackUART = w->GetUART(BACK_BOARD_PORT);
            if(pBackUART)
                MainBackProc::KeyHaltTime(pBackUART, 7);
       }
       iKeyHaltCount ++;

       if(g_xSS.iVDBError >= 2)
       {
           printf("g_xSS.iVDBError=================\n");
           g_xSS.iVDBError = 0;
       }
    }

    QDateTime xCurTime = DATETIME_32ToQDateTime(dbm_GetCurDateTime());
    QString strDate = GetDateStr(xCurTime.date());
    QString strTime;
    strTime.sprintf("%d:%02d", xCurTime.time().hour(), xCurTime.time().minute());

    QStringList vWeekDays;
    vWeekDays.append("");
    vWeekDays.append(StringTable::Str_Monday);
    vWeekDays.append(StringTable::Str_Tuesday);
    vWeekDays.append(StringTable::Str_Wednesday);
    vWeekDays.append(StringTable::Str_Thursday);
    vWeekDays.append(StringTable::Str_Friday);
    vWeekDays.append(StringTable::Str_Saturday);
    vWeekDays.append(StringTable::Str_Sunday);

    ui->lblTime->setText(strTime);
    ui->lblDate->setText(strDate + " " + vWeekDays[xCurTime.date().dayOfWeek()]);

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* pRokThread = w->GetROK();
    pRokThread->InitTime();

    SendGlobalMsg(MSG_VERIFY, VF_LOOP, 0, 0);
}

void VerifyFaceForm::RetranslateUI()
{
    QString strPasscode;
    for(int i = 0; i < m_strPasscode.size(); i ++)
        strPasscode += QString::fromUtf8("✱");
    ui->lblPasscode1->setText(strPasscode);

    if(g_xSS.iAdminMode == VERIFY_USER)
    {
        QString strSheets;
        strSheets.sprintf("color: rgb(255, 255, 255);");
        ui->lblTitle->setStyleSheet(strSheets);
        ui->lblTitle->setText(StringTable::Str_Enter_Password);
    }
    else
    {
        QString strSheets;
        strSheets.sprintf("color: rgb(0, 255, 0);");
        ui->lblTitle->setStyleSheet(strSheets);
        ui->lblTitle->setText(StringTable::Str_ManagerMode);
    }

    ui->btnOk->setText("OK");

    ui->btn0->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn1->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn2->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn3->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn4->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn5->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn6->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn7->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn8->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn9->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btnDel->setButtonColors(g_UITheme->btnDelColor);
    ui->btnOk->setButtonColors(QColor(255, 64, 129));
}

void VerifyFaceForm::DrawVerifyScene(QPainter& painter, QRect xScreenRect, int iScene)
{
    LCDTask::LCD_MemClear(0);
    painter.fillRect(xScreenRect, Qt::NoBrush);

    if(iScene == SCENE_SUCC)
    {
        painter.fillRect(xScreenRect, Qt::black);

        QImage iSceneIcon(":/icons/ic_scene_succ.png");
        painter.drawImage(QPoint((xScreenRect.width() - iSceneIcon.width()) / 2, (xScreenRect.height() - iSceneIcon.height()) / 2), iSceneIcon);
    }
    else if(iScene == SCENE_FAILED)
    {
        painter.fillRect(xScreenRect, Qt::black);

        QImage iSceneIcon(":/icons/ic_scene_failed.png");
        painter.drawImage(QPoint((xScreenRect.width() - iSceneIcon.width()) / 2, (xScreenRect.height() - iSceneIcon.height()) / 2), iSceneIcon);
    }
    else if(iScene == SCENE_MAIN)
    {
        if(g_iIRCamInited < 0)
            painter.fillRect(QRect(0, 0, MAX_X, MAX_Y), Qt::black);

        if(g_xSS.iShowIrCamera == 0)
        {
            LCDTask::DrawMemFace(g_xSS.xFaceRect.x, g_xSS.xFaceRect.y, g_xSS.xFaceRect.width, g_xSS.xFaceRect.height, COLOR_GREEN, 1, g_xPS.x.iCamOffX, g_xPS.x.iCamOffY);
        }
        else
        {
            LCDTask::DrawMemFace(g_xSS.xFaceRect.x, g_xSS.xFaceRect.y, g_xSS.xFaceRect.width, g_xSS.xFaceRect.height, COLOR_GREEN, 0, 0, 0);
        }

#if (SCREEN_MODE == 1)
        painter.fillRect(QRect(0, 0, MAX_X, LCD_HEADER_HEIGHT), QColor::fromRgba(LCD_HEADER_COLOR));
#else
        painter.fillRect(QRect(0, 0, 480, LCD_HEADER_HEIGHT * 1.5), QColor::fromRgba(LCD_HEADER_COLOR));
#endif

        painter.save();
        painter.setPen(Qt::green);

        if(dbm_GetPersonCount() == 0)
        {
#if (SCREEN_MODE)
            painter.setFont(g_UITheme->TitleFont);
            painter.drawText(QRect(0, 0, MAX_X, LCD_HEADER_HEIGHT - 5), Qt::AlignBottom | Qt::AlignHCenter, StringTable::Str_Please_register_new_user);
#else
            QFont f = g_UITheme->TitleFont;
            f.setPointSize(13);
            painter.setFont(f);
            painter.drawText(QRect(0, 0, 480, LCD_HEADER_HEIGHT * 1.5 - 10), Qt::AlignBottom | Qt::AlignHCenter, StringTable::Str_Please_register_new_user);
#endif
        }
        else if(g_xSS.iAdminMode == VERIFY_MANAGER)
        {
#if (SCREEN_MODE)
            painter.setFont(g_UITheme->TitleFont);
            painter.drawText(QRect(0, 0, MAX_X, LCD_HEADER_HEIGHT - 5), Qt::AlignBottom | Qt::AlignHCenter, StringTable::Str_ManagerMode);
#else
            QFont f = g_UITheme->TitleFont;
            f.setPointSize(12);
            painter.setFont(f);
            painter.drawText(QRect(0, 0, 480, LCD_HEADER_HEIGHT * 1.5 - 10), Qt::AlignBottom | Qt::AlignHCenter, StringTable::Str_ManagerMode);
#endif
        }
        else if(g_xCS.x.bAuthMode == 1)
        {
#if (SCREEN_MODE)
            painter.setFont(g_UITheme->TitleFont);
            painter.drawText(QRect(0, 0, MAX_X, LCD_HEADER_HEIGHT - 5), Qt::AlignBottom | Qt::AlignHCenter, StringTable::Str_Multi_Verify);
#else
            QFont f = g_UITheme->TitleFont;
            f.setPointSize(12);
            painter.setFont(f);
            painter.drawText(QRect(0, 0, 480, LCD_HEADER_HEIGHT * 1.5 - 10), Qt::AlignBottom | Qt::AlignHCenter, StringTable::Str_Multi_Verify);
#endif
        }
        else
        {
#if (SCREEN_MODE)
            painter.setFont(g_UITheme->TitleFont);
            painter.drawText(QRect(0, 0, MAX_X, LCD_HEADER_HEIGHT - 5), Qt::AlignBottom | Qt::AlignHCenter, StringTable::Str_Verify_User);
#else
            QFont f = g_UITheme->TitleFont;
            f.setPointSize(12);
            painter.setFont(f);
            painter.drawText(QRect(0, 0, 480, LCD_HEADER_HEIGHT * 1.5 - 10), Qt::AlignBottom | Qt::AlignHCenter, StringTable::Str_Verify_User);
#endif
        }
        painter.restore();

        DrawDateTime(painter);
        DrawButtons(painter);

        if(g_iIRCamInited < 0)
        {
            painter.save();
            painter.setFont(g_UITheme->PrimaryFont);
            painter.setPen(Qt::red);
            painter.drawText(QPoint(20, 220), StringTable::Str_Camera_Error);
            painter.restore();
        }
    }

    LCDTask::LCD_Update();
}

void VerifyFaceForm::DrawButtons(QPainter& painter)
{
    for(int i = 0; i < m_iBtnCount; i ++)
    {
        QRect xBtnRect(QPoint(m_axBtns[i].iX1, m_axBtns[i].iY1), QPoint(m_axBtns[i].iX2, m_axBtns[i].iY2));
#if (SCREEN_MODE == 0)
        xBtnRect = QRect(QPoint(m_axBtns[i].iX1 * 1.5f, m_axBtns[i].iY1 * 1.5f), QPoint(m_axBtns[i].iX2 * 1.5f, m_axBtns[i].iY2 * 1.5f));
#endif
        QImage xIconImg(m_axBtns[i].szNormal);
        if(!xIconImg.isNull())
            painter.drawImage(xBtnRect, xIconImg);
    }
}

void VerifyFaceForm::DrawDateTime(QPainter& painter)
{
    QDateTime xCurTime = DATETIME_32ToQDateTime(dbm_GetCurDateTime());
    QString strDate = GetDateStr(xCurTime.date());
    QString strTime = GetTimeStr(xCurTime.time());

    QString strDateTime = strDate + " " + strTime;

    painter.save();
    painter.setPen(Qt::white);
    QFont f = painter.font();
#if (SCREEN_MODE)    
    f.setPointSize(15);
#else
    f.setPointSize(9);
#endif
    painter.setFont(g_UITheme->SecondaryFont);
    painter.drawText(QRect(10, 0, MAX_X, 20), Qt::AlignVCenter | Qt::AlignLeft, strDateTime);
    painter.restore();
}

void VerifyFaceForm::ResetButtons()
{
    m_xMutex.lock();
    m_iBtnCount = 0;
    memset(m_axBtns, 0, sizeof(m_axBtns));
    m_xMutex.unlock();
}

void VerifyFaceForm::AddButton(int iID, int iX1, int iY1, int iX2, int iY2, const char* szNormal, const char* szPress, unsigned int iNormalColor, int iPressColor, int iState)
{
    if(m_iBtnCount > MAX_BUTTON_CNT)
        return;

    BUTTON xBtn;
    memset(&xBtn, 0, sizeof(xBtn));
    xBtn.iID = iID;
    xBtn.iX1 = iX1;
    xBtn.iY1 = iY1;
    xBtn.iX2 = iX2;
    xBtn.iY2 = iY2;

    if(szNormal)
        strcpy(xBtn.szNormal, szNormal);

    if(szPress)
        strcpy(xBtn.szPress, szPress);

    xBtn.iNormalColor = iNormalColor;
    xBtn.iPressColor = iPressColor;

    m_xMutex.lock();

    int iExist = -1;
    for(int i = 0; i < m_iBtnCount; i ++)
    {
        if(m_axBtns[i].iID == xBtn.iID)
        {
            iExist = i;
            break;
        }
    }

    if(iExist >= 0)
    {
        m_axBtns[iExist] = xBtn;
        m_xMutex.unlock();
        return;
    }

    xBtn.iState = iState;
    m_axBtns[m_iBtnCount] = xBtn;
    m_iBtnCount ++;
    m_xMutex.unlock();
}

int VerifyFaceForm::CheckBtnState(QPoint pos, int mode)
{
    int iPressed = -1;
    for(int i = 0; i < m_iBtnCount; i ++)
    {
        QRect xBtnRect(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2 - m_axBtns[i].iX1 + 1, m_axBtns[i].iY2 - m_axBtns[i].iY1 + 1);
        if(xBtnRect.contains(pos))
        {
            iPressed = m_axBtns[i].iID;
            if(mode == TOUCH_RELEASE)
            {
                m_axBtns[i].iState = BTN_STATE_NONE;
            }
            else
                m_axBtns[i].iState = BTN_STATE_PRESSED;

            break;
        }
        else
            m_axBtns[i].iState = BTN_STATE_NONE;
    }

    return iPressed;
}


void VerifyFaceForm::SetScene(int iScene)
{
    if(iScene == SCENE_MAIN)
    {
        ResetButtons();       
        AddButton(BTN_ID_SETTINGS, 272, 0, 319, 47, ":/icons/ic_settings_1.png", NULL, 0, 0);
        AddButton(BTN_ID_ANY, 0, 0, MAX_X, MAX_Y, NULL, NULL, 0, 0);

        printf("uuuuuuuuuuuuuuuuuuuuu4\n");
        camera_set_irled(IR_CAM, g_xEP.iLedStatus, 0);

        m_pFaceRecogTask->Start();
        m_pCardRecogTask->Start();
        SendGlobalMsg(MSG_DRAW_LCD, 0, 0, 0);
    }
    else if(iScene == SCENE_VDB)
    {
        ResetButtons();

        if(g_xSS.iShowIrCamera > 0)
            camera_set_irled(IR_CAM, g_xEP.iLedStatus, 0);

        m_pFaceRecogTask->Start(1);

        SendGlobalMsg(MSG_DRAW_LCD, 0, 0, 0);
    }
    else if(iScene == SCENE_SCREEN_SAVER)
    {
        ResetButtons();
        AddButton(BTN_ID_ANY, 0, 0, MAX_X, MAX_Y, NULL, NULL, 0, 0);

        SetUI(UI_SCREEN_SAVER);
        RetranslateUI();
    }

//    m_iCurScene = iScene;
    m_iCurScene = SCENE_NONE;
    printf("-================Cur Scene: %d\n", m_iCurScene);
}

void VerifyFaceForm::SetUI(int iIdx, int iParam0)
{
    if(iIdx == UI_NONE)
    {
        ui->stackedWidget->setCurrentIndex(1);
        SetBGColor(Qt::black);
    }
    else if(iIdx == UI_PASSWORD)
    {
        m_vPasscodeMaps.clear();
        m_vPasscodeBtns.clear();

        m_vPasscodeBtns.append(ui->btn0);
        m_vPasscodeBtns.append(ui->btn1);
        m_vPasscodeBtns.append(ui->btn2);
        m_vPasscodeBtns.append(ui->btn3);
        m_vPasscodeBtns.append(ui->btn4);
        m_vPasscodeBtns.append(ui->btn5);
        m_vPasscodeBtns.append(ui->btn6);
        m_vPasscodeBtns.append(ui->btn7);
        m_vPasscodeBtns.append(ui->btn8);
        m_vPasscodeBtns.append(ui->btn9);

        for(int i = 0; i < 10; i ++)
            m_vPasscodeMaps.append(-1);

        for(int i = 0; i < 10; i ++)
        {
            int randVal = rand() % (10 - i);

            int index = 0;
            int j = 0;
            for(j = 0; j < 10; j ++)
            {
                if(m_vPasscodeMaps[j] == -1)
                {
                    if(index == randVal)
                        break;

                    index ++;
                }
            }

            m_vPasscodeMaps[j] = i;
        }

        ui->stackedWidget->setCurrentIndex(0);
        SetBGColor(g_UITheme->mainBgColor);
    }
    
    else if(iIdx == UI_VDB)
    {
        ui->lblResult->setPixmap(QPixmap(":/icons/ic_vdb.png"));
        ui->stackedWidget->setCurrentIndex(3);
        SetBGColor(Qt::black);
    }
    else if(iIdx == UI_SCREEN_SAVER)
    {        
        ui->stackedWidget->setCurrentIndex(2);
        SetBGColor(Qt::black);
    }
}

void VerifyFaceForm::SetupUI()
{
    ui = new Ui::PasscodeForm;
    ui->setupUi(this);

    ui->btnBack->SetImages(QImage(":/icons/ic_arrow_back.png"));
    ui->btnSettings->SetImages(QImage(":/icons/ic_settings_1.png"));
    ui->lblTitle->setFont(g_UITheme->TitleFont);
    ui->lblPasscode1->setFont(g_UITheme->TitleFont);

    ui->btn0->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn1->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn2->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn3->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn4->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn5->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn6->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn7->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn8->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btn9->setButtonColors(g_UITheme->itemNormalBgColor);
    ui->btnDel->setButtonColors(g_UITheme->btnDelColor);
    ui->btnOk->setButtonColors(QColor(255, 64, 129));

    ui->btn0->setFont(g_UITheme->TitleFont);
    ui->btn1->setFont(g_UITheme->TitleFont);
    ui->btn2->setFont(g_UITheme->TitleFont);
    ui->btn3->setFont(g_UITheme->TitleFont);
    ui->btn4->setFont(g_UITheme->TitleFont);
    ui->btn5->setFont(g_UITheme->TitleFont);
    ui->btn6->setFont(g_UITheme->TitleFont);
    ui->btn7->setFont(g_UITheme->TitleFont);
    ui->btn8->setFont(g_UITheme->TitleFont);
    ui->btn9->setFont(g_UITheme->TitleFont);
    ui->btnDel->setFont(g_UITheme->TitleFont);
    ui->btnOk->setFont(g_UITheme->TitleFont);

    QFont f = g_UITheme->TitleFont;
    f.setPointSize(10);
    ui->lblTime->setFont(f);

    SetUI(UI_NONE);
    RetranslateUI();
}

void  VerifyFaceForm::Click0()
{
    printf("VerifyFaceForm::Click0\n");
    PlaySoundTop();
    AddPasscode('0' + m_vPasscodeMaps[0]);
}

void  VerifyFaceForm::Click1()
{
    printf("VerifyFaceForm::Click1\n");
    PlaySoundTop();


    AddPasscode('0' + m_vPasscodeMaps[1]);
}

void VerifyFaceForm::Click2()
{
    printf("VerifyFaceForm::Click2\n");
    PlaySoundTop();

    AddPasscode('0' + m_vPasscodeMaps[2]);
}

void VerifyFaceForm::Click3()
{
    printf("VerifyFaceForm::Click3\n");
    PlaySoundTop();


    AddPasscode('0' + m_vPasscodeMaps[3]);
}

void VerifyFaceForm::Click4()
{
    PlaySoundTop();

    AddPasscode('0' + m_vPasscodeMaps[4]);
}

void VerifyFaceForm::Click5()
{
    PlaySoundTop();


    AddPasscode('0' + m_vPasscodeMaps[5]);
}

void VerifyFaceForm::Click6()
{
    PlaySoundTop();

    AddPasscode('0' + m_vPasscodeMaps[6]);
}

void VerifyFaceForm::Click7()
{
    PlaySoundTop();

    AddPasscode('0' + m_vPasscodeMaps[7]);
}

void VerifyFaceForm::Click8()
{
    PlaySoundTop();

    AddPasscode('0' + m_vPasscodeMaps[8]);
}

void VerifyFaceForm::Click9()
{
    PlaySoundTop();

    AddPasscode('0' + m_vPasscodeMaps[9]);
}

void VerifyFaceForm::DelClick()
{
    PlaySoundTop();

    m_rResetTime = Now();
    m_rClickTime = m_rResetTime;
    m_strPasscode.clear();
    RetranslateUI();
}

void VerifyFaceForm::ClickBack()
{
    if(m_iUIWaiting == 1)
        return;

    SetUI(UI_NONE);
    SendGlobalMsg(MSG_VERIFY, VF_VIDEO_START, 0, 0);
}

void VerifyFaceForm::SettingsClick()
{
    if(m_iUIWaiting == 1)
        return;

    if(dbm_GetManagerCount())
    {
        if(g_xSS.iAdminMode == VERIFY_USER)
        {
            m_iUIWaiting = 1;
            m_strPasscode.clear();
            g_xSS.iAdminMode = VERIFY_MANAGER;
            ConnectButtons(0);
            RetranslateUI();

            SendGlobalMsg(MSG_VERIFY, VF_PASSWORD_STARTED, 2, 0);
        }
        else
        {
            m_strPasscode.clear();
            g_xSS.iAdminMode = VERIFY_USER;
            RetranslateUI();
        }
    }
    else
    {
        m_iUIWaiting = 1;
        SetUI(UI_NONE);
        SendGlobalMsg(MSG_VERIFY, VF_FINISH, BACK_MENU, 0);
    }
}

void VerifyFaceForm::AddPasscode(QChar passcode)
{
    m_rClickTime = Now();
    if(m_strPasscode.length() >= 30)
        return;

    m_strPasscode.append(passcode);
    RetranslateUI();
}

void VerifyFaceForm::OkClick()
{    
    m_rClickTime = Now();
    PlaySoundTop();

    char szPasscode[256] = { 0 };
    strcpy(szPasscode, m_strPasscode.toUtf8().data());

    if(strlen(szPasscode) == 0)
    {
        if(m_rResetTime != 0 && Now() - m_rResetTime < 1000)
            SettingsClick();

        return;
    }

    m_iUIWaiting = 1;
    int iFindID = -1;
    for(int i = 0; i < dbm_GetPersonCount(); i ++)
    {
        PSMetaInfo pxMetaInfo = dbm_GetPersonMetaInfoByIndex(i);
        if(pxMetaInfo == NULL)
            continue;

        if(g_xSS.iAdminMode == VERIFY_MANAGER && pxMetaInfo->fPrivilege != EMANAGER)
            continue;

        if(strlen(szPasscode) == 0)
            continue;
    }

    if(iFindID >= 0)
    {
        if(g_xSS.iAdminMode != VERIFY_MANAGER)
        {
            SetUI(UI_RESULT, iFindID);
        }

        m_iCurScene = SCENE_NONE;
        SendGlobalMsg(MSG_VERIFY, VF_PASSWORD_RESULT, iFindID, m_iCounter);
    }
    else
    {
        if (g_xCS.x.bAuthMode == AUTH_MODE_MULTI)
        {
            if (g_xSS.iFirstPwdSuccCount > 0)
            {
                g_xSS.iFirstSuccType = LOG_TYPE_PASSCODE;
                PlayPickSound(0, 1);

                m_iCurScene = SCENE_NONE;
                SetUI(UI_NONE);
                SendGlobalMsg(MSG_VERIFY, VF_START, 0, 0);
                return;
            }
        }

        m_iCurScene = SCENE_NONE;
        SetUI(UI_RESULT, -1);
        SendGlobalMsg(MSG_VERIFY, VF_PASSWORD_RESULT, -1, m_iCounter);
    }
}

void VerifyFaceForm::ConnectButtons(int iConnect)
{
    if(iConnect)
    {
        connect(ui->btn0, SIGNAL(clicked()), this, SLOT(Click0()));
        connect(ui->btn1, SIGNAL(clicked()), this, SLOT(Click1()));
        connect(ui->btn2, SIGNAL(clicked()), this, SLOT(Click2()));
        connect(ui->btn3, SIGNAL(clicked()), this, SLOT(Click3()));
        connect(ui->btn4, SIGNAL(clicked()), this, SLOT(Click4()));
        connect(ui->btn5, SIGNAL(clicked()), this, SLOT(Click5()));
        connect(ui->btn6, SIGNAL(clicked()), this, SLOT(Click6()));
        connect(ui->btn7, SIGNAL(clicked()), this, SLOT(Click7()));
        connect(ui->btn8, SIGNAL(clicked()), this, SLOT(Click8()));
        connect(ui->btn9, SIGNAL(clicked()), this, SLOT(Click9()));
        connect(ui->btnOk, SIGNAL(clicked()), this, SLOT(OkClick()));
        connect(ui->btnDel, SIGNAL(clicked()), this, SLOT(DelClick()));
        connect(ui->btnBack, SIGNAL(clicked()), this, SLOT(ClickBack()));
        connect(ui->btnSettings, SIGNAL(clicked()), this, SLOT(SettingsClick()));

        ui->page->setEnabled(true);
    }
    else
    {
        disconnect(ui->btn0, SIGNAL(clicked()), this, SLOT(Click0()));
        disconnect(ui->btn1, SIGNAL(clicked()), this, SLOT(Click1()));
        disconnect(ui->btn2, SIGNAL(clicked()), this, SLOT(Click2()));
        disconnect(ui->btn3, SIGNAL(clicked()), this, SLOT(Click3()));
        disconnect(ui->btn4, SIGNAL(clicked()), this, SLOT(Click4()));
        disconnect(ui->btn5, SIGNAL(clicked()), this, SLOT(Click5()));
        disconnect(ui->btn6, SIGNAL(clicked()), this, SLOT(Click6()));
        disconnect(ui->btn7, SIGNAL(clicked()), this, SLOT(Click7()));
        disconnect(ui->btn8, SIGNAL(clicked()), this, SLOT(Click8()));
        disconnect(ui->btn9, SIGNAL(clicked()), this, SLOT(Click9()));
        disconnect(ui->btnOk, SIGNAL(clicked()), this, SLOT(OkClick()));
        disconnect(ui->btnDel, SIGNAL(clicked()), this, SLOT(DelClick()));
        disconnect(ui->btnBack, SIGNAL(clicked()), this, SLOT(ClickBack()));
        disconnect(ui->btnSettings, SIGNAL(clicked()), this, SLOT(SettingsClick()));

        ui->page->setEnabled(false);
    }
}


void VerifyFaceForm::SaveLog(int iLogType, int iResult, int iFindID, int iFistSuccType)
{
    int iPrivilege = 0;
    char szName[256] = { 0 };
    DATETIME_32 xNow = dbm_GetCurDateTime();

    printf("iFistSuccType = %d\n", iFistSuccType);

    if(iLogType == LOG_TYPE_FACE || iFistSuccType >= 0)
    {
        PSMetaInfo pxMetaInfo = dbm_GetPersonMetaInfoByID(iFindID);
        if(pxMetaInfo)
        {
            strcpy(szName, pxMetaInfo->szName);
            iPrivilege = pxMetaInfo->fPrivilege;
        }

        FaceEngine::GetLastFaceImage(g_xSS.abFirstFaceData, &g_xSS.iFirstJpgLen);
        int iAuthMode = g_xSS.iAdminMode;
        if(iAuthMode == VERIFY_USER && g_xCS.x.bAuthMode == AUTH_MODE_MULTI)
            iAuthMode = VERIFY_MULTI;

        dbm_AddLog(iFindID, szName, iAuthMode, iResult, iLogType, g_xSS.iFirstJpgLen, g_xSS.abFirstFaceData, xNow, 0, iFistSuccType);
    }
    else
    {
        int iAuthMode = g_xSS.iAdminMode;
        if(iAuthMode == VERIFY_USER && g_xCS.x.bAuthMode == AUTH_MODE_MULTI)
            iAuthMode = VERIFY_MULTI;

        PSMetaInfo pxMetaInfo = dbm_GetPersonMetaInfoByID(iFindID);
        if(pxMetaInfo)
        {
            strcpy(szName, pxMetaInfo->szName);
            iPrivilege = pxMetaInfo->fPrivilege;
            g_xSS.iFirstJpgLen = pxMetaInfo->iImageLen;
            memcpy(g_xSS.abFirstFaceData, pxMetaInfo->abFaceImage, sizeof(pxMetaInfo->abFaceImage));
            dbm_AddLog(iFindID, szName, iAuthMode, iResult, iLogType, g_xSS.iFirstJpgLen, g_xSS.abFirstFaceData, xNow, 0, iFistSuccType);
        }
        else
            dbm_AddLog(iFindID, szName, iAuthMode, iResult, iLogType, 0, NULL, xNow, 0, iFistSuccType);

    }
}

void VerifyFaceForm::ResetMultiVerify()
{
    g_xSS.iFirstSuccIdx = g_xSS.iFirstPwdSuccCount = 0;
    g_xSS.iFirstSuccType = LOG_TYPE_NONE;
}

bool VerifyFaceForm::event(QEvent* e)
{
    if(e->type() == EV_KEY_EVENT)
    {
        KeyEvent* pEvent = static_cast<KeyEvent*>(e);
        if(pEvent != NULL)
        {
            if(pEvent->m_iKeyID == HM_BELL_KEY)
            {
                printf("HM_BELL_KEY  %d, %d\n", pEvent->m_iEvType, m_iUIWaiting);
                SendGlobalMsg(MSG_VERIFY, VF_EVENT, pEvent->m_iKeyID, pEvent->m_iEvType == KeyEvent::EV_PRESSED? 0 : 1);
            }
            else if(pEvent->m_iKeyID == HM_DETECTED)
            {
                printf("HM_DETECTED\n");
                SendGlobalMsg(MSG_VERIFY, VF_EVENT, pEvent->m_iKeyID, 0);
            }
            else if(pEvent->m_iKeyID == E_VDB_END)
            {
                printf("E_VDB_END\n");
                SendGlobalMsg(MSG_VERIFY, VF_EVENT, pEvent->m_iKeyID, 0);
            }
            else
            {
                SendGlobalMsg(MSG_VERIFY, VF_EVENT, pEvent->m_iKeyID, 0);
            }
        }
    }

    return QWidget::event(e);
}
