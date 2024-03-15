#include "lcdtestform.h"
#include "appdef.h"
#include "i2cbase.h"
#include "settings.h"
#include <unistd.h>
#include "uitheme.h"
#include "shared.h"
#include "mototestform.h"
#include "camera_api.h"
#include "camtestform.h"
#include "enrollcardform.h"
#include "drv_gpio.h"
#include "mainwindow.h"
#include "rokthread.h"
#include "fptask.h"
#include "enrollfpform.h"
#include "touchtestform.h"
#include "irtestform.h"
#include "qrencode.h"
#include "qqrencode.h"
#include "DBManager.h"
#include "lcdtask.h"
#include "stringtable.h"
#include "mainbackproc.h"
#include "soundbase.h"
#include "playthread.h"
#include "themedef.h"
#include "uarttask.h"
#include "uartcomm.h"

#include "faceengine.h"
#include "mount_fs.h"
#include "base.h"
#include "countbase.h"

#include <QtGui>
#include <termio.h>

int g_iWifiTest = 0;

extern UARTTask* g_pUartTask;

LcdTestForm::LcdTestForm(QGraphicsView *view, FormBase* parentForm) :
    FormBase(view, parentForm),
    m_label(this),
    m_testThread(this)
{
    m_testIndex = 0;
    SetBGColor(g_UITheme->mainBgColor);

    m_testType = -1;
    m_label.setText("");
    m_isTesting = false;
    m_label.setGeometry(0, (MAX_Y - 40) / 2, MAX_X, 40 * LCD_RATE);
    QString strSheets;
    strSheets.sprintf("color: rgb(%d, %d, %d);", g_UITheme->itemMainTextColor.red(), g_UITheme->itemMainTextColor.green(), g_UITheme->itemMainTextColor.blue());
    m_label.setStyleSheet(strSheets);
    QFont font;
    font = m_label.font();
    font.setPixelSize(34);
    m_label.setFont(g_UITheme->PrimaryFont);
    m_label.setAlignment(Qt::AlignCenter | Qt::AlignVCenter);
}

void LcdTestForm::OnStart()
{
    FormBase::OnStart();
//    QTimer::singleShot(3000, this, SLOT(ClickedBack()));

    g_iWifiTest = 0;

    m_testIndex = 0;
    m_testType = -1;
    m_isTesting = false;
    m_label.setText(tr("Device Test"));

    QString strSerial = QString::fromUtf8(szSerial);
    QQREncode xEncoder;
    xEncoder.encode(strSerial);
    m_xQRImage = xEncoder.toQImage(MAX_X);
    QRcode_clearCache();


    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* r = w->GetROK();
    r->SetKeyScanMode(this);
    r->SetTimeOut(100000);
}


void LcdTestForm::setTesting(bool b)
{
    if (m_isTesting && !b)
    {
        m_label.setText(m_label.text() + tr(":Finished"));

#if 1
        if(m_testType == TEST_WIFI)
        {
            if(g_iWifiTest == 1)
                m_label.setText(m_label.text() + "\n" + StringTable::Str_Successful);
            else
                m_label.setText(m_label.text() + "\n" + StringTable::Str_Failure);
        }
#endif
    }
}

bool LcdTestForm::isTesting()
{
    return m_isTesting;
}

void LcdTestForm::OnResume()
{
    FormBase::OnResume();

    setTesting(false);
}

void LcdTestForm::doNextTest()
{
    PlayThread* pPlayThread = PlayThread::GetInstance();
    if (m_testType >= 0 && (isTesting() || pPlayThread->isRunning() ))
    {
        return;
    }
    if (m_testType >= 0)
    {
        PlayThread::WaitForFinished();
#if USING_BUZZER
        MainSTM_Command(MAIN_STM_BUZZER_SUCCESS);
        usleep(300 * 1000);
#else
        PlayAlarmSound(1);
#endif
    }
    if (m_testType + 1>= TEST_END)
    {
//        CSI_PWDN_ON();
//        MainSTM_Command(MAIN_STM_CMD_POWER_DOWN);
        LCDTask::DispOff();
        usleep(30 * 1000);
        qApp->exit(-1);
    }
    else
    {
        m_testType++;

#if (LOCK_MODE == LM_AUTO)
        if(g_xSS.iWakeupByEmer == 1 && m_testType == TEST_MOTOR)
            m_testType ++;
#endif

        doDeviceTest(m_testType);
    }
}

void LcdTestForm::ClickedBack()
{
    SigBack();
}

void LcdTestForm::mousePressEvent(QMouseEvent *)
{
    ClickedBack();
}

void LcdTestForm::paintEvent(QPaintEvent* e)
{
    QWidget::paintEvent(e);

    QPainter painter;
    painter.begin(this);
    if(m_testType == TEST_LCD)
    {
        usleep(500*1000);
        painter.drawImage(rect(), QImage(":/icons/Penguins.jpg"));
    }
    else if(m_testType == TEST_SN)
    {
        painter.fillRect(rect(), Qt::white);
        painter.drawImage((MAX_X - m_xQRImage.width()) / 2, (MAX_Y - m_xQRImage.height()) / 2, m_xQRImage);

    }
    painter.end();
}

void LcdTestForm::keyLeftClicked()
{
    if (m_testType >= 0)
    {
#if USING_BUZZER
        MainSTM_Command(MAIN_STM_BUZZER_FAILED);
#else
        PlayError5Sound();
#endif
        usleep(300 * 1000);
    }
}

void LcdTestForm::keyRightClicked()
{
    if (m_testType >= 0)
    {
        PlayThread::WaitForFinished();
#if USING_BUZZER
        MainSTM_Command(MAIN_STM_BUZZER_SUCCESS);
        usleep(300 * 1000);
#else
        PlayAlarmSound(1);
#endif
    }
    if (m_testType >= TEST_END)
    {
        ClickedBack();
    }
    else
    {
        m_testType++;
#if (LOCK_MODE == LM_AUTO)
        if(g_xSS.iWakeupByEmer == 1 && m_testType == TEST_MOTOR)
            m_testType ++;
#endif /* LOCK_MODE == LM_AUTO */
        doDeviceTest(m_testType);
    }
}

void LcdTestForm::doTestSN()
{
    m_label.setVisible(false);
    update();
    setTesting(false);
}

void LcdTestForm::doTestLCD()
{
    m_label.setVisible(false);
    update();
    setTesting(false);
}

void LcdTestForm::doTestMotor()
{
    MotoTestForm* pForm = new MotoTestForm(m_pParentView, this);
    connect(pForm, SIGNAL(SigBack()), this, SLOT(OnResume()));

    pForm->Start(TEST_TYPE_OPEN_CLOSE);
}

void LcdTestForm::doTestIRCam()
{
    IRTestForm* pCamTestForm = new IRTestForm(m_pParentView, this);
    connect(pCamTestForm, SIGNAL(SigBack(int)), this, SLOT(CamTestFinished(int)));

    pCamTestForm->StartTest();

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* r = w->GetROK();
    r->SetKeyScanMode(pCamTestForm);
    r->SetTimeOut(100000);

}

void LcdTestForm::doTestIRCam2()
{
    CamTestForm* pCamTestForm = new CamTestForm(m_pParentView, this);
    connect(pCamTestForm, SIGNAL(SigBack(int)), this, SLOT(CamTestFinished(int)));
    pCamTestForm->StartTest(DEV_IR_CAM1);

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* r = w->GetROK();
    r->SetKeyScanMode(pCamTestForm);
    r->SetTimeOut(100000);
}

void LcdTestForm::doTestClrCam()
{
    CamTestForm* pCamTestForm = new CamTestForm(m_pParentView, this);
    connect(pCamTestForm, SIGNAL(SigBack(int)), this, SLOT(CamTestFinished(int)));

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* r = w->GetROK();
    r->SetKeyScanMode(pCamTestForm);
    r->SetTimeOut(100000);
}

void LcdTestForm::CamTestFinished(int iCamError)
{
    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* r = w->GetROK();
    r->SetKeyScanMode(this);
    r->SetTimeOut(100000);

    QString str = m_label.text();
    if(iCamError)
    {
        str += tr(" Error");
        m_label.setText(str);
    }
    else
    {
        PlayThread::WaitForFinished();
        doNextTest();
    }
}

void LcdTestForm::doTestCard()
{
    EnrollCardForm* pEnrollCardForm = new EnrollCardForm(m_pParentView, this);
    connect(pEnrollCardForm, SIGNAL(SigSendEnrollFinished(int)), this, SLOT(slotEnrollFinished(int)));

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* r = w->GetROK();
    r->SetKeyScanMode(pEnrollCardForm);
    r->SetTimeOut(100000);

    pEnrollCardForm->StartEnroll(100000);
}

void LcdTestForm::slotEnrollFinished(int iEnrollResult)
{
    setTesting(false);

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* r = w->GetROK();
    r->SetKeyScanMode(this);
    r->SetTimeOut(100000);

    if(iEnrollResult == 0)
        OnResume();
    else if(iEnrollResult == 1)
    {
        OnResume();
        PlayThankyouSound(0);
    }

    PlayThread::WaitForFinished();
    doNextTest();
}

void LcdTestForm::doTestFingerprint()
{
    EnrollFPForm* pEnrollFPForm = new EnrollFPForm(m_pParentView, this);
    connect(pEnrollFPForm, SIGNAL(SigEnrollFinished(int)), this, SLOT(EnrollFPFinished(int)));

    FPTask::DeleteFP(1);

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* r = w->GetROK();
    r->SetKeyScanMode(pEnrollFPForm);
    r->SetTimeOut(100000);

    pEnrollFPForm->StartEnroll(1, 100000);
}

void LcdTestForm::EnrollFPFinished(int iEnrollResult)
{
    QString str;
    str = m_label.text();

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* r = w->GetROK();
    r->SetKeyScanMode(this);
    r->SetTimeOut(100000);

    if(iEnrollResult == 0)
    {
        OnResume();
        str += tr(" None");
    }
    else if(iEnrollResult == 1)
    {
#if (FP_MODE == FP_TUZHENG || FP_MODE == FP_GOWEI)
        FPTask::DeleteFP(1);
#endif
        str += tr(" OK");

        OnResume();
        PlayThankyouSound(0);
    }
    else if(iEnrollResult == 2)
    {
        FormBase::OnResume();
        str += tr(" Dup Err");
    }
    else if(iEnrollResult == 3)
    {
        FormBase::OnResume();
        str += tr(" Limit Err");
    }

    doNextTest();
    return;
}

void LcdTestForm::doTestTouch()
{
    TouchTestForm* pxTouchTestForm = new TouchTestForm(m_pParentView, this);
    connect(pxTouchTestForm, SIGNAL(SigBack()), this, SLOT(touchTestEnded()));

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* r = w->GetROK();
    r->SetKeyScanMode(pxTouchTestForm);

    pxTouchTestForm->OnStart();
}

void LcdTestForm::touchTestEnded()
{
    OnResume();
    setTesting(false);
    m_label.setText(tr("Device Test") + tr(":Finished"));

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* r = w->GetROK();
    r->SetKeyScanMode(this);
    r->SetTimeOut(100000);

#if USING_BUZZER
    MainSTM_Command(MAIN_STM_BUZZER_SUCCESS);
#else
    PlayAlarmSound(1);
#endif

//    doNextTest();
}

void LcdTestForm::doDeviceTest(int type)
{
    bool use_thread = false;
    m_label.setVisible(true);
    setTesting(true);
    update();
    switch(type)
    {
    case TEST_SN:
        m_label.setText(tr("SN"));
        doTestSN();
        break;
#if USING_BUZZER
    case TEST_BUZZER:
        m_label.setText(tr("Buzzer"));
        use_thread = true;
        break;
#endif
    case TEST_SPEAKER:
        m_label.setText(tr("Speaker"));
        use_thread = true;
        break;
#if (LOCK_MODE == LM_AUTO)
    case TEST_PWM:
        m_label.setText(tr("PWM"));
        use_thread = true;
        break;
#endif /* LOCK_MODE == LM_AUTO */
    case TEST_LCD:
        m_label.setText(tr("LCD"));
        doTestLCD();
        break;
    case TEST_CLRCAM:
        m_label.setText(tr("Color Camera"));
        doTestClrCam();
        break;
    case TEST_IRCAM:
        m_label.setText(tr("IR Camera"));
        doTestIRCam();
        break;
    case TEST_IRCAM2:
        m_label.setText(tr("IR Camera") + "2");
        doTestIRCam2();
        break;
    case TEST_CARD:
        m_label.setText(tr("Card"));
        doTestCard();
        break;
#if 0
    case TEST_KEYS:
        m_label.setText(tr("Keyboard"));
        setTesting(false);
        break;
#endif

#if 1
    case TEST_WIFI:
        m_label.setText(StringTable::Str_Wifi);
        use_thread = true;
        break;
#endif
    case TEST_BELL:
        setTesting(false);
        m_label.setText(tr("Bell"));
        break;
    case TEST_TOUCH:
        m_label.setText(tr(""));
        doTestTouch();
        break;
    default:
        break;
    }
    if (use_thread)
    {
        m_testThread.wait();
        m_testThread.start();
    }
}


int LcdTestForm::testType()
{
    return m_testType;
}

bool LcdTestForm::event(QEvent* e)
{
    if(e->type() == EV_KEY_EVENT)
    {
        KeyEvent* pEvent = static_cast<KeyEvent*>(e);
        qDebug() << "LcdTestForm:KeyEvent" << pEvent->m_iKeyID << pEvent->m_iEvType;
        if (pEvent->m_iKeyID == E_BTN_FUNC)
        {
            switch(pEvent->m_iEvType)
            {
            case KeyEvent::EV_CLICKED:
                doNextTest();
                break;
            case KeyEvent::EV_DOUBLE_CLICKED:
                break;
            case KeyEvent::EV_LONG_PRESSED:
            {
                qApp->exit(-1);
                break;
            }
            }
        }
        else if(pEvent->m_iKeyID == RING_KEY)
        {
            if(pEvent->m_iEvType == KeyEvent::EV_PRESSED)
            {
                if(m_testType == TEST_BELL)
                {
                    doTestBell();
                }
            }
        }
    }    

    return QWidget::event(e);
}

void LcdTestForm::doTestBell()
{
    MainBackProc::MotorCmd = 0;
    MainBackProc::SoundPlay(g_pUartTask, 41, 10, 0);
    usleep(3500 * 1000);

    setTesting(false);
}


DeviceTestThread::DeviceTestThread(LcdTestForm* parent)
{
    m_form = parent;
}

void DeviceTestThread::run()
{
    switch(m_form->testType())
    {
    case LcdTestForm::TEST_SN:
        break;
#if USING_BUZZER
    case LcdTestForm::TEST_BUZZER:
        doTestBuzzer();
        break;
#endif
    case LcdTestForm::TEST_SPEAKER:
        doTestSpeaker();
        break;
#if (LOCK_MODE == LM_AUTO)
    case LcdTestForm::TEST_PWM:
        doTestPWM();
        break;
#endif /* LOCK_MODE == LM_AUTO */
    case LcdTestForm::TEST_LCD:
        break;
    case LcdTestForm::TEST_CLRCAM:
        break;
    case LcdTestForm::TEST_IRCAM:
        break;
    case LcdTestForm::TEST_IRCAM2:
        break;
    case LcdTestForm::TEST_CARD:
        break;
    case LcdTestForm::TEST_FINGERPRINT:
        break;
#if 0
    case LcdTestForm::TEST_KEYS:
        break;
#endif
    case LcdTestForm::TEST_MOTOR:
#if (LOCK_MODE == LM_AUTO)
        doTestMotor();
#else /* LOCK_MODE == LM_AUTO */
        doTestSemiMotor();
#endif /* LOCK_MODE == LM_AUTO */
        break;
#if 1
    case LcdTestForm::TEST_WIFI:
        doTestWifi();
        break;
#endif
    case LcdTestForm::TEST_BELL:
        break;
    case LcdTestForm::TEST_TOUCH:
        break;
    default:
        break;
    }
    m_form->setTesting(false);
}

#if USING_BUZZER
void DeviceTestThread::doTestBuzzer()
{
    usleep(500 * 1000);
    for (int i = 0; i < 3; i ++)
    {
        MainSTM_Command(MAIN_STM_BUZZER_INIT);
        usleep(800 * 1000);
    }
}
#endif

void DeviceTestThread::doTestSpeaker()
{
    PlayThankyouSound(1);
}

void DeviceTestThread::doTestPWM()
{        
    MainBackProc::SoundPlay(g_pUartTask, 41, 10, 0);
    usleep(3500 * 1000);
}

#if (LOCK_MODE == LM_AUTO)
void DeviceTestThread::doTestMotor()
{
    MainBackProc::MotorTest(g_pUartTask);
    usleep(1000 * 1000);
}
#else // LOCK_MODE == LM_AUTO

#endif // LOCK_MODE == LM_AUTO

void DeviceTestThread::doTestWifi()
{
    g_pUartTask->Stop();

    MAIN_BACK_CMD xSendCmd = { 0 };
    xSendCmd.iHeader = MAIN_BACK_HEADER;
    xSendCmd.bCmd = E_CMD_MB_WIFI_ON;
    xSendCmd.bPacketLen = sizeof(MAIN_BACK_CMD);

    xSendCmd.bCheckSum = MainBackProc::CalcCheckSum(&xSendCmd);

    unsigned char openCmd[4] = {0x44, 0xF0, 0x3E, 0x10};
    unsigned char recvCmd[4] = { 0 };

    g_xUartMutex.Lock();
    UART_Send(openCmd, sizeof(openCmd));
    g_xUartMutex.Unlock();

    UART_RecvDataForWait(recvCmd, sizeof(recvCmd), 500, 1000);

    if(!memcmp(openCmd, recvCmd, sizeof(openCmd)))
    {
#if USING_BUZZER
        MainSTM_Command(MAIN_STM_BUZZER_FAILED);
#else
        PlayError5Sound();
#endif
        usleep(1000 * 1000);

        g_iWifiTest = 0;
        g_pUartTask->Start();
        return;
    }

    UART_SetBaudrate(UART_OutBaudrate());

    //wifi on
    g_xUartMutex.Lock();
    UART_Send((unsigned char*)&xSendCmd, sizeof(xSendCmd));
    g_xUartMutex.Unlock();
    usleep(500 * 1000);

    g_xUartMutex.Lock();
    UART_Send(openCmd, sizeof(openCmd));
    g_xUartMutex.Unlock();

    UART_RecvDataForWait(recvCmd, sizeof(recvCmd), 500, 1000);
    if(!memcmp(openCmd, recvCmd, sizeof(openCmd)))
    {
#if USING_BUZZER
        MainSTM_Command(MAIN_STM_BUZZER_SUCCESS);
#else
        PlayCompleteSoundAlways();
#endif
        g_iWifiTest = 1;
    }
    else
    {
#if USING_BUZZER
        MainSTM_Command(MAIN_STM_BUZZER_FAILED);
#else
        PlayError5Sound();
#endif
        g_iWifiTest = 0;
    }

    //wifi off
    xSendCmd.bCmd = E_CMD_MB_WIFI_OFF;
    xSendCmd.bCheckSum = MainBackProc::CalcCheckSum(&xSendCmd);

    g_xUartMutex.Lock();
    UART_Send((unsigned char*)&xSendCmd, sizeof(xSendCmd));
    g_xUartMutex.Unlock();

    UART_SetBaudrate(UART_InBaudrate());
    usleep(500 * 1000);

    g_pUartTask->Start();
}
