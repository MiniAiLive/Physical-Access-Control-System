#ifndef LCDTESTFORM_H
#define LCDTESTFORM_H

#include "formbase.h"
#include "appdef.h"
#include <QLabel>
#include <QThread>

class LcdTestForm;

class DeviceTestThread: public QThread
{
public:
    DeviceTestThread(LcdTestForm* parent);
protected:
    void    run();
#if USING_BUZZER
    void    doTestBuzzer();
#endif
    void    doTestSpeaker();
    void    doTestPWM();
#if (LOCK_MODE == LM_AUTO)
    void    doTestMotor();
#else
    void    doTestSemiMotor();
#endif
    void    doTestWifi();
private:
    LcdTestForm* m_form;
};

class LcdTestForm : public FormBase
{
    Q_OBJECT
public:
    explicit LcdTestForm(QGraphicsView *pView, FormBase* pParentForm);
    enum {
        TEST_SN,
#if USING_BUZZER
        TEST_BUZZER,
#endif
        TEST_SPEAKER,
#if (LOCK_MODE == LM_AUTO)
        TEST_PWM,
#endif /* LOCK_MODE == LM_AUTO */
        TEST_LCD,
        TEST_CLRCAM,
        TEST_IRCAM,
        TEST_IRCAM2,
        TEST_CARD,
        TEST_FINGERPRINT,
#if 0
        TEST_KEYS,
#endif
        TEST_WIFI,
        TEST_MOTOR,
        TEST_BELL,
        TEST_TOUCH,
        TEST_END
    };
    int testType();
    void    setTesting(bool);
    bool    isTesting();
    void    OnResume();
    void    doNextTest();

signals:

public slots:
    void    OnStart();
    void    ClickedBack();
    void    CamTestFinished(int iCamError);
    void    slotEnrollFinished(int iEnrollResult);
    void    EnrollFPFinished(int iEnrollResult);
    void    touchTestEnded();

protected:
    bool    event(QEvent* e);
    void    paintEvent(QPaintEvent *e);
    void    mousePressEvent(QMouseEvent *);
    void    keyLeftClicked();
    void    keyRightClicked();
    void    doDeviceTest(int type);
    void    doTestSN();
    void    doTestLCD();
    void    doTestMotor();
    void    doTestIRCam();
    void    doTestIRCam2();
    void    doTestClrCam();
    void    doTestCard();
    void    doTestFingerprint();
    void    doTestTouch();
    void    doTestBell();
private:
    int     m_testIndex;
    QLabel m_label;
    int m_testType;
    DeviceTestThread m_testThread;
    bool m_isTesting;

    QImage  m_xQRImage;
};

#endif // LCDTESTFORM_H
