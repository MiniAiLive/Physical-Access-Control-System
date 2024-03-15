#ifndef VERIFYFACEFORM_H
#define VERIFYFACEFORM_H

#include "formbase.h"
#include "touchthread.h"
#include <QRunnable>
#include <QThread>
#include <QMutex>
#include <QTime>


namespace Ui {
class PasscodeForm;
}

enum
{
    VF_FINISH,
    VF_START,
    VF_VIDEO_START,
    VF_VIDEO_STOP,
    VF_VDB_START,
    VF_LOOP,
    VF_PASSWORD_RESULT,
    VF_PASSWORD_STARTED,
    VF_SCREEN_SAVER_STARTED,
    VF_MAIN_STARTED,
    VF_VDB_STARTED,
    VF_EVENT,
    VF_END,
};

class QLabel;
class FaceRecogTask;
class CardRecogTask;
class CustomKeyButton;
class VerifyFaceForm : public FormBase, public QRunnable
{
    Q_OBJECT
public:
    explicit VerifyFaceForm(QGraphicsView *pView, FormBase* pParentForm);
    ~VerifyFaceForm();

    enum    {SCENE_NONE, SCENE_SUCC, SCENE_FAILED, SCENE_MAIN, SCENE_SCREEN_SAVER, SCENE_PASSWORD, SCENE_VDB};
    enum    {TOUCH_PRESS, TOUCH_MOVE, TOUCH_RELEASE};
    enum    {BACK_MENU, BACK_PASSWORD, BACK_SCREEN_SAVER, BACK_MAIN, BACK_VDB};
    enum    {UI_NONE, UI_PASSWORD, UI_RESULT, UI_SCREEN_SAVER, UI_VDB};

    void    StartVerify();
    void    run();

signals:
    void    SigBack(int iUI, int iParam);

public slots:
    void    OnStop();
    void    OnResume();
    void    OnPause();
    void    GotoBack(int iUI, int iParam);



protected:
    void    mousePressEvent(QMouseEvent* e);
    void    timerEvent(QTimerEvent* e);
    bool    event(QEvent* e);
    void    RetranslateUI();

private:
    void    DrawVerifyScene(QPainter& painter, QRect xScreenRect, int iScene);
    void    DrawButtons(QPainter& painter);
    void    DrawDateTime(QPainter& painter);

    void    SetScene(int iScene);
    void    SetUI(int iIdx, int iParam0 = 0);

    void    ResetButtons();
    void    AddButton(int iID, int iX1, int iY1, int iX2, int iY2, const char* szNormal, const char* szPress, unsigned int iNormalColor, int iPressColor, int iState = BTN_STATE_NONE);
    int     CheckBtnState(QPoint pos, int mode);

    void    SetupUI();
    void    AddPasscode(QChar);
    void    ConnectButtons(int iConnect);

    void    SaveLog(int iLogType, int iResult, int iFindID, int iFistSuccType);
    void    ResetMultiVerify();

private slots:
    void    Click0();
    void    Click1();
    void    Click2();
    void    Click3();
    void    Click4();
    void    Click5();
    void    Click6();
    void    Click7();
    void    Click8();
    void    Click9();
    void    DelClick();
    void    OkClick();
    void    SettingsClick();
    void    ClickBack();
private:
    Ui::PasscodeForm *ui;
    FaceRecogTask*  m_pFaceRecogTask;
    CardRecogTask*  m_pCardRecogTask;

    int     m_iCurScene;

    QMutex  m_xMutex;
    int     m_iBtnCount;
    BUTTON  m_axBtns[MAX_BUTTON_CNT];

    int     m_iCounter;

    QString m_strPasscode;
    float   m_rResetTime;
    float   m_rClickTime;

    int     m_iTimer;
    int     m_iOldMsg;

    QVector<int>    m_vPasscodeMaps;
    QVector<CustomKeyButton*>   m_vPasscodeBtns;

    int     m_iUIWaiting;
};


#endif // VERIFYFACEFORM_H
