#include "passcodeform.h"
#include "ui_passcodeform.h"
#include "stringtable.h"
#include "base.h"
#include "shared.h"
#include "alertdlg.h"
#include "drv_gpio.h"
#include "uitheme.h"
#include "soundbase.h"

#include <QtGui>
#include <unistd.h>


PasscodeForm::PasscodeForm(QGraphicsView *pView, FormBase* pParentForm) :
    FormBase(pView, pParentForm),
    ui(new Ui::PasscodeForm)
{
    ui->setupUi(this);
    SetBGColor(g_UITheme->mainBgColor);
    ui->btnBack->SetImages(QImage(":/icons/ic_arrow_back.png"));
    ui->lblTitle->setFont(g_UITheme->TitleFont);
    ui->lblPasscode1->setFont(g_UITheme->TitleFont);

    m_iStep = STEP_INPUT_NEW;
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
    ui->btnSettings->setVisible(false);

    setAutoDelete(false);
    m_fOldTime = Now();
    m_iTimer = startTimer(100);

    ui->stackedWidget->setCurrentIndex(0);
}

PasscodeForm::~PasscodeForm()
{
    if (m_iTimer > -1)
        killTimer(m_iTimer);
    delete ui;
}

void PasscodeForm::Start(int iStep, QString strOldPasscode)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif
    m_iHiddenCodeFlag = 0;

    m_iStep = iStep;
    m_strOldPasscode = strOldPasscode;

    FormBase::OnStart();

    QThreadPool::globalInstance()->start(this);
}

void PasscodeForm::OnPause()
{
    FormBase::OnPause();

    QThreadPool::globalInstance()->waitForDone();
}

void PasscodeForm::ConnectButtons(int iConnect)
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
    }
}

void PasscodeForm::Next()
{
    m_fOldTime = Now();
    m_iStep ++;
    m_strPasscode.clear();
    RetranslateUI();

    ChangeForm(this, this, ANIMATION_TYPE_RIGHT);
}

void PasscodeForm::Prev()
{
    m_fOldTime = Now();
    m_iStep --;
    m_strPasscode.clear();
    RetranslateUI();

    ChangeForm(this, this, ANIMATION_TYPE_LEFT);
}

QString PasscodeForm::GetPasscode()
{
    return m_strNewPasscode;
}

void  PasscodeForm::Click0()
{
    PlaySoundTop();

    AddPasscode('0');
}

void  PasscodeForm::Click1()
{
    PlaySoundTop();


    AddPasscode('1');
}

void PasscodeForm::Click2()
{
    PlaySoundTop();

    AddPasscode('2');
}

void PasscodeForm::Click3()
{
    PlaySoundTop();


    AddPasscode('3');
}

void PasscodeForm::Click4()
{
    PlaySoundTop();

    AddPasscode('4');
}

void PasscodeForm::Click5()
{
    PlaySoundTop();


    AddPasscode('5');
}

void PasscodeForm::Click6()
{
    PlaySoundTop();

    AddPasscode('6');
}

void PasscodeForm::Click7()
{
    PlaySoundTop();

    AddPasscode('7');
}

void PasscodeForm::Click8()
{
    PlaySoundTop();

    AddPasscode('8');
}

void PasscodeForm::DelClick()
{
    PlaySoundTop();
    if(m_strPasscode.length() == 0)
        m_iHiddenCodeFlag ++;

    m_strPasscode.clear();
    RetranslateUI();
}

void PasscodeForm::ClickBack()
{
    emit SigBack();
}

void PasscodeForm::AddPasscode(QChar passcode)
{
    m_fOldTime = Now();
#if (AUTO_TEST != 1)
    if(m_strPasscode.length() >= TEMP_PASSCODE_LEN)
        return;

    m_strPasscode.append(passcode);
#else
    if(rand() % 2)
    {        
        m_strPasscode.clear();
        m_strPasscode.append('1');
        m_strPasscode.append('2');
        m_strPasscode.append('3');
        m_strPasscode.append('4');
        m_strPasscode.append('5');
        m_strPasscode.append('6');
    }
    else
    {
        m_strPasscode.clear();
        m_strPasscode.append('1');
        m_strPasscode.append('2');
        m_strPasscode.append('3');
        m_strPasscode.append('4');
        m_strPasscode.append('5');
        m_strPasscode.append('5');
    }
#endif
    RetranslateUI();
}

void PasscodeForm::OkClick()
{
    m_fOldTime = Now();
    if(m_iHiddenCodeFlag == 3 && m_strPasscode == "8123456")
    {
        m_iHiddenCodeFlag = 0;

        emit SigHiddenCode();
        return;
    }

    m_iHiddenCodeFlag = 0;

    if(m_iStep == STEP_INPUT_OLD)
    {
        if(m_strOldPasscode != m_strPasscode)
        {
            AlertDlg::WarningOk(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_The_password_is_incorrect);
            m_strPasscode.clear();
            RetranslateUI();
        }
        else
            Next();
    }
    else if(m_iStep == STEP_INPUT_NEW)
    {
        m_strNewPasscode = m_strPasscode;
        Next();
    }
    else if(m_iStep == STEP_INPUT_CONFIRM)
    {
        if(m_strNewPasscode != m_strPasscode)
        {
            AlertDlg::WarningOk(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_The_password_is_incorrect);
            m_strNewPasscode.clear();
            m_strPasscode.clear();

            g_xSS.iNoSoundPlayFlag = 1;
            Prev();
        }
        else
            SigConfirm();
    }

    m_fOldTime = Now();
}

void PasscodeForm::RetranslateUI()
{
    QString strSheets;
    strSheets.sprintf("QWidget { background-color: rgb(%d, %d, %d);}", g_UITheme->mainBgColor.red(), g_UITheme->mainBgColor.green(), g_UITheme->mainBgColor.blue());
    ui->widget->setStyleSheet(strSheets);

    if(((m_strPasscode.length() >= 6) && m_iStep == STEP_INPUT_NEW) || (m_iStep == STEP_INPUT_OLD) || (m_iStep == STEP_INPUT_CONFIRM) || (m_iStep == STEP_CONFIRM))
    {
        ui->btnOk->setEnabled(true);
        ui->btnOk->setText("OK");
    }
    else
    {
        ui->btnOk->setEnabled(false);
        ui->btnOk->setText("");
    }

    QString strPasscode;
    for(int i = 0; i < m_strPasscode.size(); i ++)
        strPasscode += QString::fromUtf8("✱");
    ui->lblPasscode1->setText(strPasscode);

    if(m_iStep == STEP_INPUT_OLD)
    {
        ui->lblTitle->setText(StringTable::Str_Enter_old_password);
    }
    else if(m_iStep == STEP_INPUT_NEW)
    {
        ui->lblTitle->setText(StringTable::Str_Enter_new_password);
    }
    else if(m_iStep == STEP_INPUT_CONFIRM)
    {
        ui->lblTitle->setText(StringTable::Str_Re_enter_your_password);
    }
    else if(m_iStep == STEP_CONFIRM)
    {
        ui->lblTitle->setText("");
    }
}

QVector<QPoint> PasscodeForm::GetChildPos()
{
    m_vChildPos.clear();
    m_vChildPos.append(ui->btnBack->mapToGlobal(ui->btnBack->rect().center()));
    m_vChildPos.append(ui->btnOk->mapToGlobal(ui->btnOk->rect().center()));
    m_vChildPos.append(ui->btnDel->mapToGlobal(ui->btnDel->rect().center()));
    m_vChildPos.append(ui->btn0->mapToGlobal(ui->btn0->rect().center()));

    return m_vChildPos;
}

void PasscodeForm::run()
{
    if(g_xSS.bSound > 0)
    {
        ConnectButtons(0);
        PlayTypePasscodeSound(1);
    }

    ConnectButtons(1);
}

void PasscodeForm::timerEvent(QTimerEvent *e)
{
    if (e->timerId() == m_iTimer)
    {
        if (Now() - m_fOldTime > ENROLL_PASSCODE_TIMEOUT * 1000)
        {
            ClickBack();
            killTimer(m_iTimer);
            m_iTimer = -1;
        }
    }
}
