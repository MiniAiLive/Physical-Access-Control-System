#include "enrollcardform.h"
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
#include "i2cbase.h"
#include "drv_gpio.h"
#include "mainbackproc.h"
#include "soundbase.h"
#include "nfccard.h"

#include <QtGui>
#include <QLabel>
#include <QVBoxLayout>
#include <unistd.h>

EnrollCardForm::EnrollCardForm(QGraphicsView *view, FormBase* parentForm)
                               : FormBase(view, parentForm)
{
    m_iCardID = 0;
    m_iSectorNum = 0;
    m_iCardRand = 0;
    m_iTimeout = 0;

    m_lblTitle = new QLabel;
    m_lblTitle->setWordWrap(true);
    m_lblTitle->setAlignment(Qt::AlignHCenter | Qt::AlignVCenter);
//    m_lblTitle->setFixedHeight(360);

    QString strBorderSheets;
    strBorderSheets.sprintf("QWidget { color: rgb(%d, %d, %d);}", g_UITheme->titleMainTextColor.red(), g_UITheme->titleMainTextColor.green(), g_UITheme->titleMainTextColor.blue());

    m_lblTitle->setStyleSheet(strBorderSheets);
    m_lblTitle->setFont(g_UITheme->PrimaryFont);

    m_lblCard = new QLabel;
    m_lblCard->setPixmap(QPixmap(":/icons/putcard.png"));
    m_lblCard->setAlignment(Qt::AlignHCenter | Qt::AlignTop);

    m_lytMain = new QVBoxLayout;
    m_lytMain->addWidget(m_lblTitle);
    m_lytMain->addWidget(m_lblCard);
    m_lytMain->setMargin(30);
    setLayout(m_lytMain);

    strBorderSheets.sprintf("QWidget { background-color: rgb(%d, %d, %d);}", g_UITheme->mainBgColor.red(), g_UITheme->mainBgColor.green(), g_UITheme->mainBgColor.blue());

    setStyleSheet(strBorderSheets);
    setAutoDelete(false);

    m_lblTitle->setAttribute(Qt::WA_DeleteOnClose);
    m_lblCard->setAttribute(Qt::WA_DeleteOnClose);

    RetranslateUI();
}

EnrollCardForm::~EnrollCardForm()
{
    delete m_lblTitle;
    delete m_lblCard;
    delete m_lytMain;
}

void EnrollCardForm::StartEnroll(int iTimeout)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    FormBase::OnStart();

    m_iCardID = 0;
    m_iSectorNum = 0;
    m_iCardRand = 0;
    m_fRunning = 1;
    m_iTimeout = iTimeout;
    QThreadPool::globalInstance()->start(this);
}

void EnrollCardForm::OnPause()
{
    m_fRunning = 0;
    QThreadPool::globalInstance()->waitForDone();
}

int EnrollCardForm::GetEnrolledCardID()
{
    return m_iCardID;
}

int EnrollCardForm::GetEnrolledSectorNum()
{
    return m_iSectorNum;
}

int EnrollCardForm::GetEnrolledCardRand()
{
    return m_iCardRand;
}

void EnrollCardForm::run()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    PlayEnrollCardSound(1);

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* pRokThread = w->GetROK();

    QTime xOldTime = QTime::currentTime();
    while(xOldTime.msecsTo(QTime::currentTime()) < m_iTimeout * 1000 && m_fRunning)
    {
        pRokThread->InitTime();

#if 1
        unsigned char bSectorNum = 0;
        int iCardRand = 0;
        int iCardID = 0;

        bSectorNum = Get_CardIDCopyProtect((u8*)&iCardID, (u8*)&iCardRand, E_CARD_ENROLL);
        if (!(bSectorNum == E_CARD_RESULT_NOT_GET_CARD_ID | bSectorNum == E_CARD_RESULT_NOT_EXIST_USING_SECTOR | bSectorNum == E_CARD_RESULT_UNKNOWN_CARD))
        {
            if(bSectorNum == 0)
            {
#if USING_BUZZER
                MainSTM_Command(MAIN_STM_BUZZER_FAILED);
#else
                PlayError5Sound();
#endif
                usleep(1800 * 1000);
                break;
            }
            m_iCardID = iCardID;
            m_iSectorNum = bSectorNum;
            m_iCardRand = iCardRand;
            break;            
        }
#else
        int nRandVal = rand() % 10;
        if(nRandVal == 0)
        {
            m_iCardID = rand();
            m_iSectorNum = 1;
            m_iCardRand = 1;
#if USING_BUZZER
            MainSTM_Command(MAIN_STM_BUZZER_FAILED);
#else
            PlayError5Sound();
#endif

            break;
        }
#endif

#if (AUTO_TEST == 1)
        int nRandVal = rand() % 10;
        if(nRandVal == 0)
        {
            m_iCardID = rand();
            PlayBuzzer(BUZ_SUCCESS);
            break;
        }
#endif
        usleep(100 * 1000);
    }

    if(xOldTime.msecsTo(QTime::currentTime()) < 500)
    {
        usleep((500 - xOldTime.msecsTo(QTime::currentTime())) * 1000);
    }    

    if(!FormBase::QuitFlag)
    {
        if(m_iCardID != 0)
        {
            AlertDlg::Locked = 1;
            emit SigSendEnrollFinished(1);
            return;
        }

        emit SigSendEnrollFinished(0);

    }
}

void EnrollCardForm::mousePressEvent(QMouseEvent* e)
{
    m_fRunning = 0;
}


void EnrollCardForm::RetranslateUI()
{
    m_lblTitle->setText(StringTable::Str_Put_your_card_on_the_center_of_keyboard);
}

bool EnrollCardForm::event(QEvent* e)
{
    if(e->type() == EV_KEY_EVENT)
    {
        KeyEvent* pEvent = static_cast<KeyEvent*>(e);
        qDebug() << "EnrollCardForm:KeyEvent" << pEvent->m_iKeyID << pEvent->m_iEvType;
        if (pEvent->m_iKeyID == E_BTN_FUNC)
        {
            switch(pEvent->m_iEvType)
            {
            case KeyEvent::EV_CLICKED:
                m_fRunning = 0;
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
