#include "mainactivationform.h"
#include "qqrencode.h"
#include "base.h"
#include "qrencode.h"
#include "mainwindow.h"
#include "rokthread.h"
#include "qrcodereader.h"
#include "settings.h"
#include "alertdlg.h"
#include "stringtable.h"
#include "DBManager.h"
#include "shared.h"
#include "sha1.h"
#include "i2cbase.h"
#include "drv_gpio.h"
#include "mainbackproc.h"
#include "desinterface.h"
#include "camera_api.h"
#include "desinterface.h"
#include "KeyGenBase.h"
#include "b64.h"
#include "desinterface.h"
#include "themedef.h"
#include "lcdtask.h"
#include "aescrypt.h"
#include "soundbase.h"

#include <QtGui>
#include <QPushButton>

#include <unistd.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mount.h>

extern unsigned char g_abKey[16];
extern void GetLicense(char* szLicenseDst);
extern void GetUniquID(char* szDst);
extern int g_iUniqueID;

void calculate_xor(unsigned int *msg, int msg_len, unsigned int *key,  int key_len);
void GetSN(char* serialData);
int ProcessActivation(char* pbUID, int iUniqueID);

MainActivationForm::MainActivationForm(QGraphicsView *pView, FormBase* pParentForm)
    : FormBase(pView, pParentForm)
{
    QFont xFont = qApp->font();
    xFont.setPixelSize(14);

    QPushButton* pBtnBack = new QPushButton(this);
    pBtnBack->setText(tr("Back"));
    pBtnBack->setGeometry(35 * LCD_RATE, 5 * LCD_RATE, 120, 25);
    pBtnBack->setFont(xFont);
    pBtnBack->show();

    QPushButton* pBtnAct = new QPushButton(this);
    pBtnAct->setText(tr("Activation"));
    pBtnAct->setGeometry(165 * LCD_RATE, 5 * LCD_RATE, 120, 25);
    pBtnAct->show();

    pBtnAct->setFont(xFont);

    connect(pBtnBack, SIGNAL(clicked()), this, SIGNAL(SigBack()));
    connect(pBtnAct, SIGNAL(clicked()), this, SLOT(QRCodeReadClick()));
}

MainActivationForm::~MainActivationForm()
{
    OnStop();
}

void MainActivationForm::OnStart(int fDelPrevScene)
{
    FormBase::OnStart(fDelPrevScene);
    ROKTHread* pROKThread = ((MainWindow*)m_pParentView)->GetROK();
    printf("ggggggggggggggg %x\n", pROKThread);
//    pROKThread->SetKeyScanMode(this);
    printf("%s\n", __FUNCTION__);
}

void MainActivationForm::OnStop()
{
    ROKTHread* pROKThread = ((MainWindow*)m_pParentView)->GetROK();
//    pROKThread->SetKeyScanMode(NULL);

    FormBase::OnStop();
    printf("%s\n", __FUNCTION__);
}

void MainActivationForm::OnPause()
{
    ROKTHread* pROKThread = ((MainWindow*)m_pParentView)->GetROK();
//    pROKThread->SetKeyScanMode(NULL);

    FormBase::OnPause();
    printf("%s\n", __FUNCTION__);
}

void MainActivationForm::QRCodeReadFinished(QString strQRCode)
{
    FormBase::OnResume();

    if(strQRCode.isEmpty())
    {
        ROKTHread* pROKThread = ((MainWindow*)m_pParentView)->GetROK();
//        pROKThread->SetKeyScanMode(this);
        return;
    }

    ROKTHread* pROKThread = ((MainWindow*)m_pParentView)->GetROK();
//    pROKThread->SetKeyScanMode(this);
}

void MainActivationForm::QRCodeReadFinished(int iCamError)
{
    FormBase::OnResume();

    ROKTHread* pROKThread = ((MainWindow*)m_pParentView)->GetROK();
//    pROKThread->SetKeyScanMode(this);

    if(iCamError == 1)
    {
#if USING_BUZZER
        MainSTM_Command(MAIN_STM_BUZZER_ERROR);
#else
        PlayError5Sound();
#endif
    }
}

void MainActivationForm::QRCodeReadClick()
{
    QRCodeReaderForm* pQRCodeReaderForm = new QRCodeReaderForm(m_pParentView, this);
    connect(pQRCodeReaderForm, SIGNAL(SigQRCodeRead(QString)), this, SLOT(QRCodeReadFinished(QString)));
    connect(pQRCodeReaderForm, SIGNAL(SigQRCodeRead(int)), this, SLOT(QRCodeReadFinished(int)));
    pQRCodeReaderForm->StartRead(-1);
}

void MainActivationForm::paintEvent(QPaintEvent *e)
{
    QPainter painter;
    painter.begin(this);
    painter.fillRect(rect(), Qt::white);
    painter.drawImage((MAX_X - m_qrImage.width()) / 2, (MAX_Y - m_qrImage.height()) / 2, m_qrImage);
    painter.end();
}

void MainActivationForm::RetranslateUI()
{
#ifdef _NO_ENGINE_
    char szSerial[256] = { 0 };
    GetSerialNumber(szSerial);

    QString strVersion = QString::fromUtf8(DEVICE_FIRMWARE_VERSION_INNER);

    char szUniquID[256] = { 0 };
    GetUniquID(szUniquID);

    QString strCode;
    strCode.sprintf("%s\n%s\n%s\n%s", szSerial, DEVICE_MODEL_NUM, strVersion.toUtf8().data(), szUniquID);

    QQREncode xEncoder;
    xEncoder.encode(strCode);
    m_qrImage = xEncoder.toQImage(MAX_Y * 4 / 5);
    QRcode_clearCache();
    update();
#endif
}

bool MainActivationForm::event(QEvent* e)
{
    if(e->type() == EV_KEY_EVENT)
    {
        KeyEvent* pEvent = static_cast<KeyEvent*>(e);
        if(pEvent != NULL && pEvent->m_iKeyID == E_BTN_FUNC)
        {
            if(pEvent->m_iEvType == KeyEvent::EV_CLICKED)
            {
                QRCodeReadClick();
            }
            else if(pEvent->m_iEvType == KeyEvent::EV_DOUBLE_CLICKED)
            {
                SigBack();
            }
        }
    }


    return QWidget::event(e);
}



void calculate_xor(unsigned int *msg, int msg_len, unsigned int*key,  int key_len)
{
}


void GetSN(char* serialData)
{
}

int MarkActivationFailed(int iError)
{
    FILE* fp = fopen("/test/act_mark", "wb");
    if(fp)
    {
        fwrite(&iError, sizeof(int), 1, fp);
        fflush(fp);
        fclose(fp);
    }

    return 0;
}

int ProcessActivation(char* pbUID, int iUniqueID)
{
    return 0;
}

int GetIntCheckSum(int* piData, int iSize)
{
    int iCheckSum = 0;
    for(int i = 0; i < iSize / sizeof(int); i ++)
        iCheckSum = iCheckSum + piData[i];

    iCheckSum = ~iCheckSum;

    return iCheckSum;
}
