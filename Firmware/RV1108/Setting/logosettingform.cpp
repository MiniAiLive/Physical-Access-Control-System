#include "logosettingform.h"
#include "shared.h"
#include "base.h"
#include "camera_api.h"
#include "themedef.h"
#include "ui_logosettingform.h"
#include "shared.h"

#include <QtGui>

LogoSettingForm::LogoSettingForm(QGraphicsView* pView, FormBase* pParentForm) :
    FormBase(pView , pParentForm)
{
    QWidget* pWidget = new QWidget(this);
    ui = new Ui::LogoSettingForm;
    ui->setupUi(pWidget);

    ui->btnBack->SetImages(QImage(":/icons/ic_back_circle.png"));
    connect(ui->btnBack, SIGNAL(clicked()), this, SIGNAL(SigBack()));
}

LogoSettingForm::~LogoSettingForm()
{
    OnStop();
    delete ui;
}

void LogoSettingForm::OnStart()
{
    g_xSS.iCurLogo = g_xPS.x.bLogo;
    UpdateImage();
    FormBase::OnStart();
}

void LogoSettingForm::OnStop()
{
    FormBase::OnStop();

    if(g_xPS.x.bLogo != g_xSS.iCurLogo)
    {
        QString strFileName;
        strFileName.sprintf("/test/logo/%d.jpg", g_xSS.iCurLogo);
        QImage xLogoImg(strFileName);
        xLogoImg = xLogoImg.convertToFormat(QImage::Format_ARGB32);

        if(!xLogoImg.isNull())
        {
            SetLogo(xLogoImg.bits(), xLogoImg.bytesPerLine() * xLogoImg.height());

            g_xPS.x.bLogo = g_xSS.iCurLogo;
            UpdatePermanenceSettings();
        }
    }
}

void LogoSettingForm::paintEvent(QPaintEvent *e)
{
    QWidget::paintEvent(e);

    QPainter painter;
    painter.begin(this);
    painter.fillRect(rect(), Qt::black);
    painter.drawImage((MAX_X - m_xLogoImage.width()) / 2, (MAX_Y - m_xLogoImage.height()) / 2, m_xLogoImage);
    painter.end();
}

void LogoSettingForm::mousePressEvent(QMouseEvent* e)
{
    QWidget::mousePressEvent(e);

    g_xSS.iCurLogo ++;
    if(g_xSS.iCurLogo >= EL_LOGO_END)
        g_xSS.iCurLogo = EL_LOGO_START;

    UpdateImage();
}

void LogoSettingForm::UpdateImage()
{
    QString strFileName;
    strFileName.sprintf("/test/logo/%d.jpg", g_xSS.iCurLogo);
    QImage xLogoImg(strFileName);
    m_xLogoImage = xLogoImg.convertToFormat(QImage::Format_ARGB32);

    update();
}
