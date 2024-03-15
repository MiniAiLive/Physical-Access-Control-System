#include "alertdlg.h"
#include "ui_alertdlg.h"
#include "base.h"
#include "uitheme.h"
#include "customradioitem.h"
#include "stringtable.h"
#include "menuitem.h"
#include "customlineedit.h"
#include "themedef.h"
#include "camera_api.h"

#include <QtGui>
#include <QGraphicsProxyWidget>
#include <QGraphicsSceneMouseEvent>
#include <QScrollBar>
#include <QDialog>

#define RADIO_ITEM_HEIGHT (40 * LCD_RATE)

int AlertDlg::Locked = 0;

AlertDlg::AlertDlg(QGraphicsView *pView, FormBase* pParentForm) :
    FormBase(pView, pParentForm),
    ui(new Ui::AlertDlg)
{
    ui->setupUi(this);

    QString strSheets;
    strSheets.sprintf("background-color: rgb(%d, %d, %d);color: rgb(%d, %d, %d);", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue(),
                   g_UITheme->itemMainTextColor.red(), g_UITheme->itemMainTextColor.green(), g_UITheme->itemMainTextColor.blue());
    setStyleSheet(strSheets);

    strSheets.sprintf("background-color: rgb(%d, %d, %d);color: rgb(%d, %d, %d);", g_UITheme->mainBgColor.red(), g_UITheme->mainBgColor.green(), g_UITheme->mainBgColor.blue(),
                   g_UITheme->itemMainTextColor.red(), g_UITheme->itemMainTextColor.green(), g_UITheme->itemMainTextColor.blue());
    ui->lblTitle->setStyleSheet(strSheets);
    ui->lblTitle->setStyleSheet(strSheets);
    ui->lblTitle->setFont(g_UITheme->TitleFont);

    ui->btnOk->setFont(g_UITheme->DlgBtnFont);
    ui->btnOk->setFixedWidth(80 * LCD_RATE);

    ui->btnCancel->setFont(g_UITheme->DlgBtnFont);
    ui->btnCancel->setFixedWidth(80 * LCD_RATE);

    m_pContentsScene = new QGraphicsScene;
    ui->viewContents->setScene(m_pContentsScene);
    ui->viewContents->SetScrollBarColor(g_UITheme->scrollBarColor);

    m_pContentsScene->setSceneRect(QRect(0, 0, 1, 1));
    ui->viewContents->setMaximumHeight(1);

    ui->buttonGroup->setVisible(false);
    ui->btnOk->setVisible(false);
    ui->btnCancel->setVisible(false);

//    resize(width(), 40 * LCD_RATE);

    m_pContentsScene->installEventFilter(this);

    connect(ui->btnOk, SIGNAL(clicked()), this, SLOT(ClickedOk()));
    connect(ui->btnCancel, SIGNAL(clicked()), this, SLOT(ClickedCancel()));

#if (AUTO_TEST == 1)
    QTimer::singleShot(1000, this, SLOT(sltAutoTest()));
#endif
}

AlertDlg::~AlertDlg()
{
    m_pContentsScene->clear();
    delete m_pContentsScene;
    delete ui;
}


void AlertDlg::SetTitle(const QString& strTitle)
{
    ui->lblTitle->setText(strTitle);
}

void AlertDlg::SetTitle(const QString& strTitle, const int iSize)
{
    if(iSize > 0)
    {
        QFont f = ui->lblTitle->font();
        f.setPixelSize(iSize);
        ui->lblTitle->setFont(f);
    }
    ui->lblTitle->setText(strTitle);
}

void AlertDlg::SetTitleEn(bool fShow)
{
    ui->lblTitle->setVisible(fShow);
}

void AlertDlg::SetOkButton(bool fShow, QString strText)
{
    ui->btnOk->setText(strText);
    ui->btnOk->setVisible(fShow);
}

void AlertDlg::SetCancelButton(bool fShow, QString strText)
{
    ui->btnCancel->setText(strText);
    ui->btnCancel->setVisible(fShow);
}


void AlertDlg::SetButtonGroup(bool fShow)
{
    ui->buttonGroup->setVisible(fShow);
}

void AlertDlg::SetAlertCancel(bool fCancel)
{
    m_fAlertCancel = fCancel;
    if(m_fAlertCancel == true)
        m_pParentView->installEventFilter(this);
    else
        m_pParentView->removeEventFilter(this);
}

void AlertDlg::AddRadioItem(QString strText, int iID, bool fSelected)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    CustomRadioItem* pRadioItem = new CustomRadioItem(iID);
    pRadioItem->SetText(strText);
    pRadioItem->setSelected(fSelected);

    QRect xSceneRect = m_pContentsScene->sceneRect().toRect();
    pRadioItem->SetBoundingRect(QRect(xSceneRect.left(), xSceneRect.bottom(), width(), RADIO_ITEM_HEIGHT));

    if(m_pContentsScene->items().count())
        m_pContentsScene->addLine(xSceneRect.left(), xSceneRect.bottom(), xSceneRect.right(), xSceneRect.bottom(), QPen(g_UITheme->radioBorderColor));

    xSceneRect = QRect(xSceneRect.left(), xSceneRect.top(), width(), xSceneRect.height() + RADIO_ITEM_HEIGHT - 1);
    m_pContentsScene->setSceneRect(xSceneRect);
    m_pContentsScene->addItem(pRadioItem);

    ui->viewContents->setMaximumHeight(xSceneRect.height());
    ui->viewContents->resize(width(), xSceneRect.height());
    resize(width(), ui->gridLayout->maximumSize().height());

    connect(pRadioItem, SIGNAL(clicked(int)), this, SLOT(ClickedRad(int)));
}

void AlertDlg::AddWidget(QWidget *pWidget)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    QGraphicsProxyWidget* pProxyWidget =  m_pContentsScene->addWidget(pWidget);
    pProxyWidget->setPos(0, 0);
    m_pContentsScene->setFocus();

    QRect xSceneRect = m_pContentsScene->sceneRect().toRect();
    QRect xNewSceneRect = QRect(xSceneRect.left(), xSceneRect.top(), width(), pWidget->height());

    ui->viewContents->setMaximumHeight(xNewSceneRect.height() + 7);
    ui->viewContents->resize(width(), xNewSceneRect.height() + 7);

    setFixedSize(width(), ui->gridLayout->maximumSize().height());

    pWidget->setFocus();
    QFocusEvent event(QEvent::FocusIn);
    qApp->sendEvent(pWidget, &event);
}

void AlertDlg::ClickedOk()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif
    m_xExecLoop.exit(QDialog::Accepted);
}

void AlertDlg::ClickedCancel()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif
    m_xExecLoop.exit(QDialog::Rejected);
}

void AlertDlg::ClickedRad(int id)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif
    m_xExecLoop.exit(id);
}

QVector<QPoint> AlertDlg::GetChildPos()
{
    m_vChildPos.clear();
    m_vChildPos.append(ui->btnOk->mapToGlobal(ui->btnOk->rect().center()));
    m_vChildPos.append(ui->btnCancel->mapToGlobal(ui->btnCancel->rect().center()));

    return m_vChildPos;
}

bool AlertDlg::eventFilter(QObject *obj, QEvent *e)
{
    if(obj == m_pContentsScene && e->type() == QEvent::GraphicsSceneMousePress)
    {
        QGraphicsSceneMouseEvent* pMouseEvent = static_cast<QGraphicsSceneMouseEvent *>(e);
        m_xOldPos = pMouseEvent->scenePos().toPoint();
        pMouseEvent->accept();

    }
    else if(obj == m_pContentsScene && e->type() == QEvent::GraphicsSceneMouseMove)
    {
        QGraphicsSceneMouseEvent* pMouseEvent = static_cast<QGraphicsSceneMouseEvent *>(e);
        ui->viewContents->verticalScrollBar()->setValue(ui->viewContents->verticalScrollBar()->value() - (pMouseEvent->scenePos().toPoint().y() - m_xOldPos.y()));
    }
    else if(obj == m_pContentsScene && e->type() == QEvent::GraphicsSceneMouseRelease)
    {
        QGraphicsSceneMouseEvent* pMouseEvent = static_cast<QGraphicsSceneMouseEvent *>(e);
        ui->viewContents->verticalScrollBar()->setValue(ui->viewContents->verticalScrollBar()->value() - (pMouseEvent->scenePos().toPoint().y() - m_xOldPos.y()));
    }
    else if(obj == m_pParentView)
    {
        if(e->type() == QEvent::MouseButtonPress)
        {
            QMouseEvent* pEv = static_cast<QMouseEvent *>(e);
            QRect xSceneRect(0, 0, MAX_X, MAX_Y);
            if(xSceneRect.contains(pEv->pos()))
            {
                QRect xAlertRect((MAX_X - rect().width()) / 2, (MAX_Y - rect().height()) / 2, rect().width(), rect().height());
                if(!xAlertRect.contains(pEv->pos()))
                {
                    m_xExecLoop.exit(0);
                }
            }
        }
    }

    return false;
}


int AlertDlg::WarningYesNo(QGraphicsView* pView, FormBase* pParentForm, QString sTitle, QString sMsg)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call] AlertDlg" << __FUNCTION__;
#endif

    QLabel* lblMessage = new QLabel;
    lblMessage->setWordWrap(true);
    lblMessage->setAlignment(Qt::AlignCenter);

    QString sSheets;
    sSheets.sprintf("background-color: rgb(%d, %d, %d);color: rgb(%d, %d, %d);", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue(),
                   g_UITheme->itemMainTextColor.red(), g_UITheme->itemMainTextColor.green(), g_UITheme->itemMainTextColor.blue());

    lblMessage->setStyleSheet(sSheets);
    lblMessage->setText(sMsg);
    lblMessage->setFont(g_UITheme->MainTextFont);
    lblMessage->setAttribute(Qt::WA_DeleteOnClose);

    QFontMetrics xFontMetrics(lblMessage->font());
    int iMsgHeight = xFontMetrics.boundingRect(sMsg).height();
    lblMessage->setGeometry(0, 0, 188, iMsgHeight + 32);

    AlertDlg* pDlg = new AlertDlg(pView, pParentForm);
    pDlg->setFixedWidth(220 * LCD_RATE);
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_Yes);
    pDlg->SetCancelButton(true, StringTable::Str_No);
    pDlg->SetTitle(sTitle);
    pDlg->AddWidget(lblMessage);
    pDlg->setAttribute(Qt::WA_DeleteOnClose);
    int iRet = pDlg->OnExec();

    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    pDlgScene->clear();
    delete pDlgScene;

    return iRet;
}

int AlertDlg::WarningOk(QGraphicsView* pView, FormBase* pParentForm, QString sTitle, QString sMsg, int multiLine)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call] AlertDlg" << __FUNCTION__;
#endif

    QLabel* lblMessage = new QLabel;
    lblMessage->setWordWrap(true);
    lblMessage->setAlignment(Qt::AlignCenter);
    lblMessage->setFont(g_UITheme->PrimaryFont);

    QString sSheets;
    sSheets.sprintf("background-color: rgb(%d, %d, %d);color: rgb(%d, %d, %d);", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue(),
                   g_UITheme->itemMainTextColor.red(), g_UITheme->itemMainTextColor.green(), g_UITheme->itemMainTextColor.blue());

    lblMessage->setStyleSheet(sSheets);
    lblMessage->setText(sMsg);
    lblMessage->setFont(g_UITheme->MainTextFont);
    lblMessage->setAttribute(Qt::WA_DeleteOnClose);

    QFontMetrics xFontMetrics(lblMessage->font());
    int iMsgHeight = xFontMetrics.boundingRect(sMsg).height();
    lblMessage->setGeometry(0, 0, 188, iMsgHeight + 32);

    AlertDlg* pDlg = new AlertDlg(pView, pParentForm);
    pDlg->setFixedWidth(220 * LCD_RATE);
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_OK);
    pDlg->SetCancelButton(false, StringTable::Str_Cancel);
    pDlg->SetTitle(sTitle);
    pDlg->AddWidget(lblMessage);
    pDlg->setAttribute(Qt::WA_DeleteOnClose);

    int iRet = pDlg->OnExec();

    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    pDlgScene->clear();
    delete pDlgScene;

    return iRet;
}

QString AlertDlg::ContainLineEdit(QGraphicsView* pView, FormBase* pParentForm, QString strTitle, QString strText, int maxLen,
                                  int mode, QString _input_mask)
{
    g_fCustomKeyMode = mode;
    CustomLineEdit* pLineEdit = new CustomLineEdit;
    pLineEdit->setGeometry(0, 0, 194 * LCD_RATE,  30 * LCD_RATE);
    pLineEdit->setFrame(false);
    pLineEdit->setMaxLength(maxLen);
    pLineEdit->setInputMask(_input_mask);
    pLineEdit->sendIMENull(1);
    pLineEdit->setAttribute(Qt::WA_DeleteOnClose);

    QString strSheets;
    strSheets.sprintf("background-color: rgb(%d, %d, %d);color: rgb(%d, %d, %d);", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue(),
                   g_UITheme->itemMainTextColor.red(), g_UITheme->itemMainTextColor.green(), g_UITheme->itemMainTextColor.blue());
    pLineEdit->setStyleSheet(strSheets);


    pLineEdit->setFont(g_UITheme->PrimaryFont);
    pLineEdit->setBorder(true);
    pLineEdit->setText(strText);

    AlertDlg* pDlg = new AlertDlg(pView, pParentForm);
    pDlg->setFixedWidth(210 * LCD_RATE);
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_OK);
    pDlg->SetCancelButton(false, StringTable::Str_Cancel);
    pDlg->SetTitle(strTitle);
    pDlg->AddWidget(pLineEdit);
    pDlg->SetLineEdit(pLineEdit);
    pDlg->setAttribute(Qt::WA_DeleteOnClose);

    pLineEdit->setParent(pDlg);
    pDlg->OnExec(1);
    QString strRet = pLineEdit->text();

    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    pDlgScene->clear();
    delete pDlgScene;

    return strRet;
}


void AlertDlg::AutoTest()
{
    int idx = rand() % 2;
    m_xExecLoop.exit(idx);
}
