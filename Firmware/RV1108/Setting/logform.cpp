#include "logform.h"
#include "stringtable.h"
#include "DBManager.h"
#include "base.h"
#include "menuitem.h"
#include "shared.h"
#include "i2cbase.h"
#include "uitheme.h"

#include <QtGui>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QScrollBar>

LogForm::LogForm(QGraphicsView *pView, FormBase* pParentForm) :
    ItemFormBase(pView, pParentForm)
{
    SetBGColor(g_UITheme->itemNormalBgColor);
}

LogForm::~LogForm()
{
}

void LogForm::OnStart()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    InitLogView();
    FormBase::OnStart();
}

void LogForm::BackClick()
{
    emit SigBack();
}

void LogForm::PressedLog(int fState)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    if(fState == 0)
    {
        m_pFaceImageItem->setVisible(false);
        return;
    }

    MenuItem* pMenuItem = qobject_cast<MenuItem*>(sender());
    if(pMenuItem == NULL)
        return;

    int iLogIndex = pMenuItem->data(KEY_DATA2).toInt();

    SLogInfo xLogInfo = { 0 };
    dbm_GetLogInfo(iLogIndex, &xLogInfo);

    if(xLogInfo.iImageLen == 0 || xLogInfo.abFaceImg[0] == 0)
        m_pFaceImageItem->setVisible(false);
    else
    {
        m_pFaceImageItem->setVisible(true);
        m_pFaceImageItem->setData(KEY_ICON, ConvertData2QImage(xLogInfo.abFaceImg, xLogInfo.iImageLen));
    }
}


void LogForm::InitLogView()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    m_pScene->clear();
    m_vLogItems.clear();

    int iLogCount = dbm_GetLogCount();
    for(int i = 0; i < iLogCount; i ++)
    {
        MenuItem* pLogItem = new MenuItem();
        pLogItem->setPos(QPoint(0, i * LOG_ITEM_HEIGHT));
        pLogItem->SetBoundingRect(QRect(0, 0, LOG_ITEM_WIDTH, LOG_ITEM_HEIGHT));        
        pLogItem->setData(KEY_DATA2, iLogCount - i - 1);
        pLogItem->setData(KEY_TYPE, TYPE_LOG_ITEM);
        m_pScene->addItem(pLogItem);
        m_vLogItems.append(pLogItem);

        connect(pLogItem, SIGNAL(pressed(int)), this, SLOT(PressedLog(int)));
    }

    m_pFaceImageItem = new MenuItem;
    m_pFaceImageItem->SetBoundingRect(QRect(0, 0, LOG_FACE_ITEM_WIDTH, LOG_FACE_ITEM_HEIGHT));
    m_pFaceImageItem->setPos(QPoint(190, 0));
    m_pFaceImageItem->setData(KEY_TYPE, TYPE_LOG_FACE_ITEM);
    m_pFaceImageItem->setData(KEY_FIX_POS, QPoint(190, 0));
    m_pFaceImageItem->setZValue(1);
    m_pScene->addItem(m_pFaceImageItem);

    connect(GetItemView()->verticalScrollBar(), SIGNAL(valueChanged(int)), m_pFaceImageItem, SLOT(ScrollChnaged(int)));

    QGraphicsDropShadowEffect* pEffect = new QGraphicsDropShadowEffect;
    pEffect->setColor(QColor(0, 0, 0));
    pEffect->setOffset(2, 2);
    pEffect->setBlurRadius(8);

    m_pFaceImageItem->setGraphicsEffect(pEffect);
    PressedLog(0);

    RetranslateUI();
}


void LogForm::RetranslateUI()
{
    QString strLogNum = "<span style=\" font-size:4pt; color:#ffffff;\"> (" + QString::number(dbm_GetLogCount()) + ")</span>";
    SetTitle(StringTable::Str_Log + strLogNum);
}

