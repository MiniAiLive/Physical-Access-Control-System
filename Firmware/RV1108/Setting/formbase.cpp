#include "formbase.h"
#include "base.h"
#include "appdef.h"
#include "mainwindow.h"
#include "shared.h"
#include "imerequestevent.h"
#include "customkeyinputpanel.h"
#include "camera_api.h"
#include "soundbase.h"
#include "themedef.h"

#include <QtGui>
#include <QGraphicsProxyWidget>
#include <QGraphicsDropShadowEffect>

FormBase* FormBase::CurForm = NULL;
int FormBase::QuitFlag = 0;

FormBase::FormBase(QGraphicsView *view, FormBase* parentForm)
    : QWidget(0),
    m_pParentView(view),
    m_pParentForm(parentForm)
{
    m_pSurfaceScene = NULL;
    m_pLineEdit = NULL;

//    m_pAnimationScene = new QGraphicsScene;
//    m_pSceneStartAnimationPixmapItem = m_pAnimationScene->addPixmap(QPixmap(MAX_X, MAX_Y));

//    m_pSceneAnimationProperty = new QPropertyAnimation(this, "GetSceneAnimation");
//    m_pSceneAnimationProperty->setEasingCurve(QEasingCurve::OutQuad);
//    connect(m_pSceneAnimationProperty, SIGNAL(finished()), this, SLOT(SceneAnimationFinished()));

    m_pProxyIME = NULL;
    m_iShowIME = 0;
    m_pImePanel = NULL;

//    m_pImeAnimationProperty = new QPropertyAnimation(this, "GetImeAnimation");
//    m_pImeAnimationProperty->setEasingCurve(QEasingCurve::OutQuad);
//    connect(m_pImeAnimationProperty, SIGNAL(finished()), this, SLOT(IMEAnimationFinished()));
    connect(this, SIGNAL(SigShowIME()), this, SLOT(ShowIME()), Qt::QueuedConnection);

    resize(MAX_X, MAX_Y);

    if(m_pParentForm)
        connect(this, SIGNAL(SigQuit()), m_pParentForm, SLOT(OnResume()));
}

FormBase::~FormBase()
{
//    m_pAnimationScene->clear();
//    delete m_pAnimationScene;
//    delete m_pSceneAnimationProperty;
//    delete m_pImeAnimationProperty;
}

QGraphicsScene* FormBase::GetSurfaceScene()
{
    return m_pSurfaceScene;
}

void FormBase::ShowIME(QLineEdit* pLineEdit, QPoint xEditPos)
{
    if(m_pImePanel != NULL)
        return;

    m_pImePanel = new CustomKeyInputPanel;

    if(pLineEdit != NULL)
    {
        QPoint xLineEditPos(xEditPos - m_pParentView->mapToGlobal(QPoint(0, 0)));
        for(int i = 0; i < m_pSurfaceScene->items().size(); i ++)
        {
            QGraphicsItem* pItem = m_pSurfaceScene->items().at(i);
            QGraphicsProxyWidget* pProxyWidget = qgraphicsitem_cast<QGraphicsProxyWidget*>(pItem);
            if(pProxyWidget == NULL)
                continue;

            if(pProxyWidget->widget() == this)
            {
                m_pProxyForm = pProxyWidget;
                if(xLineEditPos.y() + pLineEdit->height() > (MAX_Y - (m_pImePanel->height() + CAND_PANEL_HEIGHT)))
                {
                    m_xOldFormPos = pProxyWidget->pos().toPoint();
                    m_xStartFormPos = m_xOldFormPos;
                    m_xEndFormPos = QPoint(m_xOldFormPos.x(), m_xOldFormPos.y() + (MAX_Y - (m_pImePanel->height() + CAND_PANEL_HEIGHT)) - (xLineEditPos.y() + pLineEdit->height()));
                }
                else
                {
                    m_xStartFormPos = pProxyWidget->pos().toPoint();
                    m_xEndFormPos = m_xStartFormPos;
                    m_xOldFormPos = m_xEndFormPos;
                }
            }
        }
    }
    else
    {
        QPoint xLineEditPos(QPoint(MAX_X / 2, MAX_Y / 2 + 25 * LCD_RATE) - m_pParentView->mapToGlobal(QPoint(0, 0)));
        for(int i = 0; i < m_pSurfaceScene->items().size(); i ++)
        {
            QGraphicsItem* pItem = m_pSurfaceScene->items().at(i);
            QGraphicsProxyWidget* pProxyWidget = qgraphicsitem_cast<QGraphicsProxyWidget*>(pItem);
            if(pProxyWidget == NULL)
                continue;

            if(pProxyWidget->widget() == this)
            {
                m_pProxyForm = pProxyWidget;
                if(MAX_Y / 2 + 25 * LCD_RATE > (MAX_Y- (m_pImePanel->height() + CAND_PANEL_HEIGHT)))
                {
                    m_xOldFormPos = pProxyWidget->pos().toPoint();
                    m_xStartFormPos = m_xOldFormPos;
                    m_xEndFormPos = QPoint(m_xOldFormPos.x(), m_xOldFormPos.y() + (MAX_Y - (m_pImePanel->height() + CAND_PANEL_HEIGHT)) - (xLineEditPos.y()));
                }
                else
                {
                    m_xStartFormPos = pProxyWidget->pos().toPoint();
                    m_xEndFormPos = m_xStartFormPos;
                    m_xOldFormPos = m_xEndFormPos;
                }
            }
        }
    }


    m_pImePanel->setEnabled(false);
    setEnabled(false);

    m_xStartIMEPos = QPoint(0, MAX_Y - (m_pImePanel->height() + CAND_PANEL_HEIGHT) / 2);
    m_xEndIMEPos = QPoint(0, MAX_Y - m_pImePanel->height());

    m_pProxyIME = m_pSurfaceScene->addWidget(m_pImePanel);
    m_pProxyIME->setPos(0, m_pParentView->height());

//    m_pImeAnimationProperty->setStartValue(m_xStartIMEPos.y());
//    m_pImeAnimationProperty->setEndValue(m_xEndIMEPos.y());
//    m_pImeAnimationProperty->setDuration(100);
//    m_pImeAnimationProperty->setLoopCount(1); // forever
//    m_pImeAnimationProperty->start();

//    m_xImeAnimationLoop.exec();

    IMEAnimationFinished();

    if(pLineEdit)
        m_pImePanel->startIME(this, pLineEdit);
    else
        m_pImePanel->startIME(this, m_pLineEdit);
    m_pImePanel->setEnabled(true);
    setEnabled(true);

    if(pLineEdit)
        pLineEdit->setFocus();
    else
        m_pLineEdit->setFocus();
}

void FormBase::HideIME()
{
    if(m_pImePanel == NULL)
        return;

    setEnabled(false);
    m_pImePanel->setEnabled(false);
    m_xStartIMEPos = QPoint(0, m_pParentView->height() - (m_pImePanel->height() + CAND_PANEL_HEIGHT) / 2);
    m_xEndIMEPos = QPoint(0, m_pParentView->height());
    m_xStartFormPos = m_xEndFormPos;
    m_xEndFormPos = m_xOldFormPos;

//    m_pImeAnimationProperty->setStartValue(m_xStartIMEPos.y());
//    m_pImeAnimationProperty->setEndValue(m_xEndIMEPos.y());
//    m_pImeAnimationProperty->setDuration(100);
//    m_pImeAnimationProperty->setLoopCount(1); // forever
//    m_pImeAnimationProperty->start();

//    m_xImeAnimationLoop.exec();

    IMEAnimationFinished();

    m_pProxyForm->setPos(m_xEndFormPos);

    m_pImePanel->stopIME();
    m_pSurfaceScene->removeItem(m_pProxyIME);
    m_pSurfaceScene->setSceneRect(QRect(0, 0, MAX_X, MAX_Y));
    m_pImePanel->deleteLater();
    m_pImePanel = NULL;
    setEnabled(true);


}

int FormBase::GetImeAnimation()
{
    if(m_pProxyIME)
        return m_pProxyIME->pos().toPoint().y();

    return 0;
}

void FormBase::SetImeAnimation(int iImeAnimation)
{
    if(m_pProxyIME == NULL)
        return;

//    m_pProxyIME->setPos(0, iImeAnimation);

//    float rOpacity = (abs(iImeAnimation - m_pImeAnimationProperty->startValue().toInt())) / (float)abs((m_pImeAnimationProperty->endValue().toInt() - m_pImeAnimationProperty->startValue().toInt()));
//    m_pProxyForm->setPos(m_xEndFormPos.x(), (m_xEndFormPos.y() - m_xStartFormPos.y()) * rOpacity + m_xStartFormPos.y());
}

void FormBase::ShowIME()
{
    qDebug() << "FormBase::ShowIME";
    ShowIME(NULL, QPoint(0, 0));
}

void FormBase::OnStart(int fDelPrevScene)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    if(m_pSurfaceScene == NULL)
        m_pSurfaceScene = new QGraphicsScene;

    RetranslateUI();
    m_pSurfaceScene->clear();

    if(size().width() == MAX_X && size().height() == MAX_Y)
        m_pSurfaceScene->addWidget(this);
    else
    {
        QPixmap xTmpBackPix(size());
        render(&xTmpBackPix);

        QPixmap xBackPix(MAX_X, MAX_Y);
        QPainter xPainter;
        xPainter.begin(&xBackPix);
        xPainter.fillRect(xBackPix.rect(),QColor(0, 0, 0));
        if(m_pParentForm)
        {
            xPainter.setOpacity(0.4);
            render(&xPainter);
        }
        xPainter.end();

        m_pSurfaceScene->addPixmap(xBackPix);
        m_pSurfaceScene->addWidget(this);
    }

    if(m_pParentForm)
        m_pParentForm->OnPause();

    ChangeForm(this, m_pParentForm, ANIMATION_TYPE_RIGHT, fDelPrevScene);
}

void FormBase::OnResume()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    qDebug() << "OnResume" << this;

    FormBase* pSenderForm = qobject_cast<FormBase*>(sender());
    QGraphicsScene* pSenderScene = pSenderForm->GetSurfaceScene();

    RetranslateUI();
    pSenderForm->OnPause();

    ChangeForm(this, pSenderForm, ANIMATION_TYPE_LEFT);
    pSenderForm->OnStop();

    connect(pSenderScene, SIGNAL(destroyed()), pSenderScene, SLOT(clear()));
    pSenderScene->deleteLater();

    if(FormBase::QuitFlag)
        Quit();
}

int FormBase::OnExec(int iShowIME)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif
    qDebug() << "OnExec" << this;

    if(m_pSurfaceScene == NULL)
        m_pSurfaceScene = new QGraphicsScene;

    RetranslateUI();
    m_pSurfaceScene->clear();

    QPixmap xBackPix(MAX_X, MAX_Y);
    QPainter xPainter;
    xPainter.begin(&xBackPix);
    xPainter.fillRect(xBackPix.rect(),QColor(0, 0, 0));
    if(m_pParentForm)
    {
        xPainter.setOpacity(0.4);
        m_pParentForm->render(&xPainter);
    }

    xPainter.end();

    m_pSurfaceScene->addPixmap(xBackPix);

    QGraphicsProxyWidget* pProxyWidget = m_pSurfaceScene->addWidget(this);
    pProxyWidget->setPos((MAX_X - width()) / 2, (MAX_Y - height()) / 2);

    QGraphicsDropShadowEffect* pEffect = new QGraphicsDropShadowEffect;
    pEffect->setColor(QColor(0, 0, 0));
    pEffect->setOffset(2, 2);
    pEffect->setBlurRadius(8);
    pProxyWidget->setGraphicsEffect(pEffect);

    if(m_pParentForm)
        m_pParentForm->OnPause();

    m_iShowIME = iShowIME;
    ChangeForm(this, m_pParentForm, ANIMATION_TYPE_EXEC);
    int iRet = m_xExecLoop.exec();

    OnPause();
    HideIME();
    ChangeForm(m_pParentForm, this, ANIMATION_TYPE_DONE);
    OnStop();

    if(FormBase::QuitFlag)
        Quit();

    return iRet;
}

void FormBase::Quit()
{
    if(m_xExecLoop.isRunning())
    {
        FormBase::QuitFlag = 1;
        g_xSS.iNoSoundPlayFlag = 1;
        m_xExecLoop.exit(0);
    }
    else if(m_pParentForm != NULL)
    {
        FormBase::QuitFlag = 1;
        g_xSS.iNoSoundPlayFlag = 1;
        emit SigQuit();
    }
    else
    {
        FormBase::QuitFlag = 0;
    }
}

void FormBase::ChangeForm(FormBase* pNewForm, FormBase* pPrevForm, int iAnimationType, int iDelPrevScene)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

//    m_iAnimationType = iAnimationType;
    m_pNewForm = pNewForm;
    if(m_pParentView->scene() && pPrevForm != NULL)
    {
#if 0
        m_pSceneAnimationProperty->stop();

        m_xSceneStartAnimationPixmap = QPixmap(MAX_X, MAX_Y);
        m_xSceneEndAnimationPixmap = QPixmap(MAX_X, MAX_Y);
        m_xSceneEndAnimationPixmap.fill(Qt::black);
        m_xSceneStartAnimationPixmap.fill(Qt::black);

        if(iAnimationType == ANIMATION_TYPE_RIGHT)
        {
            pPrevForm->render(&m_xSceneStartAnimationPixmap);
            pNewForm->render(&m_xSceneEndAnimationPixmap);

            m_pSceneAnimationProperty->setStartValue(0);
            m_pSceneAnimationProperty->setEndValue(100);
        }
        else if(iAnimationType == ANIMATION_TYPE_LEFT)
        {
            pPrevForm->render(&m_xSceneStartAnimationPixmap);
            pNewForm->render(&m_xSceneEndAnimationPixmap);

            m_pSceneAnimationProperty->setStartValue(0);
            m_pSceneAnimationProperty->setEndValue(100);
        }
        else if(iAnimationType == ANIMATION_TYPE_EXEC)
        {
            pPrevForm->render(&m_xSceneStartAnimationPixmap);

            QPainter painter;
            painter.begin(&m_xSceneEndAnimationPixmap);
            m_pSurfaceScene->render(&painter);
            painter.end();

            m_pSceneAnimationProperty->setStartValue(70);
            m_pSceneAnimationProperty->setEndValue(100);
        }
        else if(iAnimationType == ANIMATION_TYPE_DONE)
        {
            pNewForm->render(&m_xSceneEndAnimationPixmap);

            QPainter painter;
            painter.begin(&m_xSceneStartAnimationPixmap);
            m_pSurfaceScene->render(&painter);
            painter.end();

            m_pSceneAnimationProperty->setStartValue(100);
            m_pSceneAnimationProperty->setEndValue(0);
        }

        m_pSceneStartAnimationPixmapItem->setPixmap(m_xSceneStartAnimationPixmap);

        if(iDelPrevScene && m_pParentView->scene())
        {
            connect(m_pParentView->scene(), SIGNAL(destroyed()), m_pParentView->scene(), SLOT(clear()));
            m_pParentView->scene()->deleteLater();
        }
        m_pParentView->setScene(m_pAnimationScene);

        m_pSceneAnimationProperty->setDuration(100);
//        m_pSceneAnimationProperty->setDuration(0);
        m_pSceneAnimationProperty->setLoopCount(1);
        m_pSceneAnimationProperty->start();

        PlaySoundLeft();
        m_xAnimationLoop.exec();
#else
        PlaySoundLeft();
        SceneAnimationFinished();
#endif
    }
    else
    {
        m_pParentView->setScene(m_pSurfaceScene);
    }

    CurForm = pNewForm;
}

void FormBase::SceneAnimationFinished()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    m_pParentView->setScene(m_pNewForm->GetSurfaceScene());
//    m_xAnimationLoop.exit(0);

    if(m_iShowIME == 1)
    {
//        emit SigShowIME();
        QTimer::singleShot(0, this, SLOT(ShowIME()));
    }
}

int FormBase::GetSceneAnimation()
{
//    return m_iSceneAnimationValue;
    return -1;
}

void FormBase::SetSceneAnimation(int iSceneAnimation)
{
//    m_iSceneAnimationValue = iSceneAnimation;
    return;
#if 0
    if(iSceneAnimation != 0)
    {
        QPixmap xAnimationPixmap(MAX_X, MAX_Y);

        if(m_iAnimationType == ANIMATION_TYPE_RIGHT)
        {
            QPainter xPainter;
            xPainter.begin(&xAnimationPixmap);

            QRect drawRect(m_xSceneStartAnimationPixmap.rect());
            drawRect.moveTo(m_xSceneStartAnimationPixmap.rect().left() - iSceneAnimation * MAX_X / 100, 0);

            xPainter.drawPixmap(drawRect, m_xSceneStartAnimationPixmap);

            drawRect = m_xSceneEndAnimationPixmap.rect();
            drawRect.moveTo(m_xSceneEndAnimationPixmap.rect().left() + ((100 - iSceneAnimation) * MAX_X / 100), 0);
            xPainter.drawPixmap(drawRect, m_xSceneEndAnimationPixmap);

            xPainter.end();

            m_pSceneStartAnimationPixmapItem->setPixmap(xAnimationPixmap);
        }
        else if(m_iAnimationType == ANIMATION_TYPE_LEFT)
        {
            QPainter xPainter;
            xPainter.begin(&xAnimationPixmap);

            QRect xDrawRect(m_xSceneStartAnimationPixmap.rect());
            xDrawRect.moveTo(m_xSceneStartAnimationPixmap.rect().left() + iSceneAnimation * MAX_X / 100, 0);

            xPainter.drawPixmap(xDrawRect, m_xSceneStartAnimationPixmap);

            xDrawRect = m_xSceneEndAnimationPixmap.rect();
            xDrawRect.moveTo(m_xSceneEndAnimationPixmap.rect().left() - ((100 - iSceneAnimation) * MAX_X / 100), 0);
            xPainter.drawPixmap(xDrawRect, m_xSceneEndAnimationPixmap);

            xPainter.end();

            m_pSceneStartAnimationPixmapItem->setPixmap(xAnimationPixmap);
        }
        else if(m_iAnimationType == ANIMATION_TYPE_EXEC)
        {
            return;
            QPainter xPainter;
            xPainter.begin(&xAnimationPixmap);

            xPainter.drawPixmap(m_xSceneStartAnimationPixmap.rect(), m_xSceneStartAnimationPixmap);

            xPainter.setOpacity((float)iSceneAnimation / 100);
            xPainter.drawPixmap(m_xSceneStartAnimationPixmap.rect(), m_xSceneEndAnimationPixmap);

            xPainter.end();

            m_pSceneStartAnimationPixmapItem->setPixmap(xAnimationPixmap);
        }
        else if(m_iAnimationType == ANIMATION_TYPE_DONE)
        {
            return;
            QPainter xPainter;
            xPainter.begin(&xAnimationPixmap);

            xPainter.drawPixmap(m_xSceneEndAnimationPixmap.rect(), m_xSceneEndAnimationPixmap);

            QRect xAnimationRect = QRect(0, 0, iSceneAnimation * MAX_X / 100, iSceneAnimation * rect().height() / 100);
            QRect xDrawRect = QRect((MAX_X - xAnimationRect.width()) / 2, (rect().height() - xAnimationRect.height()) / 2, xAnimationRect.width(), xAnimationRect.height());

            xPainter.setOpacity((float)iSceneAnimation / 100);
            xPainter.drawPixmap(m_xSceneStartAnimationPixmap.rect(), m_xSceneStartAnimationPixmap);

            xPainter.end();

            m_pSceneStartAnimationPixmapItem->setPixmap(xAnimationPixmap);
        }
    }
#endif
}

void FormBase::RetranslateUI()
{

}

void FormBase::changeEvent(QEvent* e)
{
    QWidget::changeEvent(e);
    if(e->type() == QEvent::LanguageChange)
        RetranslateUI();
}

bool FormBase::event(QEvent* e)
{
    if(e->type() == EV_IME_REQUEST)
    {
        IMERequestEvent* imeEvent = static_cast<IMERequestEvent*>(e);
        ShowIME(imeEvent->m_edit, imeEvent->m_pos);
    }
    else if(e->type() == EV_IME_CLOSE)
    {
        HideIME();
    }
    else if(e->type() == EV_IME_NUM_REQUEST)
    {
    }
    else if(e->type() == EV_IME_NUM_CLOSE)
    {
    }

    return QWidget::event(e);
}


KeyEvent::KeyEvent(int iKeyID, int iEvType)
    : QEvent(EV_KEY_EVENT)
{
    m_iKeyID = iKeyID;
    m_iEvType = iEvType;
}

CardSettingsEvent::CardSettingsEvent(QString strSettings)
    : QEvent(EV_CARD_SETTINGS_EVENT)
{
    m_strCardSettings = strSettings;
}
