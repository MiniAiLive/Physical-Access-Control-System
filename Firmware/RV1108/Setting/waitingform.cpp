#include "waitingform.h"

#include "base.h"
#include "appdef.h"
#include "settings.h"
#include "themedef.h"

#include <QtGui>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <unistd.h>
#include <pthread.h>

WaitingForm* g_pWaitingInstance = NULL;
QGraphicsScene* g_pWaitingScene = NULL;

#define MOVIE_SIZE (50 * LCD_RATE)

WaitingForm::WaitingForm(QGraphicsView* pView, FormBase* pParentForm) :
    FormBase(pView, pParentForm)
{
    m_nTimer = 0;
    m_nIndex = 0;
    m_bCancel = false;

    for(int i = 0; i < 12; i ++)
    {
        QString strFileName;
        strFileName.sprintf(":/icons/W%02d.png", i);

        m_vMovieNames.append(strFileName);
    }

    setAutoDelete(false);
    m_pSurfaceScene = new QGraphicsScene;
    m_pSurfaceScene->addWidget(this);
}

WaitingForm::~WaitingForm()
{
}

void WaitingForm::Waiting(QGraphicsView* pView, FormBase* pParentForm, std::function<void(void)> fnFunc)
{
    WaitingForm* pWaitingForm = new WaitingForm(pView, pParentForm);
    pWaitingForm->setAttribute(Qt::WA_DeleteOnClose);
    pWaitingForm->Exec(fnFunc, false);

    QGraphicsScene* pSurfaceScene = pWaitingForm->GetSurfaceScene();
    pSurfaceScene->clear();
    delete pSurfaceScene;
}

void WaitingForm::WaitingForCancel(QGraphicsView* pView, FormBase* pParentForm, std::function<void(void)> fnFunc, bool bCancel)
{
    WaitingForm* pWaitingForm = new WaitingForm(pView, pParentForm);
    pWaitingForm->setAttribute(Qt::WA_DeleteOnClose);
    pWaitingForm->Exec(fnFunc, bCancel);

    QGraphicsScene* pSurfaceScene = pWaitingForm->GetSurfaceScene();
    pSurfaceScene->clear();
    delete pSurfaceScene;
}

void WaitingForm::FreeScene()
{
    if(g_pWaitingInstance)
    {
        QGraphicsScene* pSurfaceScene = g_pWaitingInstance->GetSurfaceScene();
        pSurfaceScene->clear();
        delete pSurfaceScene;
    }
}

void WaitingForm::Exec(std::function<void(void)> fnFunc, bool bCancel)
{
    if(m_pParentView == NULL)
        return;

    if(m_pParentForm != NULL)
    {
        QPixmap xBackPix(size());
        QPixmap xTmpBackPix(m_pParentForm->size());
        m_pParentForm->render(&xTmpBackPix);

        QPainter painter;
        painter.begin(&xBackPix);
        painter.fillRect(xBackPix.rect(),QColor(0, 0, 0));
        painter.setOpacity(0.4);
        painter.drawPixmap(0, 0, xTmpBackPix);
        painter.end();

        m_xBackPix = xBackPix;
    }
    else
    {
        QPixmap xBackPix(size());
        QPainter painter;
        painter.begin(&xBackPix);
        painter.fillRect(xBackPix.rect(), QColor(40, 40, 40));
        painter.end();

        m_xBackPix = xBackPix;
    }

    g_xSS.iWaitingCacnel = 0;
    m_bCancel = bCancel;
    m_fnWaitingFunc1 = fnFunc;

    m_nIndex = 0;
    m_nTimer = startTimer(200);

    QGraphicsScene* pPrevScene = m_pParentView->scene();
    m_pParentView->setScene(m_pSurfaceScene);

    QThreadPool::globalInstance()->start(this);
    m_xExecLoop.exec();

    killTimer(m_nTimer);
    m_pParentView->setScene(pPrevScene);

    QThreadPool::globalInstance()->waitForDone();
}

void WaitingForm::run()
{
    m_fnWaitingFunc1();
    m_xExecLoop.exit();
}

void WaitingForm::timerEvent(QTimerEvent *e)
{
    QWidget::timerEvent(e);

    if(e->timerId() == m_nTimer)
    {
        m_nIndex ++;
        if(m_nIndex >= m_vMovieNames.size())
            m_nIndex = 0;

        update();
    }
}

void WaitingForm::paintEvent(QPaintEvent *e)
{
    QPainter painter;
    painter.begin(this);

    painter.drawPixmap(rect(), m_xBackPix);

    QImage movieImage(m_vMovieNames[m_nIndex]);
    QImage scaledImage = movieImage.scaled(QSize(MOVIE_SIZE, MOVIE_SIZE), Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

    painter.drawPixmap((rect().width() - MOVIE_SIZE) / 2, (rect().height() - MOVIE_SIZE) / 2, QPixmap::fromImage(scaledImage));

    painter.end();
}

void WaitingForm::mousePressEvent(QMouseEvent* e)
{
    QWidget::mousePressEvent(e);

    if(m_bCancel)
    {
        g_xSS.iWaitingCacnel = 1;
    }
}
