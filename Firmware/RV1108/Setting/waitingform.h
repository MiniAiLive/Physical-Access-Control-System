#ifndef WAITINGSCENE_H
#define WAITINGSCENE_H

#include "formbase.h"
#include <QLabel>
#include <QThread>
#include <QEventLoop>
#include <QRunnable>

#include <functional>
using namespace std;

typedef int (*WaitingFunc)();

class QMovie;
class QGraphicsView;
class QGraphicsScene;
class QGraphicsPixmapItem;
class QGraphicsProxyWidget;
class FormBase;
class WaitingForm : public FormBase, public QRunnable
{
    Q_OBJECT
public:
    explicit WaitingForm(QGraphicsView* pView, FormBase* pParentView);
    ~WaitingForm();

    void    Exec(std::function<void(void)> fnFunc, bool bCancel);

    void    run();

    static void     Waiting(QGraphicsView* pView, FormBase* pParentForm, std::function<void(void)> fnFunc);
    static void     WaitingForCancel(QGraphicsView* pView, FormBase* pParentForm, std::function<void(void)> fnFunc, bool bCancel);
    static void		FreeScene();

signals:

public slots:

protected:
    void    timerEvent(QTimerEvent *);
    void    paintEvent(QPaintEvent *);
    void    mousePressEvent(QMouseEvent* e);

private:
    int                     m_nTimer;
    int                     m_nIndex;
    QStringList             m_vMovieNames;
    QPixmap                 m_xBackPix;

    std::function<void(void)>   m_fnWaitingFunc1;

    bool                    m_bCancel;
};

#endif // WAITINGITEM_H
