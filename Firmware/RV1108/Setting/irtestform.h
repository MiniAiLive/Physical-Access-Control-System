#ifndef IRTESTFORM_H
#define IRTESTFORM_H

#include "formbase.h"
#include <QRunnable>
#include <QThread>
#include <QMutex>

class QLabel;

class IRTestForm : public FormBase, public QRunnable
{
    Q_OBJECT
public:
    explicit IRTestForm(QGraphicsView *pView, FormBase* pParentForm);
    ~IRTestForm();

    void    StartTest();
    void    run();
signals:
    void    SigBack(int iCamError);

public slots:
    void    OnPause();

protected:
    void    RetranslateUI();
    bool    event(QEvent* e);

private:
    QMutex      m_xMutex;
    int         m_iRunning;
};


#endif // ENROLLFACEFORM_H
