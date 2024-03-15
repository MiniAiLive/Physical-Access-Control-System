#ifndef ENROLLCARDFORM_H
#define ENROLLCARDFORM_H

#include "formbase.h"
#include <QRunnable>
#include <QThread>
#include <QMutex>

class QLabel;
class QVBoxLayout;
class EnrollCardForm : public FormBase, public QRunnable
{
    Q_OBJECT
public:
    explicit EnrollCardForm(QGraphicsView *pView, FormBase* pParentForm);
    ~EnrollCardForm();

    void    StartEnroll(int iTimeout);
    int     GetEnrolledCardID();
    int     GetEnrolledSectorNum();
    int     GetEnrolledCardRand();

    void    run();

signals:
    void    SigSendEnrollFinished(int iEnrollResult);

public slots:
    void    OnPause();

protected:
    void    mousePressEvent(QMouseEvent* e);
    void    RetranslateUI();
    bool    event(QEvent* e);

private:
    int     m_fRunning;
    int     m_iCardID;
    int     m_iSectorNum;
    int     m_iCardRand;
    int     m_iTimeout;

    QLabel* m_lblTitle;
    QLabel* m_lblCard;
    QVBoxLayout*    m_lytMain;
};


#endif // ENROLLFACEFORM_H
