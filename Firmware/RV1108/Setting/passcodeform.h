#ifndef SETPASSCODEITEM_H
#define SETPASSCODEITEM_H

#include "formbase.h"
#include <QTime>
#include <QRunnable>

namespace Ui {
class PasscodeForm;
}

enum
{
    STEP_INPUT_OLD,
    STEP_INPUT_NEW,
    STEP_INPUT_CONFIRM,
    STEP_CONFIRM,
};

class PasscodeForm : public FormBase, public QRunnable
{
    Q_OBJECT
    
public:
    explicit PasscodeForm(QGraphicsView *pView, FormBase* pParentForm);
    ~PasscodeForm();

    void    Start(int iStep, QString strOldPasscode);
    void    Next();
    void    Prev();

    QString GetPasscode();
    QVector<QPoint> GetChildPos();

    void    run();
signals:
    void    SigConfirm();
    void    SigHiddenCode();

public slots:
    void    Click0();
    void    Click1();
    void    Click2();
    void    Click3();
    void    Click4();
    void    Click5();
    void    Click6();
    void    Click7();
    void    Click8();
    void    Click9();
    void    DelClick();
    void    OkClick();
    void    ClickBack();

    void    OnPause();

protected:
    void    timerEvent(QTimerEvent*);

private:
    void    AddPasscode(QChar);
    void    ConnectButtons(int iConnect);

protected:
    void    RetranslateUI();
    
private:
    Ui::PasscodeForm *ui;

    QString m_strOldPasscode;
    QString m_strNewPasscode;

    QString m_strPasscode;
    int     m_iStep;

    int     m_iHiddenCodeFlag;
    float   m_fOldTime;
    int     m_iTimer;
};

#endif // SETPASSCODEITEM_H
