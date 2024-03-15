#ifndef TEXTFORM_H
#define TEXTFORM_H

#include "itemformbase.h"

#define MAX_ERROR_TYPE 6

class TextForm : public ItemFormBase
{
    Q_OBJECT
public:
    explicit TextForm(QGraphicsView *pView, FormBase* pParentForm);

    enum
    {
        PRESENTATION_START,
        PRESENTATION_STOP,
    };

signals:
    
public slots:
    void    OnStart(int iMode);
    void    ShowBattClick();
    void    MotorClick();
    void    PresentTestStart();

protected:

protected:
    bool    event(QEvent* e);
    void    RetranslateUI();

    int     m_iMode;

    int     m_aiErrrorArray[5][MAX_ERROR_TYPE];
};

#endif // TEXTFORM_H
