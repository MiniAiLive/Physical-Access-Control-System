#ifndef MAINACTIVATIONFORM_H
#define MAINACTIVATIONFORM_H

#include "formbase.h"

class MainActivationForm : public FormBase
{
    Q_OBJECT
public:
    explicit MainActivationForm(QGraphicsView *pView, FormBase* pParentForm);
    ~MainActivationForm();

signals:
    void    SigActivationOk();

public slots:
    void    QRCodeReadFinished(QString strQRCode);
    void    QRCodeReadFinished(int iCamError);
    void    QRCodeReadClick();

    void    OnStart(int fDelPrevScene = 0);
    void    OnStop();
    void    OnPause();

protected:
    void    paintEvent(QPaintEvent *e);
    bool    event(QEvent* e);
    void    RetranslateUI();

    QImage  m_qrImage;
};

#endif // MAINACTIVATIONFORM_H
