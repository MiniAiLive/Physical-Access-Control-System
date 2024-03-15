#ifndef LOGOSETTINGFORM_H
#define LOGOSETTINGFORM_H

#include "itemformbase.h"

namespace Ui {
class LogoSettingForm;
}

class LogoSettingForm : public FormBase
{
    Q_OBJECT
public:
    explicit LogoSettingForm(QGraphicsView *pView, FormBase* pParentForm);
    ~LogoSettingForm();

signals:

public slots:
    void    OnStart();
    void    OnStop();

protected:
    void    paintEvent(QPaintEvent *e);
    void    mousePressEvent(QMouseEvent* e);

    void    UpdateImage();

protected:
    Ui::LogoSettingForm *ui;

    QImage  m_xLogoImage;
};

#endif // LOGOSETTINGFORM_H
