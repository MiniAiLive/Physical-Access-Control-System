#include "custombutton.h"
#include "base.h"
#include "uitheme.h"

#include <QtGui>

CustomButton::CustomButton(QWidget *parent)
    : QLabel(parent)
{
    m_fNormal = 0;
    m_iMousePressed = 0;
}


void CustomButton::paintEvent(QPaintEvent * e)
{
    QPainter xPainter;
    xPainter.begin(this);

    if(m_fNormal == 1)
    {
        xPainter.setOpacity(UITheme::OpPressed);
        xPainter.fillRect(rect(), g_UITheme->customBtnNormalColor);
    }

    xPainter.end();

    QLabel::paintEvent(e);
}


void CustomButton::mousePressEvent(QMouseEvent *ev)
{
    QLabel::mousePressEvent(ev);

    ev->accept();
    m_fNormal = 1;
    m_iMousePressed = 1;
    update();
}

void CustomButton::mouseMoveEvent(QMouseEvent* ev)
{
    QLabel::mouseMoveEvent(ev);
    if(m_iMousePressed == 0)
        return;

    if(rect().contains(ev->pos()))
        m_fNormal = 1;
    else
        m_fNormal = 0;

    update();
}

void CustomButton::mouseReleaseEvent(QMouseEvent* ev)
{
    QLabel::mouseReleaseEvent(ev);

    m_fNormal = 0;
    m_iMousePressed = 0;
    update();

    if(rect().contains(ev->pos()))
    {
        emit clicked();
    }
}
