#include "imagebutton.h"
#include "base.h"
#include "uitheme.h"

#include <QtGui>

ImageButton::ImageButton(QWidget *parent) :
    QLabel(parent)
{
    m_fNormal = 0;
    m_fEnabled = true;
}

void ImageButton::SetImages(QImage xNormalPix)
{
    m_xImage = xNormalPix;
    update();
}

void ImageButton::SetEnabled(bool fEnabled)
{
    m_fEnabled = fEnabled;
    update();
}

void ImageButton::paintEvent(QPaintEvent * e)
{
    QPainter painter;
    painter.begin(this);

    painter.drawImage(0, (height() - m_xImage.height()) / 2, m_xImage);

    if(m_fNormal)
    {
        painter.save();
        painter.setOpacity(UITheme::OpPressed);
        painter.fillRect(rect(), Qt::white);
        painter.restore();
    }

    painter.end();
}

void ImageButton::mousePressEvent(QMouseEvent *ev)
{
    QLabel::mousePressEvent(ev);
    if(m_fEnabled == false)
        return;

    QRect xButtonRect = rect();
    xButtonRect.moveTo(mapToGlobal(QPoint(0, 0)));
    ev->accept();
    m_fNormal = 1;
    update();
}

void ImageButton::mouseMoveEvent(QMouseEvent* ev)
{
    QLabel::mouseMoveEvent(ev);

    if(m_fEnabled == false)
        return;

    if(rect().contains(ev->pos()))
        m_fNormal = 1;
    else
        m_fNormal = 0;

    update();
}

void ImageButton::mouseReleaseEvent(QMouseEvent* ev)
{
    QLabel::mouseReleaseEvent(ev);

    if(m_fEnabled == false)
        return;

    m_fNormal = 0;
    update();

    if(rect().contains(ev->pos()))
        emit clicked();
}


