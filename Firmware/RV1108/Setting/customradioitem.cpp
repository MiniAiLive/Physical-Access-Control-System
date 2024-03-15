#include "customradioitem.h"
#include "base.h"
#include "uitheme.h"
#include "themedef.h"

#include <QtGui>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsSceneMouseEvent>
#include <QScrollBar>

CustomRadioItem::CustomRadioItem(int iID)
{
    m_ID = iID;
    m_fPressed = 0;

    setFlag(QGraphicsItem::ItemIsSelectable, true);
}

QRectF CustomRadioItem::boundingRect() const
{
    if(m_xBoundingRect.isNull())
        return QRectF(QPointF(0,0), QPointF(0,0));

    return m_xBoundingRect;
}

void CustomRadioItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED(option);
    Q_UNUSED(widget);

    if(isSelected())
        painter->drawImage(m_xBoundingRect.right() - 35 * LCD_RATE, m_xBoundingRect.top() + 10 * LCD_RATE, QImage(":/icons/ic_radio_button_checked.png"));
    else
        painter->drawImage(m_xBoundingRect.right() - 35 * LCD_RATE, m_xBoundingRect.top() + 10 * LCD_RATE, QImage(":/icons/ic_radio_button_unchecked.png"));

    painter->save();
    painter->setPen(g_UITheme->itemMainTextColor);
    painter->setFont(g_UITheme->SecondaryFont);

    QRect xTextRect = QRect(m_xBoundingRect.left() + 8 * LCD_RATE, m_xBoundingRect.top(), 150/*m_xBoundingRect.width()*/, m_xBoundingRect.height());
    painter->drawText(xTextRect, GetOmitText(painter->font(), m_sText, xTextRect.width()), Qt::AlignLeft | Qt::AlignVCenter);
    painter->restore();

    if(m_fPressed)
    {
        painter->setOpacity(UITheme::OpPressed);
        painter->fillRect(m_xBoundingRect, Qt::white);
    }
}


void CustomRadioItem::SetBoundingRect(QRect xRect)
{
    m_xBoundingRect = xRect;
}

void CustomRadioItem::MoveItem(int x, int y)
{
    m_xBoundingRect.moveTo(x, y);

    update();
}

void CustomRadioItem::SetText(QString sText)
{
    m_sText = sText;
}

void CustomRadioItem::SetChecked(bool fChecked)
{
    setSelected(fChecked);
}


void CustomRadioItem::mousePressEvent(QGraphicsSceneMouseEvent* e)
{
    m_fPressed = 1;
    m_fMoveStart = 0;

    if(scene() && scene()->views().count() > 0)
        m_iOldScrollValue = scene()->views().at(0)->verticalScrollBar()->value();

    update();
}

void CustomRadioItem::mouseMoveEvent(QGraphicsSceneMouseEvent* e)
{
    if(scene() && scene()->views().count() > 0)
    {
        int curScrollValue = scene()->views().at(0)->verticalScrollBar()->value();
        if(abs(m_iOldScrollValue - curScrollValue) > 20 * LCD_RATE)
        {
            m_fMoveStart = 1;
        }
    }
    update();
}

void CustomRadioItem::mouseReleaseEvent(QGraphicsSceneMouseEvent* e)
{
    QGraphicsItem::mouseReleaseEvent(e);

    m_fPressed = 0;

    update();

    if(m_xBoundingRect.contains(e->scenePos().toPoint()) && m_fMoveStart == 0)
    {
        emit clicked(m_ID);
    }
}
