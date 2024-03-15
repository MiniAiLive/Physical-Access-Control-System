#include "graphicsview.h"
#include "base.h"

#include <QtGui>
#include <QScrollBar>

GraphicsPixmapItem::GraphicsPixmapItem(QObject* )
{

}

QRectF GraphicsPixmapItem::boundingRect() const
{
    if(m_xBoundingRect.isNull())
        return QRectF(QPointF(0,0), QPointF(0,0));

    return m_xBoundingRect;
}

void GraphicsPixmapItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget * widget)
{
    painter->save();

    int iHorOffset = 0, iVerOffset = 0;
    if(scene()->views().size())
    {
        QRect xViewportRect = scene()->views().at(0)->viewport()->rect();
        iHorOffset = pos().x();
        iVerOffset = pos().y();
        xViewportRect.moveTo(-iHorOffset, -iVerOffset);

        painter->setClipRect(xViewportRect);
    }

    painter->drawPixmap(m_xBoundingRect, m_xNormalPix);
    painter->restore();
}

void GraphicsPixmapItem::SetPixmap(QPixmap xPixmap)
{
    m_xNormalPix = xPixmap;
    m_xBoundingRect = m_xNormalPix.rect();
}

void GraphicsPixmapItem::SetBoundingRect(QRect rect)
{
    m_xBoundingRect = rect;
}

void GraphicsPixmapItem::MoveItem(QPoint off)
{
    m_xBoundingRect.moveTo(off);
}

QPoint GraphicsPixmapItem::GetItemPos()
{
    return m_xBoundingRect.topLeft();
}


GraphicsView::GraphicsView(QWidget *parent) :
    QGraphicsView(parent)
{
    m_fShowScroll = true;
    m_xScrollColor = Qt::green;
}

GraphicsView::~GraphicsView()
{
    if(scene())
    {
        QList<QGraphicsItem*> vItems = scene()->items();
        for(int i = 0; i < vItems.size(); i ++)
            scene()->removeItem(vItems[i]);

        scene()->clear();
    }
}

void GraphicsView::SetImages(QPixmap xBgPix, QPixmap xFgPix)
{
    m_xBgPixmap = xBgPix;
    m_xFgPixmap = xFgPix;
}

void GraphicsView::SetViewColor(QColor bgColor)
{
    m_bgColor = bgColor;
}

void GraphicsView::drawBackground(QPainter* painter, const QRectF& rect)
{
    QGraphicsView::drawBackground(painter, rect);

    if(!m_xBgPixmap.isNull())
    {
        QRect mapRect = mapFromScene(rect).boundingRect();
        painter->drawPixmap(rect, m_xBgPixmap, QRect(geometry().left() + mapRect.left(), geometry().top() + mapRect.top(), mapRect.width(), mapRect.height()));
        if(!m_xFgPixmap.isNull())
            painter->drawPixmap(rect, m_xFgPixmap, QRect(2, 2, 2, 2));
    }
}

void GraphicsView::drawForeground(QPainter* painter, const QRectF& rect)
{
    QGraphicsView::drawForeground(painter, rect);

    if(verticalScrollBar()->minimum() != verticalScrollBar()->maximum() && m_fShowScroll)
    {
        painter->save();
        painter->setOpacity(0.8);
        QRect scrollBarRect = QRect(width() - 4, verticalScrollBar()->value() * height() / (verticalScrollBar()->pageStep() + verticalScrollBar()->maximum()), 3, height() * verticalScrollBar()->pageStep() / (verticalScrollBar()->pageStep() + verticalScrollBar()->maximum()));
        if(scrollBarRect.height() == 0)
            scrollBarRect.setHeight(10);

        painter->fillRect(mapToScene(scrollBarRect).boundingRect(), m_xScrollColor);
        painter->restore();
    }
}

void GraphicsView::SetScrollBar(bool fShow)
{
    m_fShowScroll = fShow;
}

void GraphicsView::SetScrollBarColor(QColor xColor)
{
    m_xScrollColor = xColor;
}

