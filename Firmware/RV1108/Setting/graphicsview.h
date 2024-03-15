#ifndef GRAPHICSVIEW_H
#define GRAPHICSVIEW_H

#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>

class GraphicsPixmapItem : public QObject, public QGraphicsItem
{
    Q_OBJECT
    Q_INTERFACES(QGraphicsItem)
public:
    explicit GraphicsPixmapItem(QObject* parent = 0);

    QRectF              boundingRect() const;
    void                paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

    void                SetPixmap(QPixmap xPixmap);
    void                SetBoundingRect(QRect xRect);

    void                MoveItem(QPoint xOff);
    QPoint              GetItemPos();
private:
    QPixmap             m_xNormalPix;
    QRect               m_xBoundingRect;
};

class GraphicsView : public QGraphicsView
{
    Q_OBJECT
public:
    explicit GraphicsView(QWidget *parent = 0);
    ~GraphicsView();

    void    SetImages(QPixmap xBgPix, QPixmap xFgPix);
    void    SetViewColor(QColor bgColor);
    void    SetScrollBar(bool fShow);
    void    SetScrollBarColor(QColor xColor);
signals:

public slots:

protected:
    void    drawBackground(QPainter* painter, const QRectF& rect);
    void    drawForeground(QPainter* painter, const QRectF& rect);

private:
    QPixmap     m_xBgPixmap;
    QPixmap     m_xFgPixmap;
    bool        m_fShowScroll;
    QColor      m_xScrollColor;
    QColor      m_bgColor;
};

#endif // GRAPHICSVIEW_H
