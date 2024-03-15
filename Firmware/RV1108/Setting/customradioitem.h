#ifndef CUSTOMRADIOITEM_H
#define CUSTOMRADIOITEM_H

#include <QObject>
#include <QGraphicsItem>

class CustomRadioItem : public QObject, public QGraphicsItem
{
    Q_OBJECT
    Q_INTERFACES(QGraphicsItem)
public:
    CustomRadioItem(int iID = -1);
    QRectF              boundingRect() const;
    void                paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget = 0);

    void                SetBoundingRect(QRect xRect);
    void                MoveItem(int x, int y);
    void                SetText(QString sText);
    void                SetChecked(bool fChecked);
    int                 GetID(){return m_ID;}

signals:
    void                clicked(int iID);

protected:
    void                mousePressEvent(QGraphicsSceneMouseEvent* e);
    void                mouseMoveEvent(QGraphicsSceneMouseEvent* e);
    void                mouseReleaseEvent(QGraphicsSceneMouseEvent* e);

    QRect               m_xBoundingRect;
    QPixmap             m_xCheckOnPix;

    int                 m_ID;
    QString             m_sText;

    int                 m_fPressed;
    int                 m_fMoveStart;
    int                 m_iOldScrollValue;
};

#endif // CUSTOMRADIOITEM_H
