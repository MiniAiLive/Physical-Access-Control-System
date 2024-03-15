#ifndef LOGITEM_H
#define LOGITEM_H

#include <QWidget>
#include <QPixmap>
#include <QTime>
#include "itemformbase.h"

class QMenu;
class QGraphicsScene;
class GraphicsScene;
class QPropertyAnimation;
class QGraphicsPixmapItem;
class LogForm : public ItemFormBase
{
    Q_OBJECT
public:
    explicit LogForm(QGraphicsView *pView, FormBase* pParentForm);
    ~LogForm();

signals:

public slots:
    void    OnStart();
    void    BackClick();
    void    PressedLog(int fState);

protected:
    void    RetranslateUI();

private:
    void    InitLogView();

private:
    QVector<MenuItem*>  m_vLogItems;
    MenuItem*           m_pFaceImageItem;
};

#endif // LOGITEM_H

