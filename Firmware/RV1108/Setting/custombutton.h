#ifndef CUSTOMBUTTON_H
#define CUSTOMBUTTON_H

#include <QLabel>
#include <QPixmap>

class CustomButton : public QLabel
{
    Q_OBJECT
public:
    CustomButton(QWidget *parent = 0);

signals:
    void    clicked();

public slots:

protected:
    void        paintEvent(QPaintEvent *);
    void        mousePressEvent(QMouseEvent *ev);
    void        mouseMoveEvent(QMouseEvent* ev);
    void        mouseReleaseEvent(QMouseEvent* eb);

private:
    int         m_fNormal;
    int         m_iMousePressed;
};

#endif // CUSTOMBUTTON_H
