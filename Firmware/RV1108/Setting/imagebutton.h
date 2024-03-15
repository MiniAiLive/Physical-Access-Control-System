#ifndef IMAGEBUTTON_H
#define IMAGEBUTTON_H

#include <QLabel>
#include <QPixmap>

class ImageButton : public QLabel
{
    Q_OBJECT
public:
    explicit ImageButton(QWidget *parent = 0);

    void    SetImages(QImage xNormalPix);
    void    SetEnabled(bool fEnabled);

signals:
    void    clicked();

public slots:

protected:
    void        paintEvent(QPaintEvent *);
    void        mousePressEvent(QMouseEvent *ev);
    void        mouseMoveEvent(QMouseEvent* ev);
    void        mouseReleaseEvent(QMouseEvent* eb);

private:
    QImage      m_xImage;
    int         m_fNormal;
    bool        m_fEnabled;
};

#endif // IMAGEBUTTON_H
