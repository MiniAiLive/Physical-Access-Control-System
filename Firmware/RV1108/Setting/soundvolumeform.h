#ifndef SOUNDVOLUMEFORM_H
#define SOUNDVOLUMEFORM_H

#include <QWidget>

namespace Ui {
class SoundVolumeForm;
}

class SoundVolumeForm : public QWidget
{
    Q_OBJECT

public:
    explicit SoundVolumeForm(QWidget *parent = 0);
    ~SoundVolumeForm();

    void    SetPos(QPoint xPos, bool bPressed = false);
    void    SetPos(int iVal, bool bPressed = false);
    int     GetPos();

protected:
    void    mousePressEvent(QMouseEvent* e);
    void    mouseMoveEvent(QMouseEvent* e);
    void    mouseReleaseEvent(QMouseEvent* e);

private:
    Ui::SoundVolumeForm *ui;
};

#endif // SOUNDVOLUMEFORM_H
