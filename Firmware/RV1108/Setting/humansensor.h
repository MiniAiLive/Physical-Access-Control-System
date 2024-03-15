#ifndef HUMANSENSOR_H
#define HUMANSENSOR_H

#include "itemformbase.h"
#include "settings.h"

class QMenu;
class MenuItem;
class HumanSensorForm : public ItemFormBase
{
    Q_OBJECT
public:
    explicit HumanSensorForm(QGraphicsView *view, FormBase* parentForm);
    ~HumanSensorForm();

signals:

public slots:
    void    OnStart();
    void    HumanSensorClick();

protected:
    void    RetranslateUI();
    void    RefreshItems();

private:
};

#endif // SETTINGFORM_H
