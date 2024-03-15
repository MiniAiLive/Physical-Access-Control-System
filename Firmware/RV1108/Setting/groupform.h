#ifndef GROUPFORM_H
#define GROUPFORM_H

#include "itemformbase.h"

class QMenu;
class GroupForm : public ItemFormBase
{
    Q_OBJECT
public:
    explicit GroupForm(QGraphicsView *view, FormBase* parentForm0);
    ~GroupForm();

signals:

public slots:
    void    onStart();
    void    GroupAdd();
    void    GroupClick(int iIdx);
    void    GroupUpdate();
    void    GroupDelete();
    void    RefreshItems();

private:

protected:

private:
    QMenu*  m_pMenu;

    int     m_iSelectIdx;
};

#endif // GROUPFORM_H
