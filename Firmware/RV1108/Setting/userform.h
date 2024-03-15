#ifndef USERITEM_H
#define USERITEM_H

#include "searchitembaseform.h"

class UserForm : public SearchItemBaseForm
{
    Q_OBJECT

public:
    explicit UserForm(QGraphicsView *pView, FormBase* pParentForm);    
    ~UserForm();

    static int UserTest;
    static int UserID;

    void DeletePerson();
signals:

public slots:
    void    OnStart(int userType);
    void    OnResume();

    void    AddClick();
    void    UserClick(int nID);
    void    UserLongClick(int nID);
	void    GroupClick(int iGroupID);
    void    GroupLongClicked();

    void    SearchEditChanged(QString);
    void    RefreshAll();
protected:
    void    RetranslateUI();
    void    RefreshItems();
    void    RefreshSearchItems();
private:

private:

    int     m_iUserType;

    int     m_iMaxGroupID;
    int     m_iMinGroupID;

    QString     m_strSearchText;
};

#endif // USERITEM_H
