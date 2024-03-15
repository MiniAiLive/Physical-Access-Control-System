#ifndef USEREDITFORM_H
#define USEREDITFORM_H

#include <QWidget>
#include "itemformbase.h"
#include "faceengine.h"
#include "shared.h"


class QLabel;
class UserEditItem;
class CustomLineEdit;
class QGraphicsProxyWidget;
class MenuItem;
class QMenu;
class UserEditForm : public ItemFormBase
{
    Q_OBJECT

public:
    explicit UserEditForm(QGraphicsView *pView, FormBase* pParentForm, int iUserType);
    ~UserEditForm();

    void    OnStart(int iEditType, int iUserType, int iEditID = -1);
    void    OnResume();
    int     GetUserID();

    void    DeletePerson();
    void    SavePerson();

signals:
    void    SigSave();

public slots:
    void    BackClick();

    void    FaceClick();
    void    FaceNew();
    void    FaceUpdate();
    void    FaceDelete();
    void    EnrollFaceFinished(int iEnrollResult);

    void    NameClick();

    void    CardClick();
    void    CardNew();
    void    CardDelete();
    void    EnrollCardFinished(int iEnrollResult);        

    void    PasscodeClick();
    void    ReceivePasscode();
    void    PasscodeNew();
    void    PasscodeDelete();

    void    PrivilegeClick();
    void    GroupClick();
    void    GroupNew();
    void    GroupUpdate();
    void    GroupDelete();

    void    SaveClick();
    void    DeleteClick();
protected:
    void    RetranslateUI();
    void    InitItems();

private:
    QWidget*          m_pWidget;

    int     m_iUserType;
    int     m_iEditType;
    int     m_iEditID;

    int     m_iFaceUpdate;
    SMetaInfo   m_xOldMetaInfo;
    SFeatInfo   m_xOldFeatInfo;

    SMetaInfo   m_xMetaInfo;
    SFeatInfo   m_xFeatInfo;

    QMenu*  m_pMenuFace;
    QMenu*  m_pMenuCard;
    QMenu*  m_pMenuPasscode;
    QMenu*  m_pMenuGroup;
    QMenu*  m_pMenu;
};

#endif // USEREDITFORM_H
