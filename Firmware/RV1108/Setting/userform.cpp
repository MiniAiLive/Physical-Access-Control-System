#include "userform.h"
#include "appdef.h"
#include "faceengine.h"
#include "stringtable.h"
#include "usereditform.h"
#include "DBManager.h"
#include "alertdlg.h"
#include "menuitem.h"
#include "base.h"
#include "groupform.h"
#include "mount_fs.h"
#include "uitheme.h"
#include "soundbase.h"
#include "waitingform.h"
#include "i2cbase.h"
#include "fptask.h"
#include "themedef.h"
#include "dbif.h"
#include "imagebutton.h"


#include <QtGui>
#include <QGraphicsView>
#include <QLineEdit>
#include <QDialog>
#include <unistd.h>

int UserForm::UserTest = 0;
int UserForm::UserID = 0;

#define GROUP_START_ID (N_MAX_USER_NUM + 200)
#define ADD_START_ID (N_MAX_USER_NUM + 300)

UserForm::UserForm(QGraphicsView *pView, FormBase* pParentForm) :
    SearchItemBaseForm(pView, pParentForm)
{
    m_iUserType = TYPE_USER;
    m_iMaxGroupID = 0;
    m_iMinGroupID = 0x7FFFFFFF;

    SetBGColor(g_UITheme->itemNormalBgColor);

    GetItemView()->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    GetSearchEdit()->setPlaceholderText(StringTable::Str_Search);
    connect(GetSearchEdit(), SIGNAL(textChanged(QString)), this, SLOT(SearchEditChanged(QString)));
}

UserForm::~UserForm()
{
}

void UserForm::OnStart(int iUserType)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    m_iUserType = iUserType;
    RefreshItems();

    FormBase::OnStart();
}

void UserForm::OnResume()
{
    RefreshItems();
    FormBase::OnResume();
}

void UserForm::AddClick()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

//    if(dbfs_get_cur_part() == DB_PART_BACKUP)
//    {
//        AlertDlg::WarningOk(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_System_damaged);
//        return;
//    }

    UserEditForm* pForm = new UserEditForm(m_pParentView, this, m_iUserType);
    connect(pForm, SIGNAL(SigBack()), this, SLOT(OnResume()));
    connect(pForm, SIGNAL(SigSave()), this, SLOT(OnResume()));

    pForm->setAttribute(Qt::WA_DeleteOnClose);
    pForm->OnStart(TYPE_NEW, m_iUserType);
}

void UserForm::RefreshItems()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif
    QTime xOldTime = QTime::currentTime();

    int iPosY = 0;
    SGroupName xOtherGroup = { 0 };
    xOtherGroup.iGroupID = 0;
    strcpy(xOtherGroup.szName, StringTable::Str_Other.toUtf8().data());

    int iGroupCount = dbm_GetGroupCount();
    for(int i = 0; i <= iGroupCount; i ++)
    {        
        SGroupName* pxGroup = dbm_GetGroup(i);
        if(pxGroup == NULL)
            pxGroup = &xOtherGroup;


        QString strGroupName = QString::fromUtf8(pxGroup->szName);
        strGroupName = CalcOmitText(g_UITheme->PrimaryFont, strGroupName, 140);

        MenuItem* pGroupItem = m_vMenuItems[pxGroup->iGroupID + GROUP_START_ID];
        if(pGroupItem == NULL)
            pGroupItem = new MenuItem(pxGroup->iGroupID + GROUP_START_ID);

        bool fExpand = pGroupItem->data(KEY_EXPAND).toBool();

//            if(m_iMaxGroupID < pxGroup->iGroupID)
//                m_iMaxGroupID = pxGroup->iGroupID;

//            if(m_iMinGroupID > pxGroup->iGroupID)
//                m_iMinGroupID = pxGroup->iGroupID;

        pGroupItem->SetBoundingRect(QRect(0, 0, CATEGORY_ITEM_WIDTH, CATEGORY_ITEM_HEIGHT));
        pGroupItem->setPos(QPoint(0, iPosY));
        pGroupItem->setData(KEY_TYPE, TYPE_CATEGORY);
        pGroupItem->setData(KEY_EXPAND, fExpand);
        pGroupItem->setData(KEY_LONG_CLICK, 1);
        pGroupItem->setVisible(true);
        iPosY += CATEGORY_ITEM_HEIGHT;

        int iGroupUserCount = 0;
        int iPersonCount = dbm_GetPersonCount();
        for(int j = 0; j < iPersonCount; j ++)
        {
            PSMetaInfo pxMetaInfo = dbm_GetPersonMetaInfoByIndex(j);
            if(pxMetaInfo->iGroupID == pxGroup->iGroupID)
            {
                MenuItem* pUserItem = m_vMenuItems[pxMetaInfo->nID];
                if(pUserItem == NULL)
                    pUserItem = new MenuItem(pxMetaInfo->nID);

                pUserItem->setPos(QPoint(0, fExpand == false ? -1 * USER_ITEM_HEIGHT : iPosY));
                pUserItem->SetBoundingRect(QRect(0, 0, USER_ITEM_WIDTH, USER_ITEM_HEIGHT));

                QString strTitle;
                strTitle.sprintf("%d. %s", pxMetaInfo->nID + 1, pxMetaInfo->szName);
                QString strOmitText = CalcOmitText(g_UITheme->PrimaryFont, strTitle, 110);
                pUserItem->setData(KEY_PRIMARY_TEXT, strOmitText);

                QSize xIconSize(40 * LCD_RATE, 40 * LCD_RATE);
                pUserItem->setData(KEY_LONG_CLICK, 1);
                pUserItem->setData(KEY_TYPE, TYPE_USER_ITEM);

                if(m_vMenuItems[pxMetaInfo->nID] == NULL)
                {
                    m_pScene->addItem(pUserItem);
                    m_vMenuItems[pxMetaInfo->nID] = pUserItem;
                    connect(pUserItem, SIGNAL(clicked(int)), this, SLOT(UserClick(int)));
                    connect(pUserItem, SIGNAL(SigLongClicked(int)), this, SLOT(UserLongClick(int)));
                }

                if(fExpand)
                {
                    iPosY += USER_ITEM_HEIGHT;
                    pUserItem->setVisible(true);
                }
                else
                    pUserItem->setVisible(false);

                iGroupUserCount ++;
            }
        }


        QString strGroupText;
        strGroupText.sprintf("%s(%d)", strGroupName.toUtf8().data(), iGroupUserCount);
        pGroupItem->setData(KEY_PRIMARY_TEXT, strGroupText);

        if(m_vMenuItems[pxGroup->iGroupID + GROUP_START_ID] == NULL)
        {
            m_pScene->addItem(pGroupItem);
            m_vMenuItems[pxGroup->iGroupID + GROUP_START_ID] = pGroupItem;

            connect(pGroupItem, SIGNAL(clicked(int)), this, SLOT(GroupClick(int)));
            connect(pGroupItem, SIGNAL(SigLongClicked()), this, SLOT(GroupLongClicked()));
        }
    }

    for(int i = N_USER_PERM_BEGIN_ID; i < N_USER_PERM_BEGIN_ID + N_MAX_USER_NUM; i ++)
    {
        PSMetaInfo pxMeaInfo = dbm_GetPersonMetaInfoByID(i);
        if(pxMeaInfo == NULL && m_vMenuItems[i])
        {
            m_vMenuItems[i]->setVisible(false);
//                m_vMenuItems[i]->setPos(QPoint(0, -1 * USER_ITEM_HEIGHT));
        }
    }

    for(int i = 1; i <= N_MAX_GROUP_COUNT; i ++)
    {
        SGroupName* pxGroup = dbm_GetGroupByID(i);
        if(pxGroup == NULL && m_vMenuItems[i + GROUP_START_ID])
        {
            m_vMenuItems[i + GROUP_START_ID]->setPos(QPoint(0, -1 * CATEGORY_ITEM_HEIGHT));
            m_vMenuItems[i + GROUP_START_ID]->setVisible(false);
        }
    }

    MenuItem* pBtnAdd = NULL;
    if(m_vMenuItems[ADD_START_ID] == NULL)
    {
        pBtnAdd = new MenuItem();
        pBtnAdd->setPos(QPoint(ADD_BTN_X, ADD_BTN_Y));
        pBtnAdd->setData(KEY_FIX_POS, QPoint(ADD_BTN_X, ADD_BTN_Y));
        pBtnAdd->SetBoundingRect(QRect(0, 0, ACTION_BUTTON_WIDTH, ACTION_BUTTON_HEIGHT));
        pBtnAdd->setData(KEY_ICON, QImage(":/icons/ic_user_plus.png"));
        pBtnAdd->setData(KEY_ICON_DISABLE, QImage(":/icons/ic_user_plus_disable.png"));
        pBtnAdd->setData(KEY_TYPE, TYPE_ACTION_BUTTON);
        pBtnAdd->setZValue(1);

        m_vMenuItems[ADD_START_ID] = pBtnAdd;
        m_pScene->addItem(pBtnAdd);

        connect(pBtnAdd, SIGNAL(clicked()), this, SLOT(AddClick()));
        connect(GetItemView()->verticalScrollBar(), SIGNAL(valueChanged(int)), pBtnAdd, SLOT(ScrollChnaged(int)));
    }
    else
        pBtnAdd = m_vMenuItems[ADD_START_ID];

    int iPersonCount = dbm_GetPersonCount();
    pBtnAdd->setEnabled(iPersonCount < N_MAX_USER_NUM);

    if(iPosY < 180)
        iPosY = 180;

    GetItemView()->setSceneRect(QRect(0, 0, MAX_X, iPosY));

//    pBtnAdd->ScrollChnaged(GetItemView()->verticalScrollBar()->value());

    m_pScene->update();

    RefreshSearchItems();
}

void UserForm::UserClick(int iID)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

#if (AUTO_TEST == 1)
    if(UserForm::UserTest == 0)
    {
        AddClick();
        return;
    }
#endif

//    if(dbfs_get_cur_part() == DB_PART_BACKUP)
//    {
//        AlertDlg::WarningOk(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_System_damaged);
//        return;
//    }

    UserEditForm* pForm = new UserEditForm(m_pParentView, this, m_iUserType);
    connect(pForm, SIGNAL(SigBack()), this, SLOT(OnResume()));
    connect(pForm, SIGNAL(SigSave()), this, SLOT(OnResume()));

    pForm->OnStart(TYPE_EDIT, m_iUserType, iID);
}

void UserForm::UserLongClick(int iID)
{
    PSMetaInfo pxMetaInfo = dbm_GetPersonMetaInfoByID(iID);
    if(pxMetaInfo == NULL)
        return;

//    if(pxMetaInfo->fPrivilege == EMANAGER && dbm_GetManagerCount() == 1)
//        return;

    if(AlertDlg::WarningYesNo(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_Are_you_sure_to_delete_the_user) == QDialog::Accepted)
    {
        UserID = iID;
        WaitingForm::Waiting(m_pParentView, this, std::bind(&UserForm::DeletePerson, this));

        QTimer::singleShot(0, this, SLOT(RefreshAll()));
    }    
}

void UserForm::RefreshAll()
{
    RefreshItems();
    RetranslateUI();
}

void UserForm::DeletePerson()
{
    dbi_RemoveUserByID(UserID);

#if (FP_MODE == FP_ZHIAN || FP_MODE == FP_CHENGYUAN)
    FPTask::DeleteFP(UserID);
#elif (FP_MODE == FP_GOWEI)
    FPTask::DeleteFP(UserID);
#else
    FPTask::DeleteFP(UserID + 1);
#endif
}

void UserForm::GroupClick(int iGroupID)
{
    PlaySoundLeft();

    MenuItem* pCurGroupItem = m_vMenuItems[iGroupID];
    bool fExpand = pCurGroupItem->data(KEY_EXPAND).toBool();
    fExpand = true - fExpand;
    pCurGroupItem->setData(KEY_EXPAND, fExpand);

    iGroupID -= GROUP_START_ID;

    SGroupName xOtherGroup = { 0 };
    xOtherGroup.iGroupID = 0;

    int iPosY = 0;
    int iGroupCount = dbm_GetGroupCount();
    for(int i = 0; i <= iGroupCount; i ++)
    {
        SGroupName* pxGroup = dbm_GetGroup(i);
        if(pxGroup == NULL)
            pxGroup = &xOtherGroup;

        MenuItem* pGroupItem = m_vMenuItems[pxGroup->iGroupID + GROUP_START_ID];
        bool fExpand = pGroupItem->data(KEY_EXPAND).toBool();
        pGroupItem->setPos(QPoint(0, iPosY));
        iPosY += CATEGORY_ITEM_HEIGHT;

        int iGroupUserCount = 0;
        int iUserCount = dbm_GetPersonCount();
        for(int j = 0; j < iUserCount; j ++)
        {
            PSMetaInfo pxMetaInfo = dbm_GetPersonMetaInfoByIndex(j);
            if(pxMetaInfo->iGroupID == pxGroup->iGroupID)
            {
                MenuItem* pUserItem = m_vMenuItems[pxMetaInfo->nID];
                pUserItem->setPos(QPoint(0, fExpand == false ? -1 * USER_ITEM_HEIGHT : iPosY));

                if(fExpand)
                {
                    pUserItem->setVisible(true);
                    iPosY += USER_ITEM_HEIGHT;
                }
                else
                    pUserItem->setVisible(false);

                iGroupUserCount ++;
            }
        }
    }

    if(iPosY < 180)
        iPosY = 180;

    GetItemView()->setSceneRect(QRect(0, 0, MAX_X, iPosY));
    RetranslateUI();
}

void UserForm::GroupLongClicked()
{
//    if(dbfs_get_cur_part() == DB_PART_BACKUP)
//    {
//        AlertDlg::WarningOk(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_System_damaged);
//        return;
//    }

    GroupForm* pGroupForm = new GroupForm(m_pParentView, this);
    connect(pGroupForm, SIGNAL(SigBack()), this, SLOT(OnResume()));

    pGroupForm->onStart();
}

void UserForm::RetranslateUI()
{
    if(m_iUserType == TYPE_MANAGER)
        SetTitle(StringTable::Str_Manager);
    else
        SetTitle(StringTable::Str_User);

//    GetSearchEdit()->setPlaceholderText(StringTable::Str_Search);
}

void UserForm::SearchEditChanged(QString)
{
    QString strText = GetSearchEdit()->text();
    if(m_strSearchText != strText)
    {
        m_strSearchText = strText;

        if(m_strSearchText.isEmpty())
            RefreshItems();
        else
            RefreshSearchItems();
    }
}

void UserForm::RefreshSearchItems()
{
    if(m_strSearchText.isEmpty())
        return;

    QList<QGraphicsItem*> vItems = m_pScene->items();

    int iPosY = 0;
    QGraphicsItem* pItem = NULL;
    foreach(pItem, vItems)
    {
        MenuItem* pUserItem = qgraphicsitem_cast<MenuItem*>(pItem);
        if(pUserItem == NULL)
        {
            qDebug() << "pUserItem is null";
            continue;
        }

        if(pUserItem->GetID() >= GROUP_START_ID)
        {
            pUserItem->setVisible(false);
            continue;
        }

        if(pUserItem->data(KEY_TYPE) != TYPE_USER_ITEM)
        {
            pUserItem->setVisible(false);
            continue;
        }

        int iID = pUserItem->GetID();
        PSMetaInfo pxMetaInfo = dbm_GetPersonMetaInfoByID(iID);
        if(pxMetaInfo == NULL)
        {
            pUserItem->setVisible(false);
            continue;
        }

        if(!QString::fromUtf8(pxMetaInfo->szName).contains(m_strSearchText))
        {
            pUserItem->setVisible(false);
            continue;
        }

        pUserItem->setPos(QPoint(0, iPosY));
        pUserItem->setVisible(true);

        iPosY += USER_ITEM_HEIGHT;
    }

    if(iPosY < 180)
        iPosY = 180;

    GetItemView()->setSceneRect(QRect(0, 0, MAX_X, iPosY));
}
