#include "usereditform.h"
#include "stringtable.h"
#include "appdef.h"
#include "base.h"
#include "faceengine.h"
#include "uitheme.h"
#include "alertdlg.h"
#include "DBManager.h"
#include "sliderspin.h"
#include "passcodeform.h"
#include "enrollfaceform.h"
#include "waitingform.h"
#include "mount_fs.h"
#include "menuitem.h"
#include "enrollcardform.h"
#include "datesettingform.h"
#include "timesettingform.h"
#include "userform.h"
#include "groupform.h"
#include "i2cbase.h"
#include "enrollfpform.h"
#include "fptask.h"
#include "enrollhandform.h"
#include "testvolmainform.h"
#include "soundbase.h"
#include "hetproc.h"
#include "dbif.h"

#include <QtGui>
#include <QGraphicsScene>
#include <QMenu>
#include <QGraphicsView>
#include <QDialog>

#define DEFAULT_TIMELEFT 8

enum
{
    ID_FACE_ITEM,
    ID_CARD_ITEM,
    ID_PASSCODE_ITEM,
    ID_NAME_ITEM,
    ID_PRIVILEGE_ITEM,
    ID_GROUP_ITEM,
    ID_DELETE_ITEM,
};

#define FP_ACT_NONE 0
#define FP_ACT_ENROLL 1
#define FP_ACT_DELETE 2

UserEditForm::UserEditForm(QGraphicsView *pView, FormBase* pParentForm, int iUserType) :
    ItemFormBase(pView, pParentForm)
{
    SetBGColor(g_UITheme->itemNormalBgColor);

    m_iUserType = iUserType;

    m_iFaceUpdate = 0;
    m_pMenu = NULL;
    m_pMenuFace = NULL;
    m_pMenuCard = NULL;
    m_pMenuPasscode = NULL;
    m_pMenuGroup = NULL;
}

UserEditForm::~UserEditForm()
{
    if(m_pMenuFace)
        delete m_pMenuFace;

    if(m_pMenuCard)
        delete m_pMenuCard;

    if(m_pMenuPasscode)
        delete m_pMenuPasscode;

    if(m_pMenuGroup)
        delete m_pMenuGroup;

    if(m_pMenu)
        delete m_pMenu;
}


void UserEditForm::InitItems()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif
    m_pScene->clear();
    m_vMenuItems.clear();

    int iPosY = 0;
    MenuItem* pFaceItem = new MenuItem();
    pFaceItem->setPos(QPoint(0, 0));
    pFaceItem->SetBoundingRect(QRect(0, 0, FACE_ITEM_WIDTH, FACE_ITEM_HEIGHT));
    pFaceItem->setData(KEY_TYPE, TYPE_FACE_ITEM);
    m_pScene->addItem(pFaceItem);
    m_vMenuItems[ID_FACE_ITEM] = pFaceItem;
    connect(pFaceItem, SIGNAL(clicked()), this, SLOT(FaceClick()));

    MenuItem* pCardItem = new MenuItem();
    pCardItem->setPos(QPoint(8, 120));
    pCardItem->SetBoundingRect(QRect(0, 0, 52, 40));
    pCardItem->setData(KEY_TYPE, TYPE_CARD_ITEM);
    m_pScene->addItem(pCardItem);
    m_vMenuItems[ID_CARD_ITEM] = pCardItem;
    connect(pCardItem, SIGNAL(clicked()), this, SLOT(CardClick()));

    MenuItem* pPasscodeItem = new MenuItem();
    pPasscodeItem->setPos(QPoint(60, 120));
    pPasscodeItem->SetBoundingRect(QRect(0, 0, 52, 40));
    pPasscodeItem->setData(KEY_TYPE, TYPE_CARD_ITEM);
    m_pScene->addItem(pPasscodeItem);
    m_vMenuItems[ID_PASSCODE_ITEM] = pPasscodeItem;
    connect(pPasscodeItem, SIGNAL(clicked()), this, SLOT(PasscodeClick()));

    MenuItem* pNameItem = new MenuItem();
    pNameItem->setPos(QPoint(120, iPosY));
    pNameItem->setData(KEY_PRIMARY_TEXT, StringTable::Str_Name);
    pNameItem->SetBoundingRect(QRect(0, 0, ENROLL_ITEM_WIDTH, ENROLL_ITEM_HEIGHT));
    pNameItem->setData(KEY_TYPE, TYPE_ENROLL_ITEM);
    iPosY += ENROLL_ITEM_HEIGHT;
    m_pScene->addItem(pNameItem);
    m_vMenuItems[ID_NAME_ITEM] = pNameItem;
    connect(pNameItem, SIGNAL(clicked()), this, SLOT(NameClick()));

    MenuItem* pPrivilegeItem = new MenuItem();
    pPrivilegeItem->setPos(QPoint(120, iPosY));
    pPrivilegeItem->setData(KEY_PRIMARY_TEXT, StringTable::Str_Privilege);
    pPrivilegeItem->SetBoundingRect(QRect(0, 0, ENROLL_ITEM_WIDTH, ENROLL_ITEM_HEIGHT));
    pPrivilegeItem->setData(KEY_TYPE, TYPE_ENROLL_ITEM);
    iPosY += ENROLL_ITEM_HEIGHT;
    m_pScene->addItem(pPrivilegeItem);
    m_vMenuItems[ID_PRIVILEGE_ITEM] = pPrivilegeItem;
    connect(pPrivilegeItem, SIGNAL(clicked()), this, SLOT(PrivilegeClick()));

    MenuItem* pGroupItem = new MenuItem();
    pGroupItem->setPos(QPoint(120, iPosY));
    pGroupItem->setData(KEY_PRIMARY_TEXT, StringTable::Str_Group);
    pGroupItem->SetBoundingRect(QRect(0, 0, ENROLL_ITEM_WIDTH, ENROLL_ITEM_HEIGHT));
    pGroupItem->setData(KEY_TYPE, TYPE_ENROLL_ITEM);
    iPosY += ENROLL_ITEM_HEIGHT;
    m_pScene->addItem(pGroupItem);
    m_vMenuItems[ID_GROUP_ITEM] = pGroupItem;
    connect(pGroupItem, SIGNAL(clicked()), this, SLOT(GroupClick()));

    iPosY += 10;
    {
        MenuItem* pDeleteItem = new MenuItem();
        pDeleteItem->setPos(QPoint(120 + 40 * LCD_RATE, iPosY));
        pDeleteItem->SetBoundingRect(QRect(0, 0, RAISED_BUTTON_WIDTH, RAISED_BUTTON_HEIGHT));
        pDeleteItem->setData(KEY_TYPE, TYPE_RAISED_BUTTON);
        pDeleteItem->setData(KEY_PRIMARY_TEXT, StringTable::Str_Delete);
        pDeleteItem->setData(KEY_BG_COLOR, g_UITheme->raisedItemColor[1]);
        m_pScene->addItem(pDeleteItem);
        m_vMenuItems[ID_DELETE_ITEM] = pDeleteItem;
        connect(pDeleteItem, SIGNAL(clicked()), this, SLOT(DeleteClick()));

        if(m_iEditType == TYPE_NEW)
            m_vMenuItems[ID_DELETE_ITEM]->setEnabled(false);
        else
            m_vMenuItems[ID_DELETE_ITEM]->setEnabled(true);
    }

    QString strMenuSteets = "QMenu {background-color: white; margin: 0px;}"
                            "QMenu::item {padding: 8px 30px 8px 20px;border: 1px solid transparent;}"
                            "QMenu::item:selected {border-color: white ;background: rgba(100, 100, 100, 150);}"
                            "QMenu::separator {height: 1px;background: lightblue;margin-left: 10px;margin-right: 5px;}"
                            "QMenu::indicator:non-exclusive:checked {image: url(:/icons/ic_check.png);}";    

    m_pMenuFace = new QMenu(this);
    m_pMenuFace->addAction(StringTable::Str_New, this, SLOT(FaceNew()));
    m_pMenuFace->addAction(StringTable::Str_Update, this, SLOT(FaceUpdate()));
	m_pMenuFace->addAction(StringTable::Str_Delete, this, SLOT(FaceDelete()));
    m_pMenuFace->setStyleSheet(strMenuSteets);

    m_pMenuFace->setFont(g_UITheme->PrimaryFont);

    m_pMenuCard = new QMenu(this);
    m_pMenuCard->addAction(StringTable::Str_Update, this, SLOT(CardNew()));
    m_pMenuCard->addAction(StringTable::Str_Delete, this, SLOT(CardDelete()));
    m_pMenuCard->setStyleSheet(strMenuSteets);

    m_pMenuCard->setFont(g_UITheme->PrimaryFont);

    m_pMenuPasscode = new QMenu(this);
    m_pMenuPasscode->addAction(StringTable::Str_Update, this, SLOT(PasscodeNew()));
    m_pMenuPasscode->addAction(StringTable::Str_Delete, this, SLOT(PasscodeDelete()));
    m_pMenuPasscode->setStyleSheet(strMenuSteets);

    m_pMenuPasscode->setFont(g_UITheme->PrimaryFont);

    m_pMenuGroup = new QMenu(this);
    m_pMenuGroup->addAction(StringTable::Str_Update, this, SLOT(GroupUpdate()));
    m_pMenuGroup->addAction(StringTable::Str_Delete, this, SLOT(GroupDelete()));
    m_pMenuGroup->setStyleSheet(strMenuSteets);

    m_pMenuGroup->setFont(g_UITheme->PrimaryFont);

    m_pMenu = new QMenu(this);
    m_pMenu->setStyleSheet("QMenu {background-color: white; margin: 0px;}"
                            "QMenu::item {padding: 8px 30px 8px 20px;border: 1px solid transparent;}"
                            "QMenu::item:selected {border-color: white ;background: rgba(100, 100, 100, 150);}"
                            "QMenu::separator {height: 1px;background: lightblue;margin-left: 10px;margin-right: 5px;}"
                            "QMenu::indicator:non-exclusive:checked {image: url(:/icons/ic_check.png);}");

    m_pMenu->setFont(g_UITheme->PrimaryFont);
}


void UserEditForm::OnStart(int iEditType, int iUserType, int iEditID)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    m_iEditType = iEditType;
    m_iEditID = iEditID;
    m_iUserType = iUserType;

    memset(&m_xMetaInfo, 0, sizeof(SMetaInfo));
    memset(&m_xFeatInfo, 0, sizeof(SFeatInfo));

    memset(&m_xOldMetaInfo, 0, sizeof(SMetaInfo));
    memset(&m_xOldFeatInfo, 0, sizeof(SFeatInfo));

    if(m_iEditType == TYPE_NEW)
    {
        m_xOldMetaInfo.nID = dbm_GetNewUserID();
        m_xOldMetaInfo.fPrivilege = iUserType;

        GetUserName(m_xOldMetaInfo.szName, m_xOldMetaInfo.nID, 0);
    }
    else
    {
        PSMetaInfo pxMetaInfo = dbm_GetPersonMetaInfoByID(iEditID);
        if(pxMetaInfo == NULL)
        {
#if (AUTO_TEST == 1)
            qDebug() << "[Call]" << this << __FUNCTION__ << "Edit Null!!!" << iEditID;
#endif
            return;
        }

        m_xOldMetaInfo = *dbm_GetPersonMetaInfoByID(iEditID);
        m_xOldFeatInfo = *dbm_GetPersonFeatInfoByID(iEditID);

        {
            int iExist = -1;
            int iGroupCount = dbm_GetGroupCount();

            for(int i = 0; i < iGroupCount; i ++)
            {
                SGroupName* pxGroup = dbm_GetGroup(i);
                if(pxGroup == NULL)
                    continue;

                if(pxGroup->iGroupID == m_xOldMetaInfo.iGroupID)
                {
                    iExist = i;
                    break;
                }
            }

            if(iExist == -1 && m_xOldMetaInfo.iGroupID != 0)
                m_xOldMetaInfo.iGroupID = 0;
        }
    }

    m_xMetaInfo = m_xOldMetaInfo;
    m_xFeatInfo = m_xOldFeatInfo;

    InitItems();

    FormBase::OnStart();
}

void UserEditForm::OnResume()
{
    FormBase::OnResume();
}

int UserEditForm::GetUserID()
{
    return m_xMetaInfo.nID;
}

void UserEditForm::BackClick()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

#if (AUTO_TEST == 1)
    if(UserForm::UserTest == 0)
        return;
#endif

    int fEnableSave = 0;
    if(memcmp(&m_xMetaInfo, &m_xOldMetaInfo, sizeof(SMetaInfo)))
        fEnableSave = 1;
    if(memcmp(&m_xFeatInfo, &m_xOldFeatInfo, sizeof(SFeatInfo)))
        fEnableSave = 1;

    int iExist = -1;
    int iGroupCount = dbm_GetGroupCount();

    for(int i = 0; i < iGroupCount; i ++)
    {
        SGroupName* pxGroup = dbm_GetGroup(i);
        if(pxGroup == NULL)
            continue;

        if(pxGroup->iGroupID == m_xMetaInfo.iGroupID)
        {
            iExist = i;
            break;
        }
    }

    if(iExist == -1 && m_xMetaInfo.iGroupID != 0)
    {
        m_xMetaInfo.iGroupID = 0;
        fEnableSave = 1;
    }

    if(m_iEditType == TYPE_NEW)
    {
        if(m_xFeatInfo.nDNNFeatCount == 0 && !strlen(m_xMetaInfo.szPasscode) && m_xMetaInfo.nCardId == 0)
            fEnableSave = 0;
    }

    if(fEnableSave == 1)
    {
     
        if(m_xFeatInfo.nDNNFeatCount != 0 || strlen(m_xMetaInfo.szPasscode) || m_xMetaInfo.nCardId != 0)
        {
            SaveClick();
        }
        else
        {
         
            DeleteClick();
        }
    }
    else
        SigBack();
}

void UserEditForm::FaceClick()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif
    if(m_xFeatInfo.nDNNFeatCount == 0 || m_xMetaInfo.iImageLen == 0)
        FaceNew();
    else
    {
//        m_pMenuFace->actions().at(2)->setVisible(true);
//        if(dbm_GetManagerCount() == 1 && m_xMetaInfo.fPrivilege == EMANAGER && m_iEditType == TYPE_EDIT)
//        {
//            if(!strlen(m_xMetaInfo.szPasscode) && m_xMetaInfo.nCardId == 0)
//            {
//                m_pMenuFace->actions().at(2)->setVisible(false);
//            }
//        }
        QPoint xViewPos(80 * LCD_RATE, m_vMenuItems[ID_FACE_ITEM]->pos().y() + FACE_ITEM_HEIGHT / 2);
        xViewPos = GetItemView()->mapFromScene(xViewPos);
        xViewPos = GetItemView()->mapToGlobal(xViewPos);

        PlaySoundLeft();

        if(xViewPos.y() > 170)
            xViewPos.setY(170);
        m_pMenuFace->exec(xViewPos);
    }
}

void UserEditForm::FaceNew()
{
    FaceEngine::UnregisterFace(m_xMetaInfo.nID);

    EnrollFaceForm* pEnrollFaceForm = new EnrollFaceForm(m_pParentView, this);
    connect(pEnrollFaceForm, SIGNAL(SigEnrollFinished(int)), this, SLOT(EnrollFaceFinished(int)));
    connect(pEnrollFaceForm, SIGNAL(SigBack()), this, SLOT(OnResume()));

    pEnrollFaceForm->StartEnroll();
}

void UserEditForm::FaceUpdate()
{
    m_iFaceUpdate = 1;
    FaceNew();
}

void UserEditForm::FaceDelete()
{
//    if(dbm_GetManagerCount() == 1 && m_xMetaInfo.fPrivilege == EMANAGER && m_iEditType == TYPE_EDIT)
//    {
//        if(!strlen(m_xMetaInfo.szPasscode) && m_xMetaInfo.nCardId == 0)
//        {
//            return;
//        }
//    }

    if(AlertDlg::WarningYesNo(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_Are_you_sure_you_want_to_delete_the_face_info) == QDialog::Accepted)
    {
        m_xFeatInfo.nDNNFeatCount = 0;
        memset(&m_xFeatInfo, 0, sizeof(m_xFeatInfo));
        m_xMetaInfo.iImageLen = 0;
        update();
    }

    RetranslateUI();
}

void UserEditForm::EnrollFaceFinished(int iEnrollResult)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    if(iEnrollResult == 1)
    {        
        FormBase::OnResume();
        PlayThankyouSound(0);

        int iRet = FaceEngine::GetRegisteredFaceImage((unsigned char*)m_xMetaInfo.abFaceImage, &m_xMetaInfo.iImageLen);
        if(iRet == ES_SUCCESS)
        {
            if(m_iFaceUpdate == 0)
                memset(&m_xFeatInfo, 0, sizeof(m_xFeatInfo));

            FaceEngine::GetRegisteredFeatInfo(&m_xFeatInfo);
        }

        RetranslateUI();
    }
    else if(iEnrollResult == 2)
    {
        FormBase::OnResume();
#if USING_BUZZER
        MainSTM_Command(MAIN_STM_BUZZER_FAILED);
#else
        PlayError5Sound();
#endif
        g_xSS.iNoSoundPlayFlag = 1;
        AlertDlg::WarningOk(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_Face_is_already_registerd);

        AlertDlg::Locked = 0;
    }
    else if(iEnrollResult == 3)
    {
        FormBase::OnResume();
#if USING_BUZZER
        MainSTM_Command(MAIN_STM_BUZZER_FAILED);
#else
        PlayError5Sound();
#endif

        AlertDlg::Locked = 0;
    }
    else if(iEnrollResult == 4)
    {
        FormBase::OnResume();
#if USING_BUZZER
        MainSTM_Command(MAIN_STM_BUZZER_FAILED);
#else
        PlayError5Sound();
#endif

        g_xSS.iNoSoundPlayFlag = 1;
        AlertDlg::WarningOk(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_Camera_Error);

        AlertDlg::Locked = 0;
    }
}

void UserEditForm::NameClick()
{
    QString strGroupName = AlertDlg::ContainLineEdit(m_pParentView, this, StringTable::Str_Name, QString::fromUtf8(m_xMetaInfo.szName), N_MAX_REAL_NAME_LEN);
    if(!strGroupName.isEmpty())
    {
        int iExists = 0;
        for(int i = 0; i < dbm_GetPersonCount(); i ++)
        {
            PSMetaInfo pxMetaInfo = dbm_GetPersonMetaInfoByIndex(i);
            if(pxMetaInfo == NULL)
                continue;

            if(m_iEditType == TYPE_EDIT && pxMetaInfo->nID == m_xOldMetaInfo.nID)
                continue;

            if(!strcmp(pxMetaInfo->szName, strGroupName.toUtf8().data()))
            {
                iExists = 1;
                break;
            }
        }

        if(iExists)
        {
            g_xSS.iNoSoundPlayFlag = 1;

            AlertDlg::WarningOk(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_Duplicated_name);
            strGroupName = QString::fromUtf8(m_xMetaInfo.szName);
        }
    }

    strcpy(m_xMetaInfo.szName, strGroupName.toUtf8().data());

    RetranslateUI();
}


void UserEditForm::CardClick()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    int iHasCard = 0;
    if(m_xMetaInfo.nCardId != 0)
        iHasCard = 1;

    if(iHasCard == 0)
    {
        CardNew();
    }
    else
    {
        PlaySoundLeft();

#if (AUTO_TEST != 1)
        QPoint xViewPos(40 * LCD_RATE, m_vMenuItems[ID_CARD_ITEM]->pos().y() + 20 * LCD_RATE);
        xViewPos = GetItemView()->mapFromScene(xViewPos);
        xViewPos = GetItemView()->mapToGlobal(xViewPos);

        if(xViewPos.y() > 170)
            xViewPos.setY(170);
        m_pMenuCard->exec(xViewPos);
#else
        int iIdx = rand() % 2;
        if(iIdx == 0)
            CardNew();
        else
            CardDelete();
#endif
    }
}

EnrollCardForm* pEnrollCardForm = NULL;

void UserEditForm::CardNew()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    pEnrollCardForm = new EnrollCardForm(m_pParentView, this);
    connect(pEnrollCardForm, SIGNAL(SigSendEnrollFinished(int)), this, SLOT(EnrollCardFinished(int)));

    pEnrollCardForm->setAttribute(Qt::WA_DeleteOnClose);
    pEnrollCardForm->StartEnroll(ENROLL_CARD_TIMEOUT);
}


void UserEditForm::CardDelete()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    if(AlertDlg::WarningYesNo(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_Are_you_sure_you_want_to_delete_the_card_info) == QDialog::Accepted)
    {
        m_xMetaInfo.nCardId = 0;
        m_xMetaInfo.iSectorNum = 0;
        m_xMetaInfo.iCardRand = 0;
    }

    RetranslateUI();
}

void UserEditForm::EnrollCardFinished(int iEnrollResult)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    if(iEnrollResult == 0)
    {
        FormBase::OnResume();
    }
    else if(iEnrollResult == 1)
    {
        EnrollCardForm* pEnrollCardForm = qobject_cast<EnrollCardForm*>(sender());
        int iCardID = pEnrollCardForm->GetEnrolledCardID();
        int iSectorNum = pEnrollCardForm->GetEnrolledSectorNum();
        int iCardRand = pEnrollCardForm->GetEnrolledCardRand();

            if(iExist >= 0)
            {
                FormBase::OnResume();
#if USING_BUZZER
                MainSTM_Command(MAIN_STM_BUZZER_FAILED);
#else
                PlayError5Sound();
#endif

                g_xSS.iNoSoundPlayFlag = 1;
                AlertDlg::WarningOk(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_Card_is_already_registerd);

                AlertDlg::Locked = 0;
                return;
            }
        }

        m_xMetaInfo.nCardId = iCardID;
        m_xMetaInfo.iSectorNum = iSectorNum;
        m_xMetaInfo.iCardRand = iCardRand;

        PlayThankyouSound(1);

        FormBase::OnResume();
    }

    AlertDlg::Locked = 0;

    if (pEnrollCardForm != NULL)
    {
        delete pEnrollCardForm;
        pEnrollCardForm = NULL;
    }
}

void UserEditForm::PasscodeClick()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    int iHasPass = 0;
    if(strlen(m_xMetaInfo.szPasscode) != 0)
        iHasPass = 1;

    if(iHasPass == 0)
    {
        PasscodeNew();
    }
    else
    {
        PlaySoundLeft();

#if (AUTO_TEST != 1)
        QPoint xViewPos(80 * LCD_RATE, m_vMenuItems[ID_PASSCODE_ITEM]->pos().y() + 20 * LCD_RATE);
        xViewPos = GetItemView()->mapFromScene(xViewPos);
        xViewPos = GetItemView()->mapToGlobal(xViewPos);
        if(xViewPos.y() > 170)
            xViewPos.setY(170);

        m_pMenuPasscode->exec(xViewPos);
#else
        int iIdx = rand() % 2;
        if(iIdx == 0)
            PasscodeNew();
        else
            PasscodeDelete();
#endif
    }
}

void UserEditForm::ReceivePasscode()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    PasscodeForm* pPasscodeForm = qobject_cast<PasscodeForm*>(sender());
    QString strNewPasscode = pPasscodeForm->GetPasscode();
    strcpy(m_xMetaInfo.szPasscode, strNewPasscode.toUtf8().data());

    PlayThankyouSound(1);

    FormBase::OnResume();
}

void UserEditForm::PasscodeNew()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    PasscodeForm* pPasscodeForm = new PasscodeForm(m_pParentView, this);
    connect(pPasscodeForm, SIGNAL(SigBack()), this, SLOT(OnResume()));
    connect(pPasscodeForm, SIGNAL(SigConfirm()), this, SLOT(ReceivePasscode()));
//    connect(pPasscodeForm, SIGNAL(SigHiddenCode()), this, SLOT(ShowHiddenCode()));

    pPasscodeForm->setAttribute(Qt::WA_DeleteOnClose);
    pPasscodeForm->Start(STEP_INPUT_NEW, "");
}

void UserEditForm::PasscodeDelete()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    if(AlertDlg::WarningYesNo(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_Are_you_sure_you_want_to_delete_the_password_info) == QDialog::Accepted)
    {
        memset(m_xMetaInfo.szPasscode, 0, sizeof(m_xMetaInfo.szPasscode));
    }

    RetranslateUI();
}

void UserEditForm::PrivilegeClick()
{
    PlaySoundLeft();

    int iCurTheme = m_xMetaInfo.fPrivilege;

    QString strPrivieleges[] =
    {
        StringTable::Str_User,
        StringTable::Str_Manager,
    };
    m_pMenu->clear();

    QAction* pThemeActions[2] = { 0 };
    for(int i = 0; i < 2; i ++)
    {
        pThemeActions[i] = m_pMenu->addAction(strPrivieleges[i]);
        pThemeActions[i]->setCheckable(true);
        pThemeActions[i]->setChecked(i == iCurTheme);
    }

    QPoint xViewPos(200 * LCD_RATE, m_vMenuItems[ID_PRIVILEGE_ITEM]->pos().y() + 20 * LCD_RATE);
    xViewPos = GetItemView()->mapFromScene(xViewPos);
    xViewPos = GetItemView()->mapToGlobal(xViewPos);

#if (AUTO_TEST != 1)
    if(xViewPos.y() > 170)
        xViewPos.setY(170);

    QAction* pSelAction = m_pMenu->exec(xViewPos);
#else
    int iIdx = rand() % g_iLangCount;
    QAction* pSelAction = pLangActions[iIdx];
#endif
    for(int i = 0; i < 2; i ++)
    {
        if(pThemeActions[i] == pSelAction)
        {
            m_xMetaInfo.fPrivilege = i;
        }
    }

    PlaySoundLeft();

    RetranslateUI();
}

void UserEditForm::GroupClick()
{
    SGroupName* pxGroup = dbm_GetGroupByID(m_xMetaInfo.iGroupID);
    if(pxGroup == NULL)
    {
        GroupUpdate();
    }
    else
    {
        PlaySoundLeft();

        QPoint xViewPos(200 * LCD_RATE, m_vMenuItems[ID_GROUP_ITEM]->pos().y() + 20 * LCD_RATE);
        xViewPos = GetItemView()->mapFromScene(xViewPos);
        xViewPos = GetItemView()->mapToGlobal(xViewPos);
        if(xViewPos.y() > 170)
            xViewPos.setY(170);

        m_pMenuGroup->exec(xViewPos);
    }
}

void UserEditForm::GroupNew()
{
    GroupForm* pGroupForm = new GroupForm(m_pParentView, this);
    connect(pGroupForm, SIGNAL(SigBack()), this, SLOT(OnResume()));

    pGroupForm->onStart();
}

void UserEditForm::GroupUpdate()
{
    AlertDlg* pDlg = new AlertDlg(m_pParentView, this);
    pDlg->SetTitle(StringTable::Str_Group);
    pDlg->SetButtonGroup(false);
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_Edit);
    pDlg->SetCancelButton(true, StringTable::Str_Cancel);
    pDlg->setAttribute(Qt::WA_DeleteOnClose);

    int iGroupCount = dbm_GetGroupCount();
    for(int i = 0; i < iGroupCount; i ++)
    {        
        SGroupName* pxGroup = dbm_GetGroup(i);
        QString strGroupName = QString::fromUtf8(pxGroup->szName);
//        QString strOmitText = CalcOmitText(qApp->font(), strGroupName, 150 * LCD_RATE);
        pDlg->AddRadioItem(strGroupName, 2 + pxGroup->iGroupID, pxGroup->iGroupID == m_xMetaInfo.iGroupID);
    }

    int iRet = pDlg->OnExec();
    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    pDlgScene->clear();
    delete pDlgScene;

    if(iRet == QDialog::Accepted)
    {
        g_xSS.iNoSoundPlayFlag = 1;
        GroupNew();
    }
    else if(iRet > QDialog::Accepted)
        m_xMetaInfo.iGroupID = iRet - 2;

    RetranslateUI();
}

void UserEditForm::GroupDelete()
{
    if(AlertDlg::WarningYesNo(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_Are_you_sure_to_delete_this_group) == QDialog::Accepted)
    {
        m_xMetaInfo.iGroupID = 0;
    }

    RetranslateUI();
}

void UserEditForm::DeletePerson()
{
    dbi_RemoveUserByID(m_xOldMetaInfo.nID);
}

void UserEditForm::SavePerson()
{
    dbi_SaveUser(&m_xMetaInfo, &m_xFeatInfo);
}

void UserEditForm::SaveClick()
{        
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

#if 1
    if(strlen(m_xMetaInfo.szName) == 0)
    {
        GetUserName(m_xMetaInfo.szName, m_xMetaInfo.nID, 0);
        RetranslateUI();
    }

    int iExists = 0;
    for(int i = 0; i < dbm_GetPersonCount(); i ++)
    {
        PSMetaInfo pxMetaInfo = dbm_GetPersonMetaInfoByIndex(i);
        if(pxMetaInfo == NULL)
            continue;

        if(m_iEditType == TYPE_EDIT && pxMetaInfo->nID == m_xOldMetaInfo.nID)
            continue;

        if(!strcmp(pxMetaInfo->szName, m_xMetaInfo.szName))
        {
            iExists = 1;
            break;
        }
    }

    if(iExists)
    {
        g_xSS.iNoSoundPlayFlag = 1;
        AlertDlg::WarningOk(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_Duplicated_name);
        return;
    }
#endif

    WaitingForm::Waiting(m_pParentView, this, std::bind(&UserEditForm::SavePerson, this));

    emit SigSave();
}

void UserEditForm::DeleteClick()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif
    if(m_iEditType == TYPE_NEW)
        return;

    if(AlertDlg::WarningYesNo(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_Are_you_sure_to_delete_the_user) == QDialog::Accepted)
    {
        g_xSS.iNoSoundPlayFlag = 1;
        WaitingForm::Waiting(m_pParentView, this, std::bind(&UserEditForm::DeletePerson, this));

        g_xSS.iNoSoundPlayFlag = 1;
        emit SigSave();
    }
}

void UserEditForm::RetranslateUI()
{
    if(m_iUserType == TYPE_MANAGER)
    {
        if(m_iEditType == TYPE_NEW)
            SetTitle(StringTable::Str_New_Manager);
        else
            SetTitle(StringTable::Str_Manager_Info);
    }
    else
    {
        if(m_iEditType == TYPE_NEW)
            SetTitle(StringTable::Str_New_User);
        else
            SetTitle(StringTable::Str_User_Info);
    }

    QImage xFaceImg;
    if(m_xMetaInfo.iImageLen == 0)
    {
        xFaceImg = QImage(":/icons/ic_face_add.png");
    }
    else
    {
        xFaceImg = ConvertData2QImage(m_xMetaInfo.abFaceImage, m_xMetaInfo.iImageLen);

        QSize iconSize(90 * LCD_RATE, 90 * LCD_RATE);
        xFaceImg = GetCircleOverlayImage(iconSize, xFaceImg);
    }

    m_vMenuItems[ID_FACE_ITEM]->setData(KEY_IS_RED_BOARDER, (m_xFeatInfo.nDNNFeatCount <= 0 && m_xMetaInfo.iImageLen > 0) ? 1 : 0);
    m_vMenuItems[ID_FACE_ITEM]->setData(KEY_ICON, xFaceImg);
    m_vMenuItems[ID_FACE_ITEM]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Face);
    m_vMenuItems[ID_FACE_ITEM]->setData(KEY_SECONDARY_TEXT, QString::number(m_xMetaInfo.nID + 1));

    if(!strlen(m_xMetaInfo.szName))
    {
        m_vMenuItems[ID_NAME_ITEM]->setData(KEY_ICON, QImage(":/icons/ic_enroll_plus.png"));
        m_vMenuItems[ID_NAME_ITEM]->setData(KEY_SECONDARY_TEXT, "");
    }
    else
    {
        QString strName = QString::fromUtf8(m_xMetaInfo.szName);
        strName = CalcOmitText(g_UITheme->SecondaryFont, strName, 100 * LCD_RATE);
        m_vMenuItems[ID_NAME_ITEM]->setData(KEY_SECONDARY_TEXT, strName);
        m_vMenuItems[ID_NAME_ITEM]->setData(KEY_ICON, QImage());
    }

    if(m_vMenuItems[ID_PRIVILEGE_ITEM])
    {
        if(m_xMetaInfo.fPrivilege == EMANAGER)
        {
            m_vMenuItems[ID_PRIVILEGE_ITEM]->setData(KEY_SECONDARY_TEXT, StringTable::Str_Manager);
        }
        else
        {
            m_vMenuItems[ID_PRIVILEGE_ITEM]->setData(KEY_SECONDARY_TEXT, StringTable::Str_User);
        }

        m_vMenuItems[ID_PRIVILEGE_ITEM]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Privilege);
        m_vMenuItems[ID_PRIVILEGE_ITEM]->setData(KEY_ICON, QImage());
    }

    if(m_xMetaInfo.nCardId != 0)
        m_vMenuItems[ID_CARD_ITEM]->setData(KEY_ICON, QImage(":/icons/ic_enroll_card.png"));
    else
        m_vMenuItems[ID_CARD_ITEM]->setData(KEY_ICON, QImage(":/icons/ic_enroll_card_null.png"));

    if(strlen(m_xMetaInfo.szPasscode) != 0)
    {
        m_vMenuItems[ID_PASSCODE_ITEM]->setData(KEY_ICON, QImage(":/icons/ic_enroll_password.png"));
//        if(m_xMetaInfo.iTempPasscodeType == E_PASS_TIMEOUT_COUNTLIMIT)
//            m_vMenuItems[ID_PASSCODE_ITEM]->setData(KEY_SECONDARY_TEXT, "( " + QString::number(m_xMetaInfo.iTempCounter) + " x )     ");
//        else
            m_vMenuItems[ID_PASSCODE_ITEM]->setData(KEY_SECONDARY_TEXT, "");
    }
    else
    {
        m_vMenuItems[ID_PASSCODE_ITEM]->setData(KEY_ICON, QImage(":/icons/ic_enroll_password_null.png"));
        m_vMenuItems[ID_PASSCODE_ITEM]->setData(KEY_SECONDARY_TEXT, "");
    }

    if(m_vMenuItems[ID_GROUP_ITEM] != NULL)
    {
        SGroupName* pxGroup = dbm_GetGroupByID(m_xMetaInfo.iGroupID);
        if(pxGroup == NULL)
        {
            m_vMenuItems[ID_GROUP_ITEM]->setData(KEY_ICON, QImage(":/icons/ic_enroll_plus.png"));
            m_vMenuItems[ID_GROUP_ITEM]->setData(KEY_SECONDARY_TEXT, "");
        }
        else
        {
            QString strGroupName = QString::fromUtf8(pxGroup->szName);
            strGroupName = CalcOmitText(g_UITheme->SecondaryFont, strGroupName, 100 * LCD_RATE);

            m_vMenuItems[ID_GROUP_ITEM]->setData(KEY_SECONDARY_TEXT, strGroupName);
            m_vMenuItems[ID_GROUP_ITEM]->setData(KEY_ICON, QImage());
        }
    }    
}


