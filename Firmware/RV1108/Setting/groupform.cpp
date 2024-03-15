#include "groupform.h"
#include "DBManager.h"
#include "menuitem.h"
#include "base.h"
#include "alertdlg.h"
#include "stringtable.h"
#include "userform.h"
#include "settings.h"
#include "uitheme.h"
#include "soundbase.h"
#include "themedef.h"

#include <QtGui>
#include <QDialog>
#include <QMenu>
#include <QScrollBar>

GroupForm::GroupForm(QGraphicsView *view, FormBase* parentForm) :
    ItemFormBase(view, parentForm)
{
    SetTitle(StringTable::Str_Group);
    QString strSheets;
    strSheets.sprintf("background-color: rgb(%d, %d, %d);", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue());
    setStyleSheet(strSheets);
    m_pMenu = new QMenu(this);
    m_pMenu->addAction(StringTable::Str_Update, this, SLOT(GroupUpdate()));
    m_pMenu->addAction(StringTable::Str_Delete, this, SLOT(GroupDelete()));
    m_pMenu->setStyleSheet("QMenu {background-color: white; margin: 0px;}"
                            "QMenu::item {padding: 8px 30px 8px 20px;border: 1px solid transparent;}"
                            "QMenu::item:selected {border-color: white ;background: rgba(100, 100, 100, 150);}"
                            "QMenu::separator {height: 1px;background: lightblue;margin-left: 10px;margin-right: 5px;}"
                            "QMenu::indicator:non-exclusive:checked {image: url(:/icons/ic_check.png);}");



    m_pMenu->setFont(g_UITheme->PrimaryFont);
}

GroupForm::~GroupForm()
{
    delete m_pMenu;
}

void GroupForm::onStart()
{
    RefreshItems();
    FormBase::OnStart();
}

void GroupForm::GroupAdd()
{
    QString strGroupName = AlertDlg::ContainLineEdit(m_pParentView, this, StringTable::Str_New_Group, QString(), N_MAX_GROUP_NAME_SIZE);
    if(strGroupName.isEmpty())
        return;

    dbm_AddGroup(strGroupName.toUtf8().data());
    QTimer::singleShot(0, this, SLOT(RefreshItems()));
}

void GroupForm::GroupClick(int iID)
{
    m_iSelectIdx = iID;

    if(m_vMenuItems[iID])
    {
        PlaySoundLeft();

        QPoint xViewPos(200 * LCD_RATE, m_vMenuItems[iID]->pos().y() + 20 * LCD_RATE);
        xViewPos = GetItemView()->mapFromScene(xViewPos);
        xViewPos = GetItemView()->mapToGlobal(xViewPos);
        if(xViewPos.y() > 170)
            xViewPos.setY(170);
        m_pMenu->exec(xViewPos);
    }
}

void GroupForm::GroupUpdate()
{
    QString strOldGroupName = QString::fromUtf8(dbm_GetGroupName(m_iSelectIdx));
    QString strGroupName = AlertDlg::ContainLineEdit(m_pParentView, this, StringTable::Str_Group, strOldGroupName, N_MAX_GROUP_NAME_SIZE);
    if(strGroupName.isEmpty() || strOldGroupName == strGroupName)
        return;

    dbm_ModifyGroup(strGroupName.toUtf8().data(), strOldGroupName.toUtf8().data());
    QTimer::singleShot(0, this, SLOT(RefreshItems()));
}

void GroupForm::GroupDelete()
{
    if(AlertDlg::WarningYesNo(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_Are_you_sure_to_delete_this_group) == QDialog::Accepted)
    {
        QString strOldGroupName = QString::fromUtf8(dbm_GetGroupName(m_iSelectIdx));
        dbm_RemoveGroupByName(strOldGroupName.toUtf8().data());
    }
    QTimer::singleShot(0, this, SLOT(RefreshItems()));
}


void GroupForm::RefreshItems()
{
    m_pScene->clear();
    GetItemView()->setSceneRect(QRect(0, 0, 0, 0));

    int iGroupCount = dbm_GetGroupCount();
    for(int i = 0; i < iGroupCount; i ++)
    {
        QString strGroupName = QString::fromUtf8(dbm_GetGroupName(i));
        strGroupName = CalcOmitText(g_UITheme->PrimaryFont, strGroupName, 180 * LCD_RATE);

        MenuItem* pGroupItem = new MenuItem(i);
        pGroupItem->setPos(QPoint(0, i * SETTING_ITEM_HEIGHT));
        pGroupItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
        pGroupItem->setData(KEY_PRIMARY_TEXT, strGroupName);
        pGroupItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
        m_pScene->addItem(pGroupItem);
        m_vMenuItems[i] = pGroupItem;

        connect(pGroupItem, SIGNAL(clicked(int)), this, SLOT(GroupClick(int)));
    }

    MenuItem* pBtnAdd = new MenuItem();
    pBtnAdd->setPos(QPoint(ADD_BTN_X, ADD_BTN_Y));
    pBtnAdd->setData(KEY_FIX_POS, QPoint(ADD_BTN_X, ADD_BTN_Y));
    pBtnAdd->SetBoundingRect(QRect(0, 0, ACTION_BUTTON_WIDTH, ACTION_BUTTON_HEIGHT));
    pBtnAdd->setData(KEY_ICON, QImage(":/icons/ic_user_plus.png"));
    pBtnAdd->setData(KEY_ICON_DISABLE, QImage(":/icons/ic_user_plus_disable.png"));
    pBtnAdd->setData(KEY_TYPE, TYPE_ACTION_BUTTON);
    pBtnAdd->setZValue(1);
    pBtnAdd->setEnabled(iGroupCount < N_MAX_GROUP_COUNT);

    m_pScene->addItem(pBtnAdd);

    connect(GetItemView()->verticalScrollBar(), SIGNAL(valueChanged(int)), pBtnAdd, SLOT(ScrollChnaged(int)));
    connect(pBtnAdd, SIGNAL(clicked()), this, SLOT(GroupAdd()));

    QRect sceneRect = GetItemView()->sceneRect().toRect();
    if(sceneRect.height() < 170)
        GetItemView()->setSceneRect(QRect(0, 0, MAX_X, 170));

    pBtnAdd->ScrollChnaged(GetItemView()->verticalScrollBar()->value());
}
