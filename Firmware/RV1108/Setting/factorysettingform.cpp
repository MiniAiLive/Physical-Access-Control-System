#include "factorysettingform.h"
#include "stringtable.h"
#include "logosettingform.h"
#include "shared.h"
#include "uitheme.h"
#include "menuitem.h"
#include "sliderspin.h"
#include "alertdlg.h"
#include "i2cbase.h"
#include "qrcodereader.h"
#include "base.h"
#include "uarttask.h"
#include "mainbackproc.h"
#include "soundbase.h"

#include <unistd.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>

#include <QtGui>
#include <QMenu>
#include <QHBoxLayout>
#include <QDialog>

enum
{
    ITEM_LOGO_ID,
    ITEM_MOTO_POLARITY_ID,
    ITEM_OVER_CURRENT_ID,
    ITEM_HIDDEN_CODE_ID,
    ITEM_HOME_AUTOMATION_ID,
    ITEM_BATT_LOW_ID,
    ITEM_BATT_STEP_ID,
    ITEM_BATT_NEW_ID,
    ITEM_BATT_POWER_DOWN_ID,
    ITEM_BATT_MENU_ID,
    ITEM_BATT_UPDATE_ID,
    ITEM_BATT_SOUND_OFF_ID,
    ITEM_SHOW_BATT_ID,
#if (LOCK_MODE == LM_AUTO)
    ITEM_MOTOR_ID,
#endif /* LOCK_MODE == LM_AUTO */
    ITEM_INNER_VERSION_ID,
    ITEM_HUMAN_SENSOR_TYPE_ID,
    #if (TEST_DUTY_CYCLE == 1)
    ITEM_DUTY_CYCLE_VALUES,
    #endif //TEST_DUTY_CYCLE
    ITEM_GYRO_AXIS_ID
};

#define ALERT_TITLE_FONT_SIZE_EN (14 * LCD_RATE)

extern UARTTask* g_pUartTask;

FactorySettingForm::FactorySettingForm(QGraphicsView *pView, FormBase* pParentForm) :
    ItemFormBase(pView , pParentForm)
{
    SetBGColor(g_UITheme->itemNormalBgColor);

    int iPosY = 0;    
    MenuItem* pLogoItem = new MenuItem();
    pLogoItem->setPos(QPoint(0, iPosY));
    pLogoItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pLogoItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pLogoItem);
    m_vMenuItems[ITEM_LOGO_ID] = pLogoItem;
    connect(pLogoItem, SIGNAL(clicked()), this, SLOT(ClickedLogo()));
    iPosY += SETTING_ITEM_HEIGHT;


#if 0
    MenuItem* pMotoTypeItem = new MenuItem();
    pMotoTypeItem->setPos(QPoint(0, iPosY));
    pMotoTypeItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pMotoTypeItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pMotoTypeItem);
    m_vMenuItems[ITEM_MOTO_TYPE_ID] = pMotoTypeItem;
    connect(pMotoTypeItem, SIGNAL(clicked()), this, SLOT(ClickedMotoType()));
    iPosY += SETTING_ITEM_HEIGHT;
#endif

#if (LOCK_MODE == LM_SEMI_AUTO)
    MenuItem* pMotoPolarityItem = new MenuItem();
    pMotoPolarityItem->setPos(QPoint(0, iPosY));
    pMotoPolarityItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pMotoPolarityItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pMotoPolarityItem);
    m_vMenuItems[ITEM_MOTO_POLARITY_ID] = pMotoPolarityItem;
    connect(pMotoPolarityItem, SIGNAL(clicked()), this, SLOT(ClickedMotoPolarity()));
    iPosY += SETTING_ITEM_HEIGHT;
#endif

#if 0
    MenuItem* pMotoTimeItem = new MenuItem();
    pMotoTimeItem->setPos(QPoint(0, iPosY));
    pMotoTimeItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pMotoTimeItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pMotoTimeItem);
    m_vMenuItems[ITEM_MOTO_TIME_ID] = pMotoTimeItem;
    connect(pMotoTimeItem, SIGNAL(clicked()), this, SLOT(ClickedMotoTime()));
    iPosY += SETTING_ITEM_HEIGHT;

#endif

#if 0
#if (LOCK_MODE == LM_AUTO)
    MenuItem* pOverCurrentItem = new MenuItem();
    pOverCurrentItem->setPos(QPoint(0, iPosY));
    pOverCurrentItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pOverCurrentItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pOverCurrentItem);
    m_vMenuItems[ITEM_OVER_CURRENT_ID] = pOverCurrentItem;
    connect(pOverCurrentItem, SIGNAL(clicked()), this, SLOT(ClickedOverCurrent()));
    iPosY += SETTING_ITEM_HEIGHT;
#endif
#endif

    MenuItem* pHiddenCodeItem = new MenuItem();
    pHiddenCodeItem->setPos(QPoint(0, iPosY));
    pHiddenCodeItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pHiddenCodeItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pHiddenCodeItem);
    m_vMenuItems[ITEM_HIDDEN_CODE_ID] = pHiddenCodeItem;
    iPosY += SETTING_ITEM_HEIGHT;

    connect(pHiddenCodeItem, SIGNAL(clicked()), this, SLOT(HiddenCodeClick()));

    MenuItem* pHomeAutomationItem = new MenuItem();
    pHomeAutomationItem->setPos(QPoint(0, iPosY));
    pHomeAutomationItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pHomeAutomationItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pHomeAutomationItem);
    m_vMenuItems[ITEM_HOME_AUTOMATION_ID] = pHomeAutomationItem;
    iPosY += SETTING_ITEM_HEIGHT;

    connect(pHomeAutomationItem, SIGNAL(clicked()), this, SLOT(HomeAutoMationClick()));

    MenuItem* pBattLowItem = new MenuItem();
    pBattLowItem->setPos(QPoint(0, iPosY));
    pBattLowItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pBattLowItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pBattLowItem);
    m_vMenuItems[ITEM_BATT_LOW_ID] = pBattLowItem;
    connect(pBattLowItem, SIGNAL(clicked()), this, SLOT(BattLowClick()));
    iPosY += SETTING_ITEM_HEIGHT;


    MenuItem* pBattStepItem = new MenuItem();
    pBattStepItem->setPos(QPoint(0, iPosY));
    pBattStepItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pBattStepItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pBattStepItem);
    m_vMenuItems[ITEM_BATT_STEP_ID] = pBattStepItem;
    connect(pBattStepItem, SIGNAL(clicked()), this, SLOT(BattStepClick()));
    iPosY += SETTING_ITEM_HEIGHT;

    MenuItem* pBattNewItem = new MenuItem();
    pBattNewItem->setPos(QPoint(0, iPosY));
    pBattNewItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pBattNewItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pBattNewItem);
    m_vMenuItems[ITEM_BATT_NEW_ID] = pBattNewItem;
    connect(pBattNewItem, SIGNAL(clicked()), this, SLOT(BattNewClick()));
    iPosY += SETTING_ITEM_HEIGHT;

    MenuItem* pBattPowerDownItem = new MenuItem();
    pBattPowerDownItem->setPos(QPoint(0, iPosY));
    pBattPowerDownItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pBattPowerDownItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pBattPowerDownItem);
    m_vMenuItems[ITEM_BATT_POWER_DOWN_ID] = pBattPowerDownItem;
    connect(pBattPowerDownItem, SIGNAL(clicked()), this, SLOT(BattPowerDownClick()));
    iPosY += SETTING_ITEM_HEIGHT;

    MenuItem* pBattMenuItem = new MenuItem();
    pBattMenuItem->setPos(QPoint(0, iPosY));
    pBattMenuItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pBattMenuItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pBattMenuItem);
    m_vMenuItems[ITEM_BATT_MENU_ID] = pBattMenuItem;
    connect(pBattMenuItem, SIGNAL(clicked()), this, SLOT(BattMenuClick()));
    iPosY += SETTING_ITEM_HEIGHT;

    MenuItem* pBattUpdateItem = new MenuItem();
    pBattUpdateItem->setPos(QPoint(0, iPosY));
    pBattUpdateItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pBattUpdateItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pBattUpdateItem);
    m_vMenuItems[ITEM_BATT_UPDATE_ID] = pBattUpdateItem;
    connect(pBattUpdateItem, SIGNAL(clicked()), this, SLOT(BattUpdateClick()));
    iPosY += SETTING_ITEM_HEIGHT;

    MenuItem* pBattSoundOffItem = new MenuItem();
    pBattSoundOffItem->setPos(QPoint(0, iPosY));
    pBattSoundOffItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pBattSoundOffItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pBattSoundOffItem);
    m_vMenuItems[ITEM_BATT_SOUND_OFF_ID] = pBattSoundOffItem;
    connect(pBattSoundOffItem, SIGNAL(clicked()), this, SLOT(BattSoundOffClick()));
    iPosY += SETTING_ITEM_HEIGHT;

#if 0
    MenuItem* pHumanSensorTypeItem = new MenuItem();
    pHumanSensorTypeItem->setPos(QPoint(0, iPosY));
    pHumanSensorTypeItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pHumanSensorTypeItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pHumanSensorTypeItem);
    m_vMenuItems[ITEM_HUMAN_SENSOR_TYPE_ID] = pHumanSensorTypeItem;
    connect(pHumanSensorTypeItem, SIGNAL(clicked()), this, SLOT(HumanSensorClick()));
    iPosY += SETTING_ITEM_HEIGHT;
#endif

#if (TEST_DUTY_CYCLE == 1)
    MenuItem* pItem = new MenuItem();
    pItem->setPos(QPoint(0, iPosY));
    pItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
    pItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
    m_pScene->addItem(pItem);
    m_vMenuItems[ITEM_DUTY_CYCLE_VALUES] = pItem;
    iPosY += SETTING_ITEM_HEIGHT;
    connect(pItem, SIGNAL(clicked()), this, SLOT(DutyCycleValuesClick()));    
#endif //TEST_DUTY_CYCLE

    if(g_xSS.iShowInnerVersion == 1)
    {
        MenuItem* pShowBattItem = new MenuItem();
        pShowBattItem->setPos(QPoint(0, iPosY));
        pShowBattItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
        pShowBattItem->setData(KEY_TYPE, TYPE_SWITCH_ITEM);
        m_pScene->addItem(pShowBattItem);
        m_vMenuItems[ITEM_SHOW_BATT_ID] = pShowBattItem;
        connect(pShowBattItem, SIGNAL(clicked()), this, SLOT(ClickedShowBatt()));
        iPosY += SETTING_ITEM_HEIGHT;
#if (LOCK_MODE == LM_AUTO)
        MenuItem* pMotorItem = new MenuItem();
        pMotorItem->setPos(QPoint(0, iPosY));
        pMotorItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
        pMotorItem->setData(KEY_TYPE, TYPE_SWITCH_ITEM);
        m_pScene->addItem(pMotorItem);
        m_vMenuItems[ITEM_MOTOR_ID] = pMotorItem;
        connect(pMotorItem, SIGNAL(clicked()), this, SLOT(MotorClick()));
        iPosY += SETTING_ITEM_HEIGHT;

        MenuItem* pGyroAxisItem = new MenuItem();
        pGyroAxisItem->setPos(QPoint(0, iPosY));
        pGyroAxisItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
        pGyroAxisItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
        m_pScene->addItem(pGyroAxisItem);
        m_vMenuItems[ITEM_GYRO_AXIS_ID] = pGyroAxisItem;
        iPosY += SETTING_ITEM_HEIGHT;
        connect(pGyroAxisItem, SIGNAL(clicked()), this, SLOT(GyroAxisClick()));
#endif /* LOCK_MODE == LM_AUTO */
        MenuItem* pInnerVersionItem = new MenuItem();
        pInnerVersionItem->setPos(QPoint(0, iPosY));
        pInnerVersionItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
        pInnerVersionItem->setData(KEY_TYPE, TYPE_SETTING_ITEM);
        m_pScene->addItem(pInnerVersionItem);
        m_vMenuItems[ITEM_INNER_VERSION_ID] = pInnerVersionItem;        
        iPosY += SETTING_ITEM_HEIGHT;
    }

    m_pMenu = new QMenu(this);
    m_pMenu->setStyleSheet("QMenu {background-color: white; margin: 0px;}"
                            "QMenu::item {padding: 8px 30px 8px 20px;border: 1px solid transparent;}"
                            "QMenu::item:selected {border-color: white ;background: rgba(100, 100, 100, 150);}"
                            "QMenu::separator {height: 1px;background: lightblue;margin-left: 10px;margin-right: 5px;}"
                            "QMenu::indicator:non-exclusive:checked {image: url(:/icons/ic_check.png);}");


    m_pMenu->setFont(g_UITheme->PrimaryFont);
}

FactorySettingForm::~FactorySettingForm()
{
    OnStop();
}

void FactorySettingForm::OnStart()
{
    g_xSS.iCurLogo = g_xPS.x.bLogo;
    FormBase::OnStart();
}

void FactorySettingForm::ClickedLogo()
{
    LogoSettingForm* pLogoForm = new LogoSettingForm(m_pParentView, this);
    connect(pLogoForm, SIGNAL(SigBack()), this, SLOT(OnResume()));

    pLogoForm->OnStart();
}

void FactorySettingForm::ClickedMotoType()
{
#if 0
    m_pMenu->clear();

    QString strMotoType[] = {StringTable::Str_Two, StringTable::Str_One};
    QAction* pMotoTypeAction[E_MOTO_TYPE_END] = { 0 };
    for(int i = 0; i < E_MOTO_TYPE_END; i ++)
    {
        pMotoTypeAction[i] = m_pMenu->addAction(strMotoType[i]);
        pMotoTypeAction[i]->setCheckable(true);
        pMotoTypeAction[i]->setChecked(i == g_xPS.bMotoType);
    }

    QPoint xViewPos(130 * LCD_RATE, m_vMenuItems[ITEM_MOTO_TYPE_ID]->pos().y() + 20 * LCD_RATE);
    xViewPos = GetItemView()->mapFromScene(xViewPos);
    xViewPos = GetItemView()->mapToGlobal(xViewPos);

#if (AUTO_TEST != 1)
    if(xViewPos.y() > 650)
        xViewPos.setY(650);

    QAction* pSelAction = m_pMenu->exec(xViewPos);
#else
    int iIdx = rand() % E_MOTO_TYPE_END;
    QAction* pSelAction = pMotoTypeAction[iIdx];
#endif
    for(int i = 0; i < E_MOTO_TYPE_END; i ++)
    {
        if(pMotoTypeAction[i] == pSelAction)
        {
            SCardInfo xCardInfo = { 0 };
            M24C64_GetHeaderInfo(&xCardInfo);

            if(i == E_ONE)
            {
                g_xFS.bKeepOpen = 0;
                xCardInfo.fKeepOpen = 0;
                xCardInfo.fLockType = 0;
            }
            else
            {
                g_xFS.bKeepOpen = 1;
                xCardInfo.fKeepOpen = 0;
                xCardInfo.fLockType = 0;
            }

            xCardInfo.fMotoType = i;
            M24C64_SetHeaderInfo(&xCardInfo);

            g_xFS.bMotoType = i;
            UpdateFactorySettings();
            UpdateCommonSettings();

            RetranslateUI();
        }
    }
#endif
}

void FactorySettingForm::ClickedMotoPolarity()
{
    m_pMenu->clear();

    QString strMotoPolarity[] = {"+/-", "-/+"};
    QAction* pMotoPolarityActions[E_MOTO_POLARITY_END] = { 0 };
    QPoint xViewPos(130 * LCD_RATE, m_vMenuItems[ITEM_MOTO_POLARITY_ID]->pos().y() + 20 * LCD_RATE);
    xViewPos = GetItemView()->mapFromScene(xViewPos);
    xViewPos = GetItemView()->mapToGlobal(xViewPos);

#if (AUTO_TEST != 1)
    if(xViewPos.y() > 650)
        xViewPos.setY(650);

    QAction* pSelAction = m_pMenu->exec(xViewPos);
#else
    int iIdx = rand() % E_MOTO_POLARITY_END;
    QAction* pSelAction = pMotoPolarityActions[iIdx];
#endif
    for(int i = 0; i < E_MOTO_POLARITY_END; i ++)
    {
        if(pMotoPolarityActions[i] == pSelAction)
        {
            g_xPS.x.bSemiMotorPolarity = i;
            UpdatePermanenceSettings();

            RetranslateUI();
        }
    }
}

void FactorySettingForm::ClickedMotoTime()
{
#if 1
    if(g_xPS.x.bSemiMotorType == E_ONE)
        return;

    QWidget* pSurface = new QWidget;
    SliderSpin* pChannelSpin = new SliderSpin;
    QHBoxLayout* pLayout = new QHBoxLayout;
    QString strSheets;
    pLayout->setMargin(0);
    pLayout->addWidget(pChannelSpin);
    pSurface->setFixedSize(188 * LCD_RATE, pChannelSpin->height());
    pSurface->setLayout(pLayout);
    strSheets.sprintf("QWidget { background-color: rgb(%d, %d, %d); }", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue());
    pSurface->setStyleSheet(strSheets);

    QStringList vRangeList;
    for(int i = 0; i <= 190; i += 5)
        vRangeList.append(QString::number(i * 10 + 100));

    pChannelSpin->SetSpinRange(vRangeList);
    pChannelSpin->SetSelect((g_xPS.x.bSemiMotorOpenTime) / 5);

    AlertDlg* pDlg = new AlertDlg(m_pParentView, this);
    pDlg->SetTitle(StringTable::Str_Moto_Time);
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_OK);
    pDlg->SetCancelButton(true, StringTable::Str_Cancel);
    pDlg->AddWidget(pSurface);
    int iRet = pDlg->OnExec();
    if(iRet == QDialog::Accepted)
    {
        int iSelectedIdx = pChannelSpin->GetSelectedIndex();

        g_xPS.x.bSemiMotorOpenTime = (iSelectedIdx * 5);
        UpdatePermanenceSettings();
    }

    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    pDlgScene->clear();
    delete pDlgScene;

    RetranslateUI();
#endif
}

void FactorySettingForm::ClickedKeepOpen()
{
#if 0
    if(g_xFS.bMotoType == E_ONE)
        return;

    SCardInfo xCardInfo = { 0 };
    M24C64_GetHeaderInfo(&xCardInfo);

    g_xFS.bKeepOpen = 1 - g_xFS.bKeepOpen;
    xCardInfo.fKeepOpen = 0;
    xCardInfo.fLockType = 0;
    M24C64_SetHeaderInfo(&xCardInfo);

    UpdateCommonSettings();
    UpdateFactorySettings();
    RetranslateUI();
#endif
}

void FactorySettingForm::ClickedPresentation()
{
#if 0
    SCardInfo xCardInfo = { 0 };
    M24C64_GetHeaderInfo(&xCardInfo);

    g_xFS.bPresentation = 1 - g_xFS.bPresentation;
    if(g_xFS.bPresentation == 0)
        xCardInfo.fPresentation = 0;

    if(g_xFS.bBattTest == 1)
        xCardInfo.fPresentation = 1;
    else
        xCardInfo.fPresentation = 0;

    M24C64_SetHeaderInfo(&xCardInfo);

    UpdateFactorySettings();
    UpdateCommonSettings();
    RetranslateUI();
#endif
}

void FactorySettingForm::ClickedBattTest()
{
#if 0
    g_xFS.bBattTest = 1 - g_xFS.bBattTest;

    if(g_xFS.bBattTest == 1 || g_xSS.bPresentation == 1)
    {
        g_xCS.x.bPresentation = 1;
        g_xSS.bPresentation = 0;
    }
    else
    {
        g_xCS.x.bPresentation = 0;
        g_xSS.bPresentation = 0;
    }
    UpdateCommonSettings();
#endif

    RetranslateUI();
}

void FactorySettingForm::ClickedShowBatt()
{
    g_xFS.bShowBatt = 1 - g_xFS.bShowBatt;
    UpdateFactorySettings();

    RetranslateUI();
}

void FactorySettingForm::GyroAxisClick()
{
#define GYRO_AXIS_COUNT 2


    QString strGyroAxisValue[] =
    {
        StringTable::Str_Horizontal,
        StringTable::Str_Vertical
    };

    int iGyroAxis = g_xPS.x.bGyroAxis;

    m_pMenu->clear();

    QAction* pGyroAxisActions[GYRO_AXIS_COUNT] = { 0 };
    for(int i = 0; i < GYRO_AXIS_COUNT; i ++)
    {
        pGyroAxisActions[i] = m_pMenu->addAction(strGyroAxisValue[i]);
        pGyroAxisActions[i]->setCheckable(true);
        pGyroAxisActions[i]->setChecked((i + 1) == iGyroAxis);
    }

    QPoint xViewPos(130 * 2, m_vMenuItems[ITEM_GYRO_AXIS_ID]->pos().y() + 20 * 2);
    xViewPos = GetItemView()->mapFromScene(xViewPos);
    xViewPos = GetItemView()->mapToGlobal(xViewPos);

    if(xViewPos.y() > 650)
        xViewPos.setY(650);

    QAction* pSelAction = m_pMenu->exec(xViewPos);

    for(int i = 0; i < GYRO_AXIS_COUNT; i ++)
    {
        if(pGyroAxisActions[i] == pSelAction)
        {
            g_xPS.x.bGyroAxis = i + 1;
            UpdatePermanenceSettings();

            MainBackProc::MotorSetting(g_pUartTask);
        }
    }

    RetranslateUI();
}

void FactorySettingForm::ClickedOverCurrent()
{
    QWidget* pSurface = new QWidget;
    SliderSpin* pOverCurrentSpin = new SliderSpin;
    QHBoxLayout* pLayout = new QHBoxLayout;
    QString strSheets;
    pLayout->setMargin(0);
    pLayout->addWidget(pOverCurrentSpin);
    pSurface->setFixedSize(188 * LCD_RATE, pOverCurrentSpin->height());
    pSurface->setLayout(pLayout);
    strSheets.sprintf("QWidget { background-color: rgb(%d, %d, %d); }", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue());
    pSurface->setStyleSheet(strSheets);

    QStringList vRangeList;
    for(int i = MOTO_OVER_CURRENT_MIN; i <= MOTO_OVER_CURRENT_MAX; i += MOTO_OVER_CURRENT_STEP)
        vRangeList.append(QString::number(i));

    pOverCurrentSpin->SetSpinRange(vRangeList);
    pOverCurrentSpin->SetRepeat(true);
    pOverCurrentSpin->SetSelect(g_xPS.x.bOverCurrent - (MOTO_OVER_CURRENT_MIN / MOTO_OVER_CURRENT_STEP));

    AlertDlg* pDlg = new AlertDlg(m_pParentView, this);
    pDlg->SetTitle(StringTable::Str_Overcurrent + " (mA)");
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_OK);
    pDlg->SetCancelButton(true, StringTable::Str_Cancel);
    pDlg->AddWidget(pSurface);
    pDlg->setFixedSize(210 * LCD_RATE, 265 * LCD_RATE);
    int nRet = pDlg->OnExec();
    if(nRet == QDialog::Accepted)
    {
        g_xPS.x.bOverCurrent = pOverCurrentSpin->GetSelectedIndex() + (MOTO_OVER_CURRENT_MIN / MOTO_OVER_CURRENT_STEP);
        UpdatePermanenceSettings();

        MainBackProc::MotorSetting(g_pUartTask);
    }

    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    delete pDlgScene;

    RetranslateUI();
}

void FactorySettingForm::FingerprintClick()
{    
    g_xPS.x.bFingerprint = 1 - g_xPS.x.bFingerprint;
    UpdatePermanenceSettings();

    RetranslateUI();
}

void FactorySettingForm::BattLowClick()
{
    QWidget* pSurface = new QWidget;
    SliderSpin* pBattLowSpin = new SliderSpin;
    QHBoxLayout* pLayout = new QHBoxLayout;
    QString strSheets;
    pLayout->setMargin(0);
    pLayout->addWidget(pBattLowSpin);
    pSurface->setFixedSize(188 * LCD_RATE, pBattLowSpin->height());
    pSurface->setLayout(pLayout);
    strSheets.sprintf("QWidget { background-color: rgb(%d, %d, %d); }", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue());
    pSurface->setStyleSheet(strSheets);


    QStringList vRangeList;
    for(int i = BATT_SETTINGS_MIN; i <= BATT_SETTINGS_MAX; i += BATT_SETTINGS_STEP)
        vRangeList.append(QString::number(i));

    pBattLowSpin->SetSpinRange(vRangeList);
    pBattLowSpin->SetRepeat(true);
    pBattLowSpin->SetSelect((g_xBSS.x.iBattLow - BATT_SETTINGS_MIN) / BATT_SETTINGS_STEP);

    AlertDlg* pDlg = new AlertDlg(m_pParentView, this);
    pDlg->SetTitle(StringTable::Str_Batt_Low);
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_OK);
    pDlg->SetCancelButton(true, StringTable::Str_Cancel);
    pDlg->AddWidget(pSurface);
    pDlg->setFixedSize(210 * LCD_RATE, 265 * LCD_RATE);
    int nRet = pDlg->OnExec();
    if(nRet == QDialog::Accepted)
    {
        g_xBSS.x.iBattLow = (pBattLowSpin->GetSelectedIndex() * BATT_SETTINGS_STEP) + BATT_SETTINGS_MIN;
        UpdateBattSettings();
    }

    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    pDlgScene->clear();
    delete pDlgScene;

    RetranslateUI();
}

void FactorySettingForm::BattStepClick()
{
    QWidget* pSurface = new QWidget;
    SliderSpin* pBattLowSpin = new SliderSpin;
    QHBoxLayout* pLayout = new QHBoxLayout;
    QString strSheets;
    pLayout->setMargin(0);
    pLayout->addWidget(pBattLowSpin);
    pSurface->setFixedSize(188 * LCD_RATE, pBattLowSpin->height());
    pSurface->setLayout(pLayout);
    strSheets.sprintf("QWidget { background-color: rgb(%d, %d, %d); }", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue());
    pSurface->setStyleSheet(strSheets);

    QStringList vRangeList;
    for(int i = BATT_LEVEL_MIN; i <= BATT_LEVEL_MAX; i += BATT_LEVEL_STEP_STEP)
        vRangeList.append(QString::number(i));

    pBattLowSpin->SetSpinRange(vRangeList);
    pBattLowSpin->SetRepeat(true);
    pBattLowSpin->SetSelect((g_xBSS.x.iBattStep - BATT_LEVEL_MIN) / BATT_LEVEL_STEP_STEP);

    AlertDlg* pDlg = new AlertDlg(m_pParentView, this);
    pDlg->SetTitle(StringTable::Str_Batt_Step);
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_OK);
    pDlg->SetCancelButton(true, StringTable::Str_Cancel);
    pDlg->AddWidget(pSurface);
    pDlg->setFixedSize(210 * LCD_RATE, 265 * LCD_RATE);
    int nRet = pDlg->OnExec();

    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    pDlgScene->clear();
    delete pDlgScene;

    RetranslateUI();
}

void FactorySettingForm::BattNewClick()
{
    QWidget* pSurface = new QWidget;
    SliderSpin* pBattLowSpin = new SliderSpin;
    QHBoxLayout* pLayout = new QHBoxLayout;
    QString strSheets;
    pLayout->setMargin(0);
    pLayout->addWidget(pBattLowSpin);
    pSurface->setFixedSize(188 * LCD_RATE, pBattLowSpin->height());
    pSurface->setLayout(pLayout);
    strSheets.sprintf("QWidget { background-color: rgb(%d, %d, %d); }", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue());
    pSurface->setStyleSheet(strSheets);

    QStringList vRangeList;
    for(int i = BATT_SETTINGS_MIN; i <= BATT_SETTINGS_MAX; i += BATT_SETTINGS_STEP)
        vRangeList.append(QString::number(i));

    pBattLowSpin->SetSpinRange(vRangeList);
    pBattLowSpin->SetSelect((g_xBSS.x.iBattNew - BATT_SETTINGS_MIN) / BATT_SETTINGS_STEP);

    AlertDlg* pDlg = new AlertDlg(m_pParentView, this);
    pDlg->SetTitle(StringTable::Str_Batt_New);
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_OK);
    pDlg->SetCancelButton(true, StringTable::Str_Cancel);
    pDlg->AddWidget(pSurface);
    pDlg->setFixedSize(210 * LCD_RATE, 265 * LCD_RATE);
    int nRet = pDlg->OnExec();
    if(nRet == QDialog::Accepted)
    {
        g_xBSS.x.iBattNew = (pBattLowSpin->GetSelectedIndex() * BATT_SETTINGS_STEP) + BATT_SETTINGS_MIN;
        UpdateBattSettings();
    }

    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    pDlgScene->clear();
    delete pDlgScene;

    RetranslateUI();
}

void FactorySettingForm::BattPowerDownClick()
{
    QWidget* pSurface = new QWidget;
    SliderSpin* pBattLowSpin = new SliderSpin;
    QHBoxLayout* pLayout = new QHBoxLayout;
    QString strSheets;
    pLayout->setMargin(0);
    pLayout->addWidget(pBattLowSpin);
    pSurface->setFixedSize(188 * LCD_RATE, pBattLowSpin->height());
    pSurface->setLayout(pLayout);
    strSheets.sprintf("QWidget { background-color: rgb(%d, %d, %d); }", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue());
    pSurface->setStyleSheet(strSheets);

    QStringList vRangeList;
    for(int i = BATT_SETTINGS_MIN; i <= BATT_SETTINGS_MAX; i += BATT_SETTINGS_STEP)
        vRangeList.append(QString::number(i));

    pBattLowSpin->SetSpinRange(vRangeList);
    pBattLowSpin->SetRepeat(true);
    pBattLowSpin->SetSelect((g_xBSS.x.iBattPowerDown - BATT_SETTINGS_MIN) / BATT_SETTINGS_STEP);

    AlertDlg* pDlg = new AlertDlg(m_pParentView, this);
    pDlg->SetTitle(StringTable::Str_Batt_Powerdown);
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_OK);
    pDlg->SetCancelButton(true, StringTable::Str_Cancel);
    pDlg->AddWidget(pSurface);
    pDlg->setFixedSize(210 * LCD_RATE, 265 * LCD_RATE);
    int nRet = pDlg->OnExec();
    if(nRet == QDialog::Accepted)
    {
        g_xBSS.x.iBattPowerDown = (pBattLowSpin->GetSelectedIndex() * BATT_SETTINGS_STEP) + BATT_SETTINGS_MIN;
        UpdateBattSettings();
    }

    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    pDlgScene->clear();
    delete pDlgScene;

    RetranslateUI();
}

void FactorySettingForm::BattMenuClick()
{
    QWidget* pSurface = new QWidget;
    SliderSpin* pBattLowSpin = new SliderSpin;
    QHBoxLayout* pLayout = new QHBoxLayout;
    QString strSheets;
    pLayout->setMargin(0);
    pLayout->addWidget(pBattLowSpin);
    pSurface->setFixedSize(188 * LCD_RATE, pBattLowSpin->height());
    pSurface->setLayout(pLayout);
    strSheets.sprintf("QWidget { background-color: rgb(%d, %d, %d); }", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue());
    pSurface->setStyleSheet(strSheets);

    QStringList vRangeList;
    for(int i = BATT_SETTINGS_MIN; i <= BATT_SETTINGS_MAX; i += BATT_SETTINGS_STEP)
        vRangeList.append(QString::number(i));

    pBattLowSpin->SetSpinRange(vRangeList);
    pBattLowSpin->SetRepeat(true);
    pBattLowSpin->SetSelect((g_xBSS.x.iBattMenu - BATT_SETTINGS_MIN) / BATT_SETTINGS_STEP);

    AlertDlg* pDlg = new AlertDlg(m_pParentView, this);
    pDlg->SetTitle(StringTable::Str_Batt_Menu, 35);
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_OK);
    pDlg->SetCancelButton(true, StringTable::Str_Cancel);
    pDlg->AddWidget(pSurface);
    pDlg->setFixedSize(210 * LCD_RATE, 265 * LCD_RATE);
    int nRet = pDlg->OnExec();
    if(nRet == QDialog::Accepted)
    {
        g_xBSS.x.iBattMenu = (pBattLowSpin->GetSelectedIndex() * BATT_SETTINGS_STEP) + BATT_SETTINGS_MIN;
        UpdateBattSettings();
    }

    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    pDlgScene->clear();
    delete pDlgScene;

    RetranslateUI();
}

void FactorySettingForm::BattUpdateClick()
{
    QWidget* pSurface = new QWidget;
    SliderSpin* pBattLowSpin = new SliderSpin;
    QHBoxLayout* pLayout = new QHBoxLayout;
    QString strSheets;
    pLayout->setMargin(0);
    pLayout->addWidget(pBattLowSpin);
    pSurface->setFixedSize(188 * LCD_RATE, pBattLowSpin->height());
    pSurface->setLayout(pLayout);
    strSheets.sprintf("QWidget { background-color: rgb(%d, %d, %d); }", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue());
    pSurface->setStyleSheet(strSheets);

    QStringList vRangeList;

    pBattLowSpin->SetSpinRange(vRangeList);
    pBattLowSpin->SetRepeat(true);
    pBattLowSpin->SetSelect((g_xBSS.x.iBattUpdate - BATT_UPDATE_MIN) / BATT_UPDATE_STEP);

    AlertDlg* pDlg = new AlertDlg(m_pParentView, this);
    pDlg->SetTitle(StringTable::Str_Batt_Update);
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_OK);
    pDlg->SetCancelButton(true, StringTable::Str_Cancel);
    pDlg->AddWidget(pSurface);
    pDlg->setFixedSize(210 * LCD_RATE, 265 * LCD_RATE);
    int nRet = pDlg->OnExec();
    if(nRet == QDialog::Accepted)
    {
        g_xBSS.x.iBattUpdate = (pBattLowSpin->GetSelectedIndex() * BATT_UPDATE_STEP) + BATT_UPDATE_MIN;
        UpdateBattSettings();
    }

    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    pDlgScene->clear();
    delete pDlgScene;

    RetranslateUI();
}

void FactorySettingForm::BattSoundOffClick()
{
    QWidget* pSurface = new QWidget;
    SliderSpin* pBattLowSpin = new SliderSpin;
    QHBoxLayout* pLayout = new QHBoxLayout;
    QString strSheets;
    pLayout->setMargin(0);
    pLayout->addWidget(pBattLowSpin);
    pSurface->setFixedSize(188 * LCD_RATE, pBattLowSpin->height());
    strSheets.sprintf("QWidget { background-color: rgb(%d, %d, %d); }", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue());
    pSurface->setStyleSheet(strSheets);

    QStringList vRangeList;
    for(int i = BATT_SETTINGS_MIN; i <= BATT_SETTINGS_MAX; i += BATT_SETTINGS_STEP)
        vRangeList.append(QString::number(i));

    pBattLowSpin->SetSpinRange(vRangeList);
    pBattLowSpin->SetRepeat(true);
    pBattLowSpin->SetSelect((g_xBSS.x.iBattSoundOff - BATT_SETTINGS_MIN) / BATT_SETTINGS_STEP);

    AlertDlg* pDlg = new AlertDlg(m_pParentView, this);
    pDlg->SetTitle(StringTable::Str_Batt_SoundOff);
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_OK);
    pDlg->SetCancelButton(true, StringTable::Str_Cancel);
    pDlg->AddWidget(pSurface);
    pDlg->setFixedSize(210 * LCD_RATE, 265 * LCD_RATE);
    int nRet = pDlg->OnExec();
    if(nRet == QDialog::Accepted)
    {
        g_xBSS.x.iBattSoundOff = (pBattLowSpin->GetSelectedIndex() * BATT_SETTINGS_STEP) + BATT_SETTINGS_MIN;
        UpdateBattSettings();
    }

    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    pDlgScene->clear();
    delete pDlgScene;

    RetranslateUI();
}

void FactorySettingForm::HiddenCodeClick()
{
    QRCodeReaderForm* pQRCodeReaderForm = new QRCodeReaderForm(m_pParentView, this);
    connect(pQRCodeReaderForm, SIGNAL(SigQRCodeRead(QString)), this, SLOT(QRCodeReadFinished(QString)));
    connect(pQRCodeReaderForm, SIGNAL(SigQRCodeRead(int)), this, SLOT(QRCodeReadFinished(int)));
    pQRCodeReaderForm->StartRead(SETTING_TIMEOUT);
}

void FactorySettingForm::HomeAutoMationClick()
{
#if 0
    m_pMenu->clear();

    QAction* pZigbeeAction[10] = { 0 };
    pZigbeeAction[GetZigbeeIdx(ZIGBEE_DISABLE)] = m_pMenu->addAction(GetZigbeeStrByMode(ZIGBEE_DISABLE));
    pZigbeeAction[GetZigbeeIdx(ZIGBEE_OUR)] = m_pMenu->addAction(GetZigbeeStrByMode(ZIGBEE_OUR));
    pZigbeeAction[GetZigbeeIdx(WIFI_YINGHUA_JIWEI)] = m_pMenu->addAction(GetZigbeeStrByMode(WIFI_YINGHUA_JIWEI));
    pZigbeeAction[GetZigbeeIdx(WIFI_YINGHUA_SIGE)] = m_pMenu->addAction(GetZigbeeStrByMode(WIFI_YINGHUA_SIGE));
    pZigbeeAction[GetZigbeeIdx(WIFI_GESANG_SIGE)] = m_pMenu->addAction(GetZigbeeStrByMode(WIFI_GESANG_SIGE));

    for(int i = 0; i < 5; i ++)
    {
        pZigbeeAction[i]->setCheckable(true);
        pZigbeeAction[i]->setChecked(GetZigbeeMode(i) == g_xPS.x.bZigbeeMode);
    }

    QPoint xViewPos(130 * LCD_RATE, m_vMenuItems[ITEM_HOME_AUTOMATION_ID]->pos().y() + 20 * LCD_RATE);
    xViewPos = GetItemView()->mapFromScene(xViewPos);
    xViewPos = GetItemView()->mapToGlobal(xViewPos);

    if(xViewPos.y() > 650)
        xViewPos.setY(650);

    QAction* pSelAction = m_pMenu->exec(xViewPos);
    for(int i = 0; i < 5; i ++)
    {
        if(pZigbeeAction[i] == pSelAction)
        {
            g_xPS.x.bZigbeeMode = GetZigbeeMode(i);
            UpdatePermanenceSettings();

            RetranslateUI();
        }
    }
#else
    QWidget* pSurface = new QWidget;
    SliderSpin* pZigbeeSpin = new SliderSpin;
    pZigbeeSpin->setFixedWidth(160 * LCD_RATE);
    QHBoxLayout* pLayout = new QHBoxLayout;
    QString strSheets;
    pLayout->setMargin(0);
    pLayout->addWidget(pZigbeeSpin);
    pSurface->setFixedSize(188 * LCD_RATE, pZigbeeSpin->height());
    pSurface->setLayout(pLayout);
    strSheets.sprintf("QWidget { background-color: rgb(%d, %d, %d); }", g_UITheme->itemNormalBgColor.red(), g_UITheme->itemNormalBgColor.green(), g_UITheme->itemNormalBgColor.blue());
    pSurface->setStyleSheet(strSheets);

    QStringList vRangeList;
    for(int i = 0; i < ZIGBEE_MAX_NUM; i ++)
        vRangeList.append(GetZigbeeStrByIdx(i));

    pZigbeeSpin->SetSpinRange(vRangeList);
    pZigbeeSpin->SetRepeat(true);
    pZigbeeSpin->SetSelect(GetZigbeeIdx(g_xPS.x.bZigbeeMode));

    AlertDlg* pDlg = new AlertDlg(m_pParentView, this);
    pDlg->SetTitle(StringTable::Str_Home_Automation);
    pDlg->SetButtonGroup(true);
    pDlg->SetOkButton(true, StringTable::Str_OK);
    pDlg->SetCancelButton(true, StringTable::Str_Cancel);
    pDlg->AddWidget(pSurface);
    pDlg->setFixedSize(210 * LCD_RATE, 265 * LCD_RATE);

    if(nRet == QDialog::Accepted)
    {
        g_xPS.x.bZigbeeMode = GetZigbeeMode(pZigbeeSpin->GetSelectedIndex());
        g_xPS.x.bWifiUsingSTM = 0;

        if(g_xPS.x.bZigbeeMode == WIFI_OUR || g_xPS.x.bZigbeeMode == WIFI_LANCENS)
            g_xPS.x.bWifiUsingSTM = 1;

        if((g_xPS.x.bZigbeeMode & WIFI_BASE) &&
                (g_xPS.x.bZigbeeMode != WIFI_YINGHUA_JIWEI_V2
                 && g_xPS.x.bZigbeeMode != WIFI_XIONGMAI
                 && g_xPS.x.bZigbeeMode != WIFI_OUR
                 && g_xPS.x.bZigbeeMode != WIFI_LANCENS))
            g_xPS.x.bBaudRateMode = Baud_Rate_9600;
        else
            g_xPS.x.bBaudRateMode = Baud_Rate_115200;

        UpdatePermanenceSettings();

        g_xCS.x.bZigbee = 0;
        UpdateCommonSettings();

        MainBackProc::MotorSetting(g_pUartTask);
    }

    QGraphicsScene* pDlgScene = pDlg->GetSurfaceScene();
    pDlgScene->clear();
    delete pDlgScene;

    RetranslateUI();
#endif
}

void FactorySettingForm::HumanSensorClick()
{
    m_pMenu->clear();

    QString strHumanSensorType[] = {"NIR", "PIR"};
    QAction* pHumanSensorAction[2] = { 0 };
    for(int i = 0; i < 2; i ++)
    {
        pHumanSensorAction[i] = m_pMenu->addAction(strHumanSensorType[i]);
        pHumanSensorAction[i]->setCheckable(true);
        pHumanSensorAction[i]->setChecked(i == g_xPS.x.bHumanSensorType);
    }

    QPoint xViewPos(130, m_vMenuItems[ITEM_HUMAN_SENSOR_TYPE_ID]->pos().y() + 20);
    xViewPos = GetItemView()->mapFromScene(xViewPos);
    xViewPos = GetItemView()->mapToGlobal(xViewPos);

    if(xViewPos.y() > 250)
        xViewPos.setY(250);

    QAction* pSelAction = m_pMenu->exec(xViewPos);
    for(int i = 0; i < 2; i ++)
    {
        if(pHumanSensorAction[i] == pSelAction)
        {
            g_xPS.x.bHumanSensorType = i;
            UpdatePermanenceSettings();
            RetranslateUI();
        }
    }
}


#if (TEST_DUTY_CYCLE == 1)
void FactorySettingForm::DutyCycleValuesClick()
{
    unsigned char abDutyCycles[DUTY_CYCLE_TABLE_SIZE] = { 0 };		  	// 10 duty cycles.
    MainSTM_GetDutyCycleTable(abDutyCycles, DUTY_CYCLE_TABLE_SIZE);

    if (abDutyCycles[0] > 0 && abDutyCycles[1] > abDutyCycles[0])
    {
        QString szShowMsg = StringTable::Str_HumanSensor_DutyCycleTable + "<br/>" + "<br/>";
        for (int i = 0 ; i < DUTY_CYCLE_TABLE_SIZE ; i ++)
        {
            szShowMsg += QString::number(abDutyCycles[i]) + " ";
        }
        AlertDlg::WarningOk(m_pParentView, this, StringTable::Str_Information, szShowMsg, 1);
    }
}
#endif //TEST_DUTY_CYCLE

#ifdef ITEM_BAUD_RATE_ID
void FactorySettingForm::BaudrateClick()
{
    m_pMenu->clear();

    QString strBaudrates[] = {"115200", "9600"};
    QAction* pBaudrateActions[2] = { 0 };
    for(int i = 0; i < 2; i ++)
    {
        pBaudrateActions[i] = m_pMenu->addAction(strBaudrates[i]);
        pBaudrateActions[i]->setCheckable(true);
        pBaudrateActions[i]->setChecked(i == g_xPS.x.bBaudRateMode);
    }

    QPoint xViewPos(130 * LCD_RATE, m_vMenuItems[ITEM_BAUD_RATE_ID]->pos().y() + 20 * LCD_RATE);
    xViewPos = GetItemView()->mapFromScene(xViewPos);
    xViewPos = GetItemView()->mapToGlobal(xViewPos);

#if (AUTO_TEST != 1)
    if(xViewPos.y() > 650)
        xViewPos.setY(650);

    QAction* pSelAction = m_pMenu->exec(xViewPos);
#else
    int iIdx = rand() % E_MOTO_POLARITY_END;
    QAction* pSelAction = pMotoPolarityActions[iIdx];
#endif
    for(int i = 0; i < 2; i ++)
    {
        if(pBaudrateActions[i] == pSelAction)
        {
            g_xPS.x.bBaudRateMode = i;
            UpdatePermanenceSettings();

            RetranslateUI();
        }
    }
}
#endif //ITEM_BAUD_RATE_ID

void FactorySettingForm::QRCodeReadFinished(QString strQRCode)
{
    FormBase::OnResume();

    if(strQRCode.isEmpty())
        return;

    int iLen = strQRCode.toUtf8().length();
    if(iLen > WORD_SIZE)
        iLen = WORD_SIZE;

    char szHiddenCode[256] = { 0 };
    strncpy(szHiddenCode, strQRCode.toUtf8().data(), iLen);

    int isOk = 1;
    for(int i = 0; i < strlen(szHiddenCode); i ++)
    {
        if(szHiddenCode[i] & 0x80)
            isOk = 0;
    }

    if(isOk)
    {
        M24C64_SetHidenCode((unsigned char*)szHiddenCode);
#if USING_BUZZER
        MainSTM_Command(MAIN_STM_BUZZER_SUCCESS);
#else
        PlayCompleteSoundAlways();
#endif
    }
    else
    {
#if USING_BUZZER
        MainSTM_Command(MAIN_STM_BUZZER_FAILED);
#else
        PlayError5Sound();
#endif
    }

    RetranslateUI();
}

void FactorySettingForm::QRCodeReadFinished(int iCamError)
{
    FormBase::OnResume();

    if(iCamError)
    {
        g_xSS.iNoSoundPlayFlag = 1;
        AlertDlg::WarningOk(m_pParentView, this, StringTable::Str_Warning, StringTable::Str_Camera_Error);
    }

    AlertDlg::Locked = 0;
}


void FactorySettingForm::RetranslateUI()
{
    SetTitle(StringTable::Str_Factory_Setting);

    QString strGyroAxisValue[] =
    {
        StringTable::Str_Horizontal,
        StringTable::Str_Vertical
    };

    QString strLogoTitle;
    strLogoTitle.sprintf("Logo %d", g_xSS.iCurLogo);

    m_vMenuItems[ITEM_LOGO_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Logo);
    m_vMenuItems[ITEM_LOGO_ID]->setData(KEY_SECONDARY_TEXT, strLogoTitle);

#if 0
    QString strMotoType[] = {StringTable::Str_Two, StringTable::Str_One};
    if(m_vMenuItems[ITEM_MOTO_TYPE_ID])
    {
        m_vMenuItems[ITEM_MOTO_TYPE_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Moto_Type);
        m_vMenuItems[ITEM_MOTO_TYPE_ID]->setData(KEY_SECONDARY_TEXT, strMotoType[g_xFS.bMotoType]);
    }

#endif
    if(m_vMenuItems[ITEM_MOTO_POLARITY_ID])
    {
        QString strMotoPolarity[] = {"+/-", "-/+"};
        m_vMenuItems[ITEM_MOTO_POLARITY_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Moto_Polarity);
        m_vMenuItems[ITEM_MOTO_POLARITY_ID]->setData(KEY_SECONDARY_TEXT, strMotoPolarity[g_xPS.x.bSemiMotorPolarity]);
    }

#if (LOCK_MODE == LM_SEMI_AUTO && 0)
    if(m_vMenuItems[ITEM_MOTO_TIME_ID])
    {
        int iMotoTime = g_xPS.x.bSemiMotorOpenTime * 10 + 100;
        if(g_xPS.x.bSemiMotorType == E_ONE)
            iMotoTime = 1400;

        m_vMenuItems[ITEM_MOTO_TIME_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Moto_Time);
        m_vMenuItems[ITEM_MOTO_TIME_ID]->setData(KEY_SECONDARY_TEXT, QString::number(iMotoTime) + "ms");
    }
#endif // LOCK_MODE == LM_SEMI_AUTO

#if 0
    if(m_vMenuItems[ITEM_KEEP_OPEN_ID])
    {
        m_vMenuItems[ITEM_KEEP_OPEN_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Keep_Open);
        m_vMenuItems[ITEM_KEEP_OPEN_ID]->setData(KEY_CHECKED, g_xFS.bKeepOpen);
    }

    if(m_vMenuItems[ITEM_PRESENTATION_ID])
    {
        m_vMenuItems[ITEM_PRESENTATION_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Presentation);
        m_vMenuItems[ITEM_PRESENTATION_ID]->setData(KEY_CHECKED, g_xFS.bPresentation);
    }

    if(m_vMenuItems[ITEM_BATT_TEST_ID])
    {
        m_vMenuItems[ITEM_BATT_TEST_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Batt_Test);
        m_vMenuItems[ITEM_BATT_TEST_ID]->setData(KEY_CHECKED, g_xFS.bBattTest);
    }
#endif

    if(m_vMenuItems[ITEM_HIDDEN_CODE_ID])
    {
        char szHiddenCode[256] = { 0 };
        M24C64_GetHidenCode((unsigned char*)szHiddenCode);

        if(!strlen(szHiddenCode))
            strcpy(szHiddenCode, StringTable::Str_None.toUtf8().data());

        m_vMenuItems[ITEM_HIDDEN_CODE_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Hidden_Code);
        m_vMenuItems[ITEM_HIDDEN_CODE_ID]->setData(KEY_SECONDARY_TEXT, QString::fromUtf8(szHiddenCode));
    }

    if(m_vMenuItems[ITEM_HOME_AUTOMATION_ID])
    {
        m_vMenuItems[ITEM_HOME_AUTOMATION_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Home_Automation);
        m_vMenuItems[ITEM_HOME_AUTOMATION_ID]->setData(KEY_SECONDARY_TEXT, GetZigbeeStrByMode(g_xPS.x.bZigbeeMode));
    }

    if(m_vMenuItems[ITEM_HUMAN_SENSOR_TYPE_ID])
    {
        m_vMenuItems[ITEM_HUMAN_SENSOR_TYPE_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Human_Sensor);

        if(g_xPS.x.bHumanSensorType == HUMAN_NIR)
            m_vMenuItems[ITEM_HUMAN_SENSOR_TYPE_ID]->setData(KEY_SECONDARY_TEXT, tr("NIR"));
        else if(g_xPS.x.bHumanSensorType == HUMAN_PIR)
            m_vMenuItems[ITEM_HUMAN_SENSOR_TYPE_ID]->setData(KEY_SECONDARY_TEXT, tr("PIR"));
    }

#ifdef ITEM_BAUD_RATE_ID
    if(m_vMenuItems[ITEM_BAUD_RATE_ID])
    {        
        if(g_xPS.x.bZigbeeMode != ZIGBEE_DISABLE)
        {
            if(g_xPS.x.bZigbeeMode == ZIGBEE_OUR)
                g_xPS.x.bBaudRateMode = 0;
            else if(g_xPS.x.bZigbeeMode & WIFI_BASE)
                g_xPS.x.bBaudRateMode = 1;
        }

        m_vMenuItems[ITEM_BAUD_RATE_ID]->setEnabled(g_xPS.x.bZigbeeMode == ZIGBEE_DISABLE);
        m_vMenuItems[ITEM_BAUD_RATE_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Baud_Rate);
        m_vMenuItems[ITEM_BAUD_RATE_ID]->setData(KEY_SECONDARY_TEXT, g_xPS.x.bBaudRateMode ? QString::number(9600) : QString::number(115200));
    }
#endif //ITEM_BAUD_RATE_ID

#if 0
    if(m_vMenuItems[ITEM_FINGER_PRINT_ID])
    {
        m_vMenuItems[ITEM_FINGER_PRINT_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Fingerprint);
        m_vMenuItems[ITEM_FINGER_PRINT_ID]->setData(KEY_CHECKED, g_xPS.x.bFingerprint);
    }
#endif

    if(m_vMenuItems[ITEM_OVER_CURRENT_ID])
    {
        m_vMenuItems[ITEM_OVER_CURRENT_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Overcurrent);
        m_vMenuItems[ITEM_OVER_CURRENT_ID]->setData(KEY_SECONDARY_TEXT, QString::number(g_xPS.x.bOverCurrent * MOTO_OVER_CURRENT_STEP) + "mA");
    }

    if(m_vMenuItems[ITEM_BATT_LOW_ID])
    {
        m_vMenuItems[ITEM_BATT_LOW_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Batt_Low);
        m_vMenuItems[ITEM_BATT_LOW_ID]->setData(KEY_SECONDARY_TEXT, QString::number(g_xBSS.x.iBattLow) + "mV");
    }

    if(m_vMenuItems[ITEM_BATT_STEP_ID])
    {
        m_vMenuItems[ITEM_BATT_STEP_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Batt_Step);
        m_vMenuItems[ITEM_BATT_STEP_ID]->setData(KEY_SECONDARY_TEXT, QString::number(g_xBSS.x.iBattStep) + "mV");
    }

    if(m_vMenuItems[ITEM_BATT_NEW_ID])
    {
        m_vMenuItems[ITEM_BATT_NEW_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Batt_New);
        m_vMenuItems[ITEM_BATT_NEW_ID]->setData(KEY_SECONDARY_TEXT, QString::number(g_xBSS.x.iBattNew) + "mV");
    }

    if(m_vMenuItems[ITEM_BATT_POWER_DOWN_ID])
    {
        m_vMenuItems[ITEM_BATT_POWER_DOWN_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Batt_Powerdown);
        m_vMenuItems[ITEM_BATT_POWER_DOWN_ID]->setData(KEY_SECONDARY_TEXT, QString::number(g_xBSS.x.iBattPowerDown) + "mV");
    }

    if(m_vMenuItems[ITEM_BATT_MENU_ID])
    {
        m_vMenuItems[ITEM_BATT_MENU_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Batt_Menu);
        m_vMenuItems[ITEM_BATT_MENU_ID]->setData(KEY_SECONDARY_TEXT, QString::number(g_xBSS.x.iBattMenu) + "mV");
    }

    if(m_vMenuItems[ITEM_BATT_UPDATE_ID])
    {
        m_vMenuItems[ITEM_BATT_UPDATE_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Batt_Update);
        m_vMenuItems[ITEM_BATT_UPDATE_ID]->setData(KEY_SECONDARY_TEXT, QString::number(g_xBSS.x.iBattUpdate) + "mV");
    }

    if(m_vMenuItems[ITEM_BATT_SOUND_OFF_ID])
    {
        m_vMenuItems[ITEM_BATT_SOUND_OFF_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Batt_SoundOff);
        m_vMenuItems[ITEM_BATT_SOUND_OFF_ID]->setData(KEY_SECONDARY_TEXT, QString::number(g_xBSS.x.iBattSoundOff) + "mV");
    }

    if(m_vMenuItems[ITEM_SHOW_BATT_ID])
    {
        m_vMenuItems[ITEM_SHOW_BATT_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Show_Batt);
        m_vMenuItems[ITEM_SHOW_BATT_ID]->setData(KEY_CHECKED, g_xFS.bShowBatt);
    }

#if (LOCK_MODE == LM_AUTO)
    if(m_vMenuItems[ITEM_MOTOR_ID])
    {
        m_vMenuItems[ITEM_MOTOR_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Motor);
        m_vMenuItems[ITEM_MOTOR_ID]->setData(KEY_CHECKED, 1 - g_xFS.iMotorControl);
    }
#endif

    if(m_vMenuItems[ITEM_GYRO_AXIS_ID])
    {
        int iGyroAxis = g_xPS.x.bGyroAxis;
        m_vMenuItems[ITEM_GYRO_AXIS_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_GyroAxis);
        m_vMenuItems[ITEM_GYRO_AXIS_ID]->setData(KEY_SECONDARY_TEXT, strGyroAxisValue[iGyroAxis - 1]);
    }

    if(m_vMenuItems[ITEM_INNER_VERSION_ID])
    {
        QString strVersion = QString::fromUtf8(DEVICE_FIRMWARE_VERSION_INNER);

        char szSubversion[256] = { 0 };
        int iRet = MainSTM_Command(MAIN_STM_INNER_VERSION, (unsigned char*)szSubversion);
        if(iRet == 1)
            strVersion += "\n" + QString::fromUtf8(szSubversion);

        m_vMenuItems[ITEM_INNER_VERSION_ID]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Version);
        m_vMenuItems[ITEM_INNER_VERSION_ID]->setData(KEY_SECONDARY_TEXT, strVersion);
    }

#if (TEST_DUTY_CYCLE == 1)
    if (m_vMenuItems[ITEM_DUTY_CYCLE_VALUES])
    {
        m_vMenuItems[ITEM_DUTY_CYCLE_VALUES]->setData(KEY_PRIMARY_TEXT, StringTable::Str_HumanSensor_DutyCycleTable);
        m_vMenuItems[ITEM_DUTY_CYCLE_VALUES]->setEnabled(g_xPS.x.bHumanSensorType == HUMAN_NIR);
    }
#endif //TEST_DUTY_CYCLE
}
