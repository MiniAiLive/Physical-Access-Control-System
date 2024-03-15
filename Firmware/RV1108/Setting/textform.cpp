#include "textform.h"
#include "uitheme.h"
#include "stringtable.h"
#include "menuitem.h"
#include "settings.h"
#include "drv_gpio.h"
#include "base.h"
#include "mainwindow.h"
#include "i2cbase.h"
#include "shared.h"
#include "rokthread.h"
#include "DBManager.h"
#include "mainbackproc.h"
#include "countbase.h"
#include "themedef.h"
#include "soundbase.h"

#include <QtGui>
#include <unistd.h>

enum
{
    ID_SHOW_BATT,
    ID_MOTO_CONTROL,
    ID_CUR_BATT,
    ID_VERSION,
    ID_INNER_VERSION,
    ID_MODEL,
    ID_SERIAL,
    ID_BOOTING_COUNT,
    ID_LOG_COUNT,
    ID_ERROR_1,
    ID_END
};

TextForm::TextForm(QGraphicsView *pView, FormBase* pParentForm) :
    ItemFormBase(pView, pParentForm)
{
    SetBGColor(g_UITheme->itemNormalBgColor);
}

void TextForm::OnStart(int iMode)
{
    m_iMode = iMode;

    if(m_iMode == PRESENTATION_START)
    {
        int iPosY = 0;
        MenuItem* pVersionItem = new MenuItem();
        pVersionItem->setPos(QPoint(0, iPosY));
        pVersionItem->SetBoundingRect(QRect(0, 0, INFO_ITEM_WIDTH, INFO_ITEM_HEIGHT));
        pVersionItem->setData(KEY_TYPE, TYPE_INFO_ITEM);
        m_pScene->addItem(pVersionItem);
        m_vMenuItems[ID_VERSION] = pVersionItem;
        iPosY += SETTING_ITEM_HEIGHT;

        MenuItem* pInnerVersionItem = new MenuItem();
        pInnerVersionItem->setPos(QPoint(0, iPosY));
        pInnerVersionItem->SetBoundingRect(QRect(0, 0, INFO_ITEM_WIDTH, INFO_ITEM_HEIGHT));
        pInnerVersionItem->setData(KEY_TYPE, TYPE_INFO_ITEM);
        m_pScene->addItem(pInnerVersionItem);
        m_vMenuItems[ID_INNER_VERSION] = pInnerVersionItem;
        iPosY += SETTING_ITEM_HEIGHT;

        MenuItem* pSerialItem = new MenuItem();
        pSerialItem->setPos(QPoint(0, iPosY));
        pSerialItem->SetBoundingRect(QRect(0, 0, INFO_ITEM_WIDTH, INFO_ITEM_HEIGHT));
        pSerialItem->setData(KEY_TYPE, TYPE_INFO_ITEM);
        m_pScene->addItem(pSerialItem);
        m_vMenuItems[ID_SERIAL] = pSerialItem;
        iPosY += INFO_ITEM_HEIGHT;

        MenuItem* pShowBattItem = new MenuItem();
        pShowBattItem->setPos(QPoint(0, iPosY));
        pShowBattItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
        pShowBattItem->setData(KEY_TYPE, TYPE_SWITCH_ITEM);
        m_pScene->addItem(pShowBattItem);
        m_vMenuItems[ID_SHOW_BATT] = pShowBattItem;
        connect(pShowBattItem, SIGNAL(clicked()), this, SLOT(ShowBattClick()));
        iPosY += SETTING_ITEM_HEIGHT;

#if (LOCK_MODE == LM_AUTO)
        MenuItem* pMotorItem = new MenuItem();
        pMotorItem->setPos(QPoint(0, iPosY));
        pMotorItem->SetBoundingRect(QRect(0, 0, SETTING_ITEM_WIDTH, SETTING_ITEM_HEIGHT));
        pMotorItem->setData(KEY_TYPE, TYPE_SWITCH_ITEM);
        m_pScene->addItem(pMotorItem);
        m_vMenuItems[ID_MOTO_CONTROL] = pMotorItem;
        connect(pMotorItem, SIGNAL(clicked()), this, SLOT(MotorClick()));
        iPosY += SETTING_ITEM_HEIGHT;
#endif /* LOCK_MODE == LM_AUTO */

        MenuItem* pCurBattItem = new MenuItem();
        pCurBattItem->setPos(QPoint(0, iPosY));
        pCurBattItem->SetBoundingRect(QRect(0, 0, INFO_ITEM_WIDTH, INFO_ITEM_HEIGHT));
        pCurBattItem->setData(KEY_TYPE, TYPE_INFO_ITEM);
        m_pScene->addItem(pCurBattItem);
        m_vMenuItems[ID_CUR_BATT] = pCurBattItem;
        iPosY += INFO_ITEM_HEIGHT;

        QTimer::singleShot(3000, this, SLOT(PresentTestStart()));
    }
    else if(m_iMode == PRESENTATION_STOP)
    {
        int iPosY = 0;
        MenuItem* pVersionItem = new MenuItem();
        pVersionItem->setPos(QPoint(0, iPosY));
        pVersionItem->SetBoundingRect(QRect(0, 0, INFO_ITEM_WIDTH, INFO_ITEM_HEIGHT));
        pVersionItem->setData(KEY_TYPE, TYPE_INFO_ITEM);
        m_pScene->addItem(pVersionItem);
        m_vMenuItems[ID_VERSION] = pVersionItem;
        iPosY += SETTING_ITEM_HEIGHT;

        MenuItem* pInnerVersionItem = new MenuItem();
        pInnerVersionItem->setPos(QPoint(0, iPosY));
        pInnerVersionItem->SetBoundingRect(QRect(0, 0, INFO_ITEM_WIDTH, INFO_ITEM_HEIGHT));
        pInnerVersionItem->setData(KEY_TYPE, TYPE_INFO_ITEM);
        m_pScene->addItem(pInnerVersionItem);
        m_vMenuItems[ID_INNER_VERSION] = pInnerVersionItem;
        iPosY += SETTING_ITEM_HEIGHT;

        MenuItem* pBootingCountItem = new MenuItem();
        pBootingCountItem->setPos(QPoint(0, iPosY));
        pBootingCountItem->SetBoundingRect(QRect(0, 0, INFO_ITEM_WIDTH, INFO_ITEM_HEIGHT));
        pBootingCountItem->setData(KEY_TYPE, TYPE_INFO_ITEM);
        m_pScene->addItem(pBootingCountItem);
        m_vMenuItems[ID_BOOTING_COUNT] = pBootingCountItem;
        iPosY += INFO_ITEM_HEIGHT;

        memset(m_aiErrrorArray, 0, sizeof(m_aiErrrorArray));
        FILE* fp = fopen("/db/error.bin", "r");
        if(fp)
        {
            int iCount = 0;
            char buffer[255];
            while(fgets(buffer, 255, (FILE*) fp) && (iCount++ < 10000000)) //limit 10000000
            {
                int nData1 = 0, nData2 = 0;
                char szDateTime[100] = { 0 };
                int nRet = sscanf(buffer, "%d-%d, %s\n", &nData1, &nData2, szDateTime);
                if (nRet < 0)
                    break;

                if (nData1 > ERROR_NONE && nData1 <= ERROR_I2C && nData2 >= 0 && nData2 < MAX_ERROR_TYPE)
                    m_aiErrrorArray[nData1 - 1][nData2] ++;
            }

            fclose (fp);
        }

#if 0
        if((g_xROKLog.i7Error_No_Secure + g_xROKLog.i7Error_Secure + g_xROKLog.i60Error) > 0)
        {
            if(g_xROKLog.i7Error_No_Secure > 0)
                m_aiErrrorArray[ERROR_I2C - 1][1] = g_xROKLog.i7Error_No_Secure;

            if(g_xROKLog.i60Error > 0)
                m_aiErrrorArray[ERROR_I2C - 1][2] = g_xROKLog.i60Error;

            if(g_xROKLog.i7Error_Secure > 0)
                m_aiErrrorArray[ERROR_I2C - 1][3] = g_xROKLog.i7Error_Secure;
        }
#endif

        for(int i = ERROR_NONE; i < ERROR_I2C; i ++)
        {
            int iErrorCount = 0;
            for(int j = 0; j < MAX_ERROR_TYPE; j ++)
                iErrorCount += m_aiErrrorArray[i][j];

            if(iErrorCount > 0)
            {
                MenuItem* pErrorItem = new MenuItem();
                pErrorItem->setPos(QPoint(0, iPosY));
                pErrorItem->SetBoundingRect(QRect(0, 0, INFO_ITEM_WIDTH, INFO_ITEM_HEIGHT));
                pErrorItem->setData(KEY_TYPE, TYPE_INFO_ITEM);
                m_pScene->addItem(pErrorItem);
                m_vMenuItems[ID_ERROR_1 + i] = pErrorItem;
                iPosY += INFO_ITEM_HEIGHT;
            }
        }
    }

    FormBase::OnStart();

    MainWindow* w = (MainWindow*)m_pParentView;
    ROKTHread* r = w->GetROK();
    r->SetKeyScanMode(this);
    r->SetTimeOut(60);
    r->InitTime();
}

void TextForm::PresentTestStart()
{
#if 0
    g_xPS.x.bPresentTest = 0x5A;

    UpdatePermanenceSettings();
    UpdateFactorySettings();

    g_xCS.x.bPresentation = 1;
    UpdateCommonSettings(&g_xCS);

    cp_rm_dir1("/db", false, NULL, "booting_count.ini", NULL);
    cp_rm_dir1("/db", false, NULL, "booting_count1.ini", NULL);

    ResetHeadInfos2();
    IncreaseSystemRunningCount();

#if USING_BUZZER
    MainSTM_Command(MAIN_STM_BUZZER_SUCCESS);
#else
    PlayCompleteSoundAlways();
#endif

    qApp->exit(RET_EXEC_LOCK);
#endif
}

void TextForm::ShowBattClick()
{
#if 0
    g_xFS.bShowBatt = 1 - g_xFS.bShowBatt;
    RetranslateUI();
#endif
}

void TextForm::MotorClick()
{
    g_xFS.iMotorControl = 1 - g_xFS.iMotorControl;
    RetranslateUI();
}

void TextForm::RetranslateUI()
{
    QString strMsg;

    SetTitle(StringTable::Str_Present_Test);        

    if(m_vMenuItems[ID_SHOW_BATT])
    {
        m_vMenuItems[ID_SHOW_BATT]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Show_Batt);
        m_vMenuItems[ID_SHOW_BATT]->setData(KEY_CHECKED, g_xFS.bShowBatt);
    }

    if(m_vMenuItems[ID_MOTO_CONTROL])
    {
        m_vMenuItems[ID_MOTO_CONTROL]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Motor);
        m_vMenuItems[ID_MOTO_CONTROL]->setData(KEY_CHECKED, 1 - g_xFS.iMotorControl);
    }

    if(m_vMenuItems[ID_VERSION])
    {
        QString strVersion = QString::fromUtf8(DEVICE_FIRMWARE_VERSION);
        char szSubversion[256] = { 0 };
        int iRet = MainSTM_Command(MAIN_STM_VERSION, (unsigned char*)szSubversion);
        if(iRet == 1)
            strVersion += "\n" + QString::fromUtf8(szSubversion);

        m_vMenuItems[ID_VERSION]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Firmware_Version);
        m_vMenuItems[ID_VERSION]->setData(KEY_SECONDARY_TEXT, strVersion);

    }

    if(m_vMenuItems[ID_INNER_VERSION])
    {
        QString strVersion = QString::fromUtf8(DEVICE_FIRMWARE_VERSION_INNER);

        char szSubversion[256] = { 0 };
        int iRet = MainSTM_Command(MAIN_STM_INNER_VERSION, (unsigned char*)szSubversion);
        if(iRet == 1)
            strVersion += "\n" + QString::fromUtf8(szSubversion);

        m_vMenuItems[ID_INNER_VERSION]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Version);
        m_vMenuItems[ID_INNER_VERSION]->setData(KEY_SECONDARY_TEXT, strVersion);
    }

    if(m_vMenuItems[ID_MODEL])
    {
        strMsg = DEVICE_MODEL_NUM;
        m_vMenuItems[ID_MODEL]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Model_Number);
        m_vMenuItems[ID_MODEL]->setData(KEY_SECONDARY_TEXT, strMsg);
    }

    if(m_vMenuItems[ID_SERIAL])
    {
        char szSerial[256] = { 0 };
        GetSerialNumber(szSerial);
        m_vMenuItems[ID_SERIAL]->setData(KEY_PRIMARY_TEXT, StringTable::Str_S_N);
        m_vMenuItems[ID_SERIAL]->setData(KEY_SECONDARY_TEXT, QString::fromUtf8(szSerial));
    }

    if(m_vMenuItems[ID_BOOTING_COUNT])
    {
        strMsg.sprintf("%d / %d", ReadSystemRunningCount(), g_xPS.x.iPresentCount);
        m_vMenuItems[ID_BOOTING_COUNT]->setData(KEY_PRIMARY_TEXT, StringTable::Str_System_Booting_Count);
        m_vMenuItems[ID_BOOTING_COUNT]->setData(KEY_SECONDARY_TEXT, strMsg);
    }

    if(m_vMenuItems[ID_LOG_COUNT])
    {
        int iAllLogCount = dbm_GetLogCount();
        strMsg.sprintf("%d / %d", iAllLogCount, N_MAX_LOG_NUM);

        m_vMenuItems[ID_LOG_COUNT]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Log);
        m_vMenuItems[ID_LOG_COUNT]->setData(KEY_SECONDARY_TEXT, strMsg);

    }

    if(m_vMenuItems[ID_CUR_BATT])
    {
        strMsg.sprintf("%d", GetVoltage());
        m_vMenuItems[ID_CUR_BATT]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Battery_Capacity);
        m_vMenuItems[ID_CUR_BATT]->setData(KEY_SECONDARY_TEXT, strMsg);
    }

    for(int i = ERROR_NONE; i < ERROR_I2C; i ++)
    {
        if(m_vMenuItems[ID_ERROR_1 + i])
        {
            char szMsg[256] = { 0 };
            char szError[10];
            for (int j = 0 ; j < MAX_ERROR_TYPE ; j ++)
            {
                if (m_aiErrrorArray[i][j] > 0)
                {
                    sprintf(szError, "%d(%d) ", j, m_aiErrrorArray[i][j]);
                    strcat(szMsg, szError);
                }
            }

            strMsg.sprintf("%d", GetVoltage());

            QString strTitle;
            strTitle.sprintf("%s%d", StringTable::Str_Error.toUtf8().data(), i + 1);

            m_vMenuItems[ID_ERROR_1 + i]->setData(KEY_PRIMARY_TEXT, strTitle);
            m_vMenuItems[ID_ERROR_1 + i]->setData(KEY_SECONDARY_TEXT, QString::fromUtf8(szMsg));
        }
    }
}

bool TextForm::event(QEvent* e)
{
    if(e->type() == EV_KEY_EVENT)
    {
        KeyEvent* pEvent = static_cast<KeyEvent*>(e);
        qDebug() << "KeyEvent" << pEvent->m_iKeyID << pEvent->m_iEvType;

        if(m_iMode == PRESENTATION_START)
        {
            if(pEvent->m_iKeyID == E_BTN_FUNC)
            {
                if(pEvent->m_iEvType == KeyEvent::EV_CLICKED)
                {
                    PresentTestStart();
                }
                else if(pEvent->m_iEvType == KeyEvent::EV_DOUBLE_CLICKED)
                {
                    qApp->exit(RET_EXEC_LOCK);
                }
            }
        }
        else if(m_iMode == PRESENTATION_STOP)
        {
            if(pEvent->m_iEvType == KeyEvent::EV_LONG_PRESSED)
            {
#if USING_BUZZER
                MainSTM_Command(MAIN_STM_BUZZER_SUCCESS);
#else
                PlayCompleteSoundAlways();
#endif


                MainWindow* w = (MainWindow*)m_pParentView;
                ::ResetDevice(w->GetROK());

                g_xPS.x.bPresentTest = 0;
                UpdatePermanenceSettings();

                g_xFS.iMotorControl = 0;
                g_xFS.bShowBatt = 0;
                UpdateFactorySettings();

                sync();

                qApp->exit(RET_EXEC_LOCK);
            }
            else if(pEvent->m_iEvType == KeyEvent::EV_DOUBLE_CLICKED)
            {
                qApp->exit(RET_EXEC_LOCK);
            }
        }
    }

    return QWidget::event(e);
}
