#include "humansensor.h"
#include "menuitem.h"
#include "base.h"
#include "stringtable.h"
#include "shared.h"
#include "datetimesettingform.h"
#include "alertdlg.h"
#include "selftestform.h"
#include "DBManager.h"
#include "mount_fs.h"
#include "waitingform.h"
#include "engineparam.h"
#include "fptask.h"
#include "aplayer.h"
#include "themedef.h"

#include <QtGui>
#include <unistd.h>

//#define ID_THEME_ITEM 0
#define ID_HUMAN_SENSOR_ITEM 0

HumanSensorForm::HumanSensorForm(QGraphicsView *pView, FormBase* pParentForm) :
    ItemFormBase(pView, pParentForm)
{
    SetBGColor(QColor::fromRgba(UITheme::BgColor2));

    int iPosY = 0;

    MenuItem* pHumanSensorForm = new MenuItem();
    pHumanSensorForm->setPos(QPoint(0, iPosY));
    pHumanSensorForm->SetBoundingRect(QRect(0, 0, SWTICH_ITEM_WIDTH, SWTICH_ITEM_HEIGHT));
    pHumanSensorForm->setData(KEY_TYPE, TYPE_SWITCH_ITEM);
    m_pScene->addItem(pHumanSensorForm);
    m_vMenuItems[ID_HUMAN_SENSOR_ITEM] = pHumanSensorForm;
    iPosY += SWTICH_ITEM_HEIGHT;
    connect(pHumanSensorForm, SIGNAL(clicked()), this, SLOT(HumanSensorClick()));

}

HumanSensorForm::~HumanSensorForm()
{

}

void HumanSensorForm::OnStart()
{
    FormBase::OnStart();
}

void HumanSensorForm::HumanSensorClick()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    PlaySoundLeft();

    g_xCS.x.bHumanSensor = 1 - g_xCS.x.bHumanSensor;
    UpdateCommonSettings();

    RetranslateUI();
}

void HumanSensorForm::RetranslateUI()
{
    SetTitle(StringTable::Str_Human_Sensor);

    m_vMenuItems[ID_HUMAN_SENSOR_ITEM]->setData(KEY_PRIMARY_TEXT, StringTable::Str_Human_Sensor);
    m_vMenuItems[ID_HUMAN_SENSOR_ITEM]->setData(KEY_CHECKED, g_xCS.x.bHumanSensor);
}
