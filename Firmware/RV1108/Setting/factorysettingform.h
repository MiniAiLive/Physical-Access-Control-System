#ifndef FACTORYSETTINGFORM_H
#define FACTORYSETTINGFORM_H

#include "itemformbase.h"
#include "appdef.h"

class QMenu;
class FactorySettingForm : public ItemFormBase
{
    Q_OBJECT
public:
    explicit FactorySettingForm(QGraphicsView *pView, FormBase* pParentForm);
    ~FactorySettingForm();

signals:

public slots:
    void    ClickedLogo();
    void    ClickedMotoType();
    void    ClickedMotoPolarity();
    void    ClickedMotoTime();
    void    ClickedKeepOpen();
    void    ClickedPresentation();
    void    ClickedBattTest();
    void    ClickedShowBatt();
    void    ClickedOverCurrent();

    void    BattLowClick();
    void    BattStepClick();
    void    BattNewClick();
    void    BattPowerDownClick();
    void    BattMenuClick();
    void    BattUpdateClick();
    void    BattSoundOffClick();

    void    FingerprintClick();

    void    HiddenCodeClick();
    void    HomeAutoMationClick();
    void    HumanSensorClick();
#if 0
    void    BaudrateClick();
#endif
    void    QRCodeReadFinished(QString strQRCode);
    void    QRCodeReadFinished(int iCamError);

    void    MotorClick();
    void    GyroAxisClick();
#if (TEST_DUTY_CYCLE == 1)
    void    DutyCycleValuesClick();
#endif //TEST_DUTY_CYCLE

    void    OnStart();
    void    OnStop();

protected:
    void    RetranslateUI();

private:
    QMenu*  m_pMenu;
};

#endif // FACTORYSETTINGFORM_H
