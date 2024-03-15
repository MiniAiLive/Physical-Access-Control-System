#include "soundvolumeform.h"
#include "ui_soundvolumeform.h"
#include "settings.h"
#include "shared.h"
#include "i2cbase.h"
#include "playthread.h"
#include "soundbase.h"
#include "mainbackproc.h"
#include "uarttask.h"
#include "DBManager.h"
#include "uitheme.h"

#include <QtGui>

extern UARTTask* g_pUartTask;

SoundVolumeForm::SoundVolumeForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SoundVolumeForm)
{
    ui->setupUi(this);

    ui->slider->setStyleSheet("QSlider::groove:horizontal {\
                              border: 0px solid #ffffff;\
                              height: 2px;\
                              background: #ffffff;\
                              margin: 25px 0;\
                          }\
                          QSlider::handle:horizontal {\
                              background: #2196f3;\
                              border: 1px solid #2196f3;\
                              width: 6px;\
                              margin: 10px 0;\
                              border-radius: 10px;\
                          }");
}

SoundVolumeForm::~SoundVolumeForm()
{
    delete ui;
}

int SoundVolumeForm::GetPos()
{
    return ui->slider->value();
}

void SoundVolumeForm::SetPos(QPoint xPos, bool bPressed)
{
    float iStep = (ui->slider->geometry().width() / 5.0f) - 1;
    int iPos = xPos.x();
    if(iPos < ui->slider->geometry().x())
        iPos = ui->slider->geometry().x();
    if(iPos >= ui->slider->geometry().right())
        iPos = ui->slider->geometry().right();

    iPos = iPos - ui->slider->geometry().x() + 1;
    int iVal = (int)(iPos / iStep + 0.5f);
    if(iPos >= ui->slider->geometry().width())
        iVal = ui->slider->maximum();

    SetPos(iVal, bPressed);
}

void SoundVolumeForm::SetPos(int iVal, bool bPressed)
{
    ui->slider->setValue(iVal);

    QString strVal;
    strVal.sprintf("%d %%", iVal * 20);
    ui->lblVolumn->setText(strVal);
    ui->lblVolumn->setFont(g_UITheme->PrimaryFont);

    if(iVal == 0)
        ui->lblIcon->setPixmap(QPixmap(":/icons/ic_sound_off.png"));
    else
        ui->lblIcon->setPixmap(QPixmap(":/icons/ic_sound_on.png"));

    if(bPressed == false)
    {
        ui->slider->setStyleSheet("QSlider::groove:horizontal {\
                                  border: 0px solid #ffffff;\
                                  height: 2px;\
                                  background: #ffffff;\
                                  margin: 25px 0;\
                              }\
                              QSlider::handle:horizontal {\
                                  background: #2196f3;\
                                  border: 1px solid #2196f3;\
                                  width: 5px;\
                                  margin: 10px 0;\
                                  border-radius: 10px;\
                              }");

    }
    else
    {
        ui->slider->setStyleSheet("QSlider::groove:horizontal {\
                          border: 0px solid #ffffff;\
                          height: 2px;\
                          background: #ffffff;\
                          margin: 25px 0;\
                      }\
                      QSlider::handle:horizontal {\
                          background: #f0f0f0;\
                          border: 1px solid #f0f0f0;\
                          width: 5px;\
                          margin: 10px 0;\
                          border-radius: 10px;\
                      }");
    }
}


void SoundVolumeForm::mousePressEvent(QMouseEvent* e)
{
    e->accept();

    SetPos(e->pos(), true);
}

void SoundVolumeForm::mouseMoveEvent(QMouseEvent* e)
{
    SetPos(e->pos(), true);
}

void SoundVolumeForm::mouseReleaseEvent(QMouseEvent* e)
{
    SetPos(e->pos(), false);

    int iVal = GetPos();
    g_xCS.x.bSound = iVal * 2;
    g_xSS.bSound = g_xCS.x.bSound;

    UpdateCommonSettings();

    SetSoundVol(g_xSS.bSound * 10);

    PlayThread::PlaySound(STM_SID_KEY_LEFT, 0, g_xSS.bSound);
}
