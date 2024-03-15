#include "datesettingform.h"
#include "ui_datesettingform.h"
#include "stringtable.h"
#include "base.h"

#include <QtGui>

DateSettingForm::DateSettingForm(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DateSettingForm)
{
    ui->setupUi(this);

    m_iDateFormat = 0;
    m_xDate = QDate::currentDate();
    RetranslateUI();

    connect(ui->spin1, SIGNAL(SigSelectionChanged()), this, SLOT(ChangedYear()));
    connect(ui->spin2, SIGNAL(SigSelectionChanged()), this, SLOT(ChangedMonth()));
}

DateSettingForm::~DateSettingForm()
{
    delete ui;
}

void DateSettingForm::SetDateFormat(int iFormat)
{
    m_iDateFormat = iFormat;
    RetranslateUI();
}

void DateSettingForm::SetDate(QDate date)
{
    m_xDate = date;
    RetranslateUI();
}

void DateSettingForm::ChangedYear()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    int iSelectedIdx = ui->spin3->GetSelectedIndex();
    QStringList vDayList;
    QDate xDate(ui->spin1->GetSelectedIndex() + 2020, ui->spin2->GetSelectedIndex() + 1, 1);
    for(int i = 0; i < xDate.daysInMonth(); i ++)
    {
        QString sItemText;
        sItemText.sprintf("%d", 1 + i);

        vDayList << sItemText;
    }

    ui->spin3->SetSpinRange(vDayList);
    ui->spin3->SetRepeat(true);

    if(iSelectedIdx >= xDate.daysInMonth())
        iSelectedIdx = 0;

    ui->spin3->SetSelect(iSelectedIdx);
}

void DateSettingForm::ChangedMonth()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    int iSelectedIdx = ui->spin3->GetSelectedIndex();
    QStringList xDayList;
    QDate xDate(ui->spin1->GetSelectedIndex() + 2020, ui->spin2->GetSelectedIndex() + 1, 1);
    for(int i = 0; i < xDate.daysInMonth(); i ++)
    {
        QString sItemText;
        sItemText.sprintf("%d", 1 + i);

        xDayList << sItemText;
    }

    ui->spin3->SetSpinRange(xDayList);
    ui->spin3->SetRepeat(true);

    if(iSelectedIdx >= xDate.daysInMonth())
        iSelectedIdx = 0;

    ui->spin3->SetSelect(iSelectedIdx);
}

QDate DateSettingForm::GetDate()
{
    QDate xDate(ui->spin1->GetSelectedIndex() + 2020, ui->spin2->GetSelectedIndex() + 1, ui->spin3->GetSelectedIndex() + 1);
    return xDate;
}

void DateSettingForm::changeEvent(QEvent* e)
{
    QWidget::changeEvent(e);
    if(e->type() == QEvent::LanguageChange)
        RetranslateUI();
}


void DateSettingForm::RetranslateUI()
{
    ui->horizontalLayout->removeWidget(ui->spin1);
    ui->horizontalLayout->removeWidget(ui->spin2);
    ui->horizontalLayout->removeWidget(ui->spin3);

    if(m_iDateFormat == 0)
    {
        ui->horizontalLayout->addWidget(ui->spin2);
        ui->horizontalLayout->addWidget(ui->spin3);
        ui->horizontalLayout->addWidget(ui->spin1);
    }
    else if(m_iDateFormat == 1)
    {
        ui->horizontalLayout->addWidget(ui->spin1);
        ui->horizontalLayout->addWidget(ui->spin2);
        ui->horizontalLayout->addWidget(ui->spin3);
    }

    QStringList xYearList, xMonList, xDayList;
    for(int i = 0; i <= 16; i ++)
    {
        QString sItemText;
        sItemText.sprintf("%d", 2020 + i);

        xYearList << sItemText;
    }
    xMonList << StringTable::Str_Jan;
    xMonList << StringTable::Str_Feb;
    xMonList << StringTable::Str_Mar;
    xMonList << StringTable::Str_Apr;
    xMonList << StringTable::Str_May;
    xMonList << StringTable::Str_Jun;
    xMonList << StringTable::Str_Jul;
    xMonList << StringTable::Str_Aug;
    xMonList << StringTable::Str_Sep;
    xMonList << StringTable::Str_Oct;
    xMonList << StringTable::Str_Nov;
    xMonList << StringTable::Str_Dec;

    for(int i = 0; i < m_xDate.daysInMonth(); i ++)
    {
        QString sItemText;
        sItemText.sprintf("%d", 1 + i);

        xDayList << sItemText;
    }

    ui->spin1->SetSpinRange(xYearList);
    ui->spin2->SetSpinRange(xMonList);
    ui->spin3->SetSpinRange(xDayList);

    ui->spin2->SetRepeat(true);
    ui->spin3->SetRepeat(true);

    ui->spin1->SetSelect(m_xDate.year() - 2020);
    ui->spin2->SetSelect(m_xDate.month() - 1);
    ui->spin3->SetSelect(m_xDate.day() - 1);
}
