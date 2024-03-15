#ifndef DATESETTINGITEM_H
#define DATESETTINGITEM_H

#include <QWidget>
#include <QDate>

namespace Ui {
class DateSettingForm;
}

class DateSettingForm : public QWidget
{
    Q_OBJECT

public:
    explicit DateSettingForm(QWidget *parent = 0);
    ~DateSettingForm();

    void    SetDateFormat(int iFormat);
    void    SetDate(QDate);
    QDate   GetDate();

public slots:
    void    ChangedYear();
    void    ChangedMonth();

protected:
    void    changeEvent(QEvent* e);

private:
    void    RetranslateUI();

private:
    Ui::DateSettingForm *ui;

    int     m_iDateFormat;
    QDate   m_xDate;
};

#endif // DATESETTINGITEM_H
