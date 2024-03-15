#ifndef ALERTDLG_H
#define ALERTDLG_H

#include "formbase.h"
#include "shared.h"
#include <QVector>

namespace Ui {
class AlertDlg;
}

class AlertDlg : public FormBase
{
    Q_OBJECT

public:
    explicit AlertDlg(QGraphicsView *pView, FormBase* pParentForm);
    ~AlertDlg();

    static int Locked;

    void    SetTitle(const QString& sTitle);
    void    SetTitle(const QString& strTitle, const int iSize);
    void    SetTitleEn(bool fShow);
    void    SetOkButton(bool fShow, QString sTitle);
    void    SetCancelButton(bool fShow, QString sTitle);
    void    SetButtonGroup(bool fShow);
    void    SetAlertCancel(bool fCancel);

    void    AddRadioItem(QString sText, int iID, bool fSelected = false);
    void    AddWidget(QWidget* pWidget);

    QVector<QPoint> GetChildPos();

    static int WarningYesNo(QGraphicsView* pView, FormBase* pParentForm, QString sTitle, QString sMsg);
    static int WarningOk(QGraphicsView* view, FormBase* parentForm, QString title, QString msg, int multiLine = 0);
    static QString ContainLineEdit(QGraphicsView* view, FormBase* parentForm, QString title, QString text, int maxLen,
                                   int mode = CKM_DEFAULT, QString _input_mask = "");
public slots:
    void    ClickedOk();
    void    ClickedCancel();
    void    ClickedRad(int);

    void    AutoTest();

protected:
    bool    eventFilter(QObject *obj, QEvent *e);

private:
    Ui::AlertDlg *  ui;

    QGraphicsScene* m_pContentsScene;
    QPoint          m_xOldPos;

    int     m_fAlertCancel;
};

#endif // ALERTDLG_H
