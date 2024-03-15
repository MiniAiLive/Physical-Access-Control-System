#ifndef FORMBASE_H
#define FORMBASE_H

#include <QWidget>
#include <QEventLoop>
#include <QEvent>
#include <QMutex>

#define ANIMATION_TYPE_RIGHT 0
#define ANIMATION_TYPE_LEFT 1
#define ANIMATION_TYPE_EXEC 2
#define ANIMATION_TYPE_DONE 3

#define EV_KEY_EVENT ((QEvent::Type)(QEvent::User + 10))
#define EV_CARD_SETTINGS_EVENT ((QEvent::Type)(QEvent::User + 11))
#define EV_MAC_RECV_EVENT ((QEvent::Type)(QEvent::User + 12))

class KeyEvent : public QEvent
{
public:
    KeyEvent(int iKeyID, int iEvType);

    enum {EV_PRESSED, EV_CLICKED, EV_DOUBLE_CLICKED, EV_LONG_PRESSED};

    int     m_iKeyID;
    int     m_iEvType;
};

class CardSettingsEvent : public QEvent
{
public:
    CardSettingsEvent(QString strSettings);

    QString     m_strCardSettings;
};

class MacRecvEvent : public QEvent
{
public:
    MacRecvEvent(QString strSettings);

    QString     m_strMac;
};


class QLineEdit;
class QGraphicsView;
class QGraphicsScene;
class QPropertyAnimation;
class QGraphicsPixmapItem;
class CustomKeyInputPanel;
class FormBase : public QWidget
{
    Q_OBJECT
    Q_PROPERTY(int GetSceneAnimation READ GetSceneAnimation WRITE SetSceneAnimation)
    Q_PROPERTY(int GetImeAnimation READ GetImeAnimation WRITE SetImeAnimation)
public:
    explicit FormBase(QGraphicsView *pView, FormBase* pParentForm);
    ~FormBase();

    static int          QuitFlag;
    static FormBase*    CurForm;
    void    Quit();

    QGraphicsView*  GetParent() {return m_pParentView;}

    void    SetBGColor(QColor xColor);
    void    SetLineEdit(QLineEdit* pLineEdit) {m_pLineEdit = pLineEdit;}

    void    ShowIME(QLineEdit* lineEdit, QPoint editPos);
    void    HideIME();
    int     GetImeAnimation();
    void    SetImeAnimation(int GetImeAnimation);

    int     GetSceneAnimation();
    void    SetSceneAnimation(int iSceneAnimation);

    QGraphicsScene* GetSurfaceScene();
    void    ChangeForm(FormBase* pNewForm, FormBase* pOldForm, int iAnimationType, int fDelPrevScene = 0);

    virtual QVector<QPoint> GetChildPos() {return m_vChildPos;}
signals:
    void    SigBack();
    void    SigShowIME();
    void    SigQuit();

public slots:
    void    SceneAnimationFinished();
    void    IMEAnimationFinished();
    void    ShowIME();

    virtual void    OnStart(int fDelPrevScene = 0);
    virtual void    OnResume();
    virtual int     OnExec(int iShowIME = 0);
    virtual void    OnPause();
    virtual void    OnStop();

protected:
    void    changeEvent(QEvent* e);
    bool    event(QEvent* e);

    virtual void RetranslateUI();

protected:
    QGraphicsView*      m_pParentView;
    FormBase*           m_pParentForm;
    QGraphicsScene*     m_pSurfaceScene;

    QEventLoop          m_xExecLoop;

//    QGraphicsScene*     m_pAnimationScene;
//    QPropertyAnimation*     m_pSceneAnimationProperty;
//    QGraphicsPixmapItem*    m_pSceneStartAnimationPixmapItem;

//    QPixmap             m_xSceneStartAnimationPixmap;
//    QPixmap             m_xSceneEndAnimationPixmap;
//    int                 m_iSceneAnimationValue;
//    int                 m_iAnimationType;
//    QEventLoop          m_xAnimationLoop;
    FormBase*           m_pNewForm;

//    QPropertyAnimation* m_pImeAnimationProperty;
//    QEventLoop          m_xImeAnimationLoop;
    CustomKeyInputPanel*    m_pImePanel;
    QGraphicsProxyWidget*   m_pProxyForm;
    QGraphicsProxyWidget*   m_pProxyIME;
    QPoint              m_xStartIMEPos;
    QPoint              m_xEndIMEPos;
    int                 m_iShowIME;
    QMutex              m_xImeMutex;

    QPoint              m_xOldFormPos;
    QPoint              m_xStartFormPos;
    QPoint              m_xEndFormPos;

    QVector<QPoint>     m_vChildPos;

    QLineEdit*          m_pLineEdit;
};


#endif // WIDGETBASE_H
