#include "base.h"
#include "appdef.h"
#include "faceengine.h"
#include "shared.h"
#include "i2cbase.h"
#include "stringtable.h"
#include "rokthread.h"
#include "fptask.h"
#include "mount_fs.h"
#include "DBManager.h"
#include "faceengine.h"
#include "engineparam.h"
#include "countbase.h"
#include "playthread.h"
#include "uarttask.h"
#include "mainbackproc.h"
#include "wifiproc.h"
#include "wificonfigproc.h"
#include "soundbase.h"

#include <QtGui>

extern UARTTask* g_pUartTask[];

QString g_sLangNames[] =
{
    QString::fromUtf8("English"),
    QString::fromUtf8("Русский язык"),
    QString::fromUtf8("Deutsch"),
    QString::fromUtf8("Português"),
};

QString GetTimeStr(QTime xTime)
{
    int iTimeFormat = g_xCS.x.bHourFormat;

    QString sTime;
    if(iTimeFormat == 1)
        sTime.sprintf("%d:%02d", xTime.hour(), xTime.minute());
    else
    {
        QString sAm;
        if(xTime.hour() < 12)
            sAm = StringTable::Str_AM;
        else
            sAm = StringTable::Str_PM;

        if((xTime.hour() % 12) == 0)
            sTime.sprintf("12:%02d %s", xTime.minute(), sAm.toUtf8().data());
        else
            sTime.sprintf("%d:%02d %s", xTime.hour() % 12, xTime.minute(), sAm.toUtf8().data());
    }

    return sTime;
}

QString GetFullTimeStr(QTime xTime)
{
    int iTimeFormat = g_xCS.x.bHourFormat;

    QString sTime;
    if(iTimeFormat == 1)     //24
        sTime.sprintf("%d:%02d:%02d", xTime.hour(), xTime.minute(), xTime.second());
    else
    {
        QString sAm;
        if(xTime.hour() < 12)
            sAm = StringTable::Str_AM;
        else
            sAm = StringTable::Str_PM;

        if((xTime.hour() % 12) == 0)
            sTime.sprintf("12:%02d:%02d %s", xTime.minute(), xTime.second(), sAm.toUtf8().data());
        else
            sTime.sprintf("%d:%02d:%02d %s", xTime.hour() % 12, xTime.minute(), xTime.second(), sAm.toUtf8().data());
    }

    return sTime;
}

QString GetDateFormatStr(int iFormat)
{
    if(iFormat == 0)
        return QString::fromUtf8(DATE_FORMAT_STR1);
    else
        return QString::fromUtf8(DATE_FORMAT_STR2);
}

QString GetDateFormat(int iFormat)
{
    if(iFormat == 0)
        return QString::fromUtf8(DATE_FORMAT1);
    else
        return QString::fromUtf8(DATE_FORMAT2);
}

QString GetDateStr(QDate xDate)
{
    return xDate.toString(GetDateFormat(g_xCS.x.bDateFormat));
}


DATETIME_32 QDateTim2DATETIME_32(QDateTime xTime)
{
    DATETIME_32 xRet = { 0 };
    xRet.x.iYear = xTime.date().year() - 2000;
    xRet.x.iMon = xTime.date().month() - 1;
    xRet.x.iDay = xTime.date().day();
    xRet.x.iHour = xTime.time().hour();
    xRet.x.iMin = xTime.time().minute();
    xRet.x.iSec = xTime.time().second();

    return xRet;
}

QDateTime DATETIME_32ToQDateTime(DATETIME_32 xTime)
{
    QDate xInDate(xTime.x.iYear + 2000, xTime.x.iMon + 1, xTime.x.iDay);
    QTime xInTime(xTime.x.iHour, xTime.x.iMin, xTime.x.iSec);

    QDateTime xRetTime(xInDate, xInTime);

    if(!xRetTime.date().isValid())
        xRetTime.setDate(QDate(2020, 1, 1));

    if(!xRetTime.time().isValid())
        xRetTime.setTime(QTime(0, 0, 0));

    return xRetTime;
}

QString GetOmitText(QFont xFont, QString sText, int iMaxWidth)
{
    QFontMetrics xFontMetrics(xFont);


    if(xFontMetrics.width(sText)> iMaxWidth)
    {
        QString sTmpText;

        int iIdx = 0;
        for(iIdx = 0; iIdx < sText.length(); iIdx ++)
        {
            sTmpText = sText.left(iIdx);
            sTmpText += "...";

            if(xFontMetrics.width(sTmpText) > iMaxWidth)
                break;
        }

        iIdx --;

        if(iIdx < 0)
        {
            return QString();
        }
        else
        {
            sTmpText = sText.left(iIdx);
            sTmpText += "...";

            return sTmpText;
        }
    }

    return sText;
}

QImage ConvertData2QImage(unsigned char* pData, int iLen)
{
    if(pData == NULL || iLen == 0)
        return QImage();

    QImage xImage = QImage::fromData(pData, N_MAX_JPG_FACE_IMAGE_SIZE);
    return xImage;
}


QImage GetOverlayImage(QImage xSrcImage, QColor xOverColor)
{
    QImage xResultImage(xSrcImage.size(), QImage::Format_ARGB32_Premultiplied);

    QPainter xPainter(&xResultImage);
    xPainter.setCompositionMode(QPainter::CompositionMode_Source);
    xPainter.fillRect(xResultImage.rect(), Qt::transparent);
    xPainter.setCompositionMode(QPainter::CompositionMode_SourceOver);
    xPainter.fillRect(xSrcImage.rect(), xOverColor);
    xPainter.setCompositionMode(QPainter::CompositionMode_DestinationIn);
    xPainter.drawImage(0, 0, xSrcImage);
    xPainter.end();

    return xResultImage;
}

QImage GetOverlayImage(QSize dstSize, QColor bgColor, QImage srcImg)
{
    QImage xResultImage(dstSize, QImage::Format_ARGB32_Premultiplied);

    QPainter xPainter(&xResultImage);
    xPainter.fillRect(xResultImage.rect(), bgColor);
    xPainter.drawImage((dstSize.width() - srcImg.width()) / 2, (dstSize.width() - srcImg.height()) / 2, srcImg);
    xPainter.end();

    return xResultImage;
}

QImage GetCircleOverlayImage(QSize xDstSize, QImage xSrcImg)
{
    QImage xCircleImg(":/icons/circle.png");
    QImage xResultImage(xDstSize, QImage::Format_ARGB32_Premultiplied);

    xCircleImg = xCircleImg.scaled(xDstSize, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);
    QImage xScaledImg;
    if(!xSrcImg.isNull())
        xScaledImg = xSrcImg.scaled(xDstSize, Qt::KeepAspectRatioByExpanding, Qt::SmoothTransformation);

    QRect xDstRect((xDstSize.width() - xScaledImg.width()) / 2, (xDstSize.height() - xScaledImg.height()) / 2, xScaledImg.width(), xScaledImg.height());

    QPainter xPainter(&xResultImage);
    xPainter.setRenderHints(QPainter::Antialiasing);
    xPainter.setCompositionMode(QPainter::CompositionMode_Source);
    xPainter.fillRect(xResultImage.rect(), Qt::transparent);
    xPainter.setCompositionMode(QPainter::CompositionMode_SourceOver);
    xPainter.drawImage(QRect(0, 0, xDstSize.width(), xDstSize.height()), xCircleImg);
    xPainter.setCompositionMode(QPainter::CompositionMode_SourceIn);
    xPainter.drawImage(xDstRect, xScaledImg);
    xPainter.end();

    return xResultImage;
}

QColor GetAlphaColor(QColor xColor, double rOpacity)
{
    return QColor(qMin((double)255, xColor.red() * rOpacity), qMin((double)255, xColor.green() * rOpacity), qMin((double)255, xColor.blue() * rOpacity));
}


QString CalcOmitText(QFont xPaintFont, QString strOrg, int iMaxWidth)
{
    QFontMetrics xFontMetrics(xPaintFont);


    if(xFontMetrics.width(strOrg)> iMaxWidth)
    {
        QString strTmpText;
        int iIndex = 0;

        for(iIndex = 0; iIndex < strOrg.length(); iIndex ++)
        {
            strTmpText = strOrg.left(iIndex);
            strTmpText += "...";

            if(xFontMetrics.width(strTmpText) > iMaxWidth)
                break;
        }

        iIndex --;

        if(iIndex < 0)
        {
            return QString();
        }
        else
        {
            strTmpText = strOrg.left(iIndex);
            strTmpText += "...";

            return strTmpText;
        }
    }

    return strOrg;
}


QString GetMotorTorqueStrByIdx(int iTorqueIdx)
{
    if(iTorqueIdx == 0)
        return StringTable::Str_Motor_Torque_Low;
    else if(iTorqueIdx == 1)
        return StringTable::Str_Motor_Torque_Medium;
    else if(iTorqueIdx == 2)
        return StringTable::Str_Motor_Torque_High;
    else if(iTorqueIdx == 3)
        return StringTable::Str_Motor_Torque_Highest;

    return StringTable::Str_Motor_Torque_Medium;
}

void ResetDevice(ROKTHread* pROK)
{
    FaceEngine::ResetAll();
    FaceEngine::Release();

    ResetSystemState(APP_SETTINGS);
    ResetCommonSettings();

    FaceEngine::Create("/db1", 0, g_xEP.fUpdate, g_xEP.arThreshold, g_xEP.iEngineType == 2, g_xEP.fUpdateA, g_xEP.iMotionOffset, (g_xEP.iRemoveNoise << 2) | (g_xEP.iRemoveGlass << 1) | g_xEP.iMotionFlag,
                       g_xEP.iMinUserLum, g_xEP.iMaxUserLum, g_xEP.iSatThreshold, 1);
    dbm_CheckLog();

    if(pROK)
        pROK->StopROK();

    if(g_pUartTask[BACK_BOARD_PORT])
        g_pUartTask[BACK_BOARD_PORT]->Stop();

    if(g_pUartTask[PWR_BOARD_PORT])
        g_pUartTask[PWR_BOARD_PORT]->Stop();

    PlayThread::WaitForFinished();

    DATETIME_32 xNow = { 0 };
    xNow.x.iYear = 2020 - 2000;
    xNow.x.iMon = 1 - 1;
    xNow.x.iDay = 1;
    xNow.x.iHour = 0;
    xNow.x.iMin = 0;
    xNow.x.iSec = 0;
    SetCurDateTime(xNow);
    RK805_SetRTC(xNow);

    if(pROK)
        pROK->Start();

    if(g_pUartTask[BACK_BOARD_PORT])
        g_pUartTask[BACK_BOARD_PORT]->Start();

    if(g_pUartTask[PWR_BOARD_PORT])
        g_pUartTask[PWR_BOARD_PORT]->Start();

    g_xSS.bPresentation = g_xCS.x.bPresentation;
    g_xSS.bSound = g_xCS.x.bSound;

#if USING_BUZZER
    MainSTM_Command(MAIN_STM_BUZZER_INIT);
#else
    PlayPickSound(0, 1);
#endif
}
