#ifndef BASE_H
#define BASE_H

#include "EngineStruct.h"
#include "uitheme.h"
#include <QString>
#include <QImage>
#include <QDateTime>


#define RET_EXEC_LOCK           5
#define RET_EXEC_LOCK_MANAGER   6
#define RET_TIME_OUT            7
#define RET_SET_DATE_TIME       8

#define DATE_FORMAT_STR1        "MM/DD/YYYY"
#define DATE_FORMAT_STR2        "YYYY-MM-DD"

#define DATE_FORMAT1            "M/d/yyyy"
#define DATE_FORMAT2            "yyyy-M-d"

class ROKTHread;

QString     GetTimeStr(QTime xTime);
QString     GetFullTimeStr(QTime xTime);
QString     GetDateFormatStr(int iFormat);
QString     GetDateFormat(int iFormat);
QString     GetDateStr(QDate xDate);

DATETIME_32 QDateTim2DATETIME_32(QDateTime xTime);
QDateTime   DATETIME_32ToQDateTime(DATETIME_32 iTime);


QString     GetOmitText(QFont xFont, QString sText, int iMaxWidth);
QImage      ConvertData2QImage(unsigned char* pData, int iLen);
QImage      GetOverlayImage(QImage xSrcImage, QColor xOverColor);
QImage      GetOverlayImage(QSize dstSize, QColor bgColor, QImage srcImg);
QImage      GetCircleOverlayImage(QSize xDstSize, QImage xSrcImg);
QColor      GetAlphaColor(QColor xColor, double rOpacity);

QString     CalcOmitText(QFont xPaintFont, QString strOrg, int iMaxWidth);

QString     GetMotorTorqueStrByIdx(int iTorqueIdx);

void        ResetDevice(ROKTHread* pROK);

extern QString g_sLangNames[];

#endif // BASE_H
