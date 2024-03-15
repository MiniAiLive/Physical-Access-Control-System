#include "camtestform.h"
#include "base.h"
#include "mainwindow.h"
#include "camera_api.h"
#include "shared.h"
#include "camerasurface.h"
#include "rokthread.h"
#include "lcdtask.h"
#include "drv_gpio.h"
#include "i2cbase.h"
#include "mainbackproc.h"
#include "engineparam.h"
#include "my_lang.h"
#include "DBManager.h"

#include <QtGui>
#include <unistd.h>
#include <linux/videodev2.h>
#include <linux/fb.h>
#include <sys/ioctl.h>

unsigned int g_HEADER_BG_COLOR[3] = {BG_PUXIN_MSG_BTN_NORMAL_COLOR, BG_PUXIN_MSG_BTN_NORMAL_COLOR_BR, BG_PUXIN_MSG_BTN_NORMAL_COLOR_PI};

#define SLIDER_LEFT 120
#define SLIDER_RIGHT 440

extern void GetColorCtrlParam(unsigned char* pbBright, unsigned char* pbContrast, unsigned char* pbSaturation);

CamTestForm::CamTestForm(QGraphicsView *pxView, FormBase* pxParentForm) :
    FormBase(pxView, pxParentForm)
{
    SetBGColor(Qt::black);

    setAutoDelete(false);
    m_iTimeOut = -1;
    m_bIsDevTest = false;
    ResetButtons();
}

CamTestForm::~CamTestForm()
{

}

void CamTestForm::StartTest(int iCamID, int iTimeOut, int fTestVol, int iSetCam)
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    FormBase::OnStart();

    if(iCamID == DEV_CLR_CAM)
    {
        m_bIsDevTest = true;
        iCamID = CLR_CAM;
    }

    if(iCamID == DEV_IR_CAM1)
    {
        m_bIsDevTest = true;
        iCamID = IR_CAM1;
    }

    m_ID = iCamID;
    m_fRunning = 1;
    m_fTestVol = fTestVol;
    m_iTimeOut = iTimeOut;
    m_iSetClrCam = iSetCam;
#if (AUTO_TEST == 1)
    m_iTimeOut = 500;
#endif

    QThreadPool::globalInstance()->start(this);
}

void CamTestForm::OnPause()
{
    FormBase::OnPause();

    m_fRunning = 0;
    QThreadPool::globalInstance()->waitForDone();
}


extern void ImageRotation(unsigned char* imageData, int imageWidth, int imageHeight, int pixelWidth, int rotation);
extern int ConvertClrBuffer(struct buffer* src_buf, struct buffer* dst_buf, struct buffer* dst_yuv_buf);

void CamTestForm::run()
{
#if (AUTO_TEST == 1)
    qDebug() << "[Call]" << this << __FUNCTION__;
#endif

    printf("[Cam Test] Start\n");


    int fCommEnd = 0;
    int iIndex = 0;
    float rStart = Now();

    struct v4l2_buffer buf;
    MainWindow* w = (MainWindow*)m_pParentView;

    int iCamError = 0;

    if(m_ID == CLR_CAM)
    {
        int iRet = camera_init(m_ID, WIDTH_800, HEIGHT_600, 30, FRAME_NUM, 0, 0);

        if(iRet == -1)
        {
            iCamError = 1;
            SigBack(iCamError);
            return;
        }

        init_clr_cam_regs();
        camera_set_regval(CLR_CAM, 0xfe, 0x00);
        camera_clr_set_exp(INIT_CLR_EXP);
        camera_clr_set_gain(INIT_CLR_GAIN);
    }
    else
    {        
        int iRet = camera_init(IR_CAM, WIDTH_1280, HEIGHT_720, 30, FRAME_NUM, 1, 1);
        if(iRet == -1)
        {
            iCamError = 1;

            SigBack(iCamError);
            return;
        }
    }

    struct buffer tmp_yuv_buf;
    CLEAR(tmp_yuv_buf);

    create_buffer(&g_show_buf, WIDTH_1280 * HEIGHT_720 * 3 / 2);
    create_buffer(&tmp_yuv_buf, WIDTH_800 * HEIGHT_600 * 3 / 2);

    unsigned char nBright = 0;
    unsigned char nContrast = 0;
    unsigned char nSaturation = 0;
    GetColorCtrlParam(&nBright, &nContrast, &nSaturation);
    printf("Contrast = %d\n", nContrast);

    if(m_ID == IR_CAM)
    {
        camera_switch(IR_CAM, IR_CAM_SUB0);
        camera_set_exp_byreg(IR_CAM, INIT_EXP);
        camera_set_irled(IR_CAM, 2, 0);
        camera_set_irled_on(IR_CAM, 1);
    }
    else if(m_ID == IR_CAM1)
    {
        camera_switch(IR_CAM, IR_CAM_SUB1);
        camera_set_exp_byreg(IR_CAM1, INIT_EXP);
        camera_set_irled(IR_CAM, 2, 0);
        camera_set_irled_on(IR_CAM, 1);
    }
    else if(m_ID == CLR_CAM && m_iSetClrCam)
    {
#if 1
//      unsigned char flip_reg;
//      camera_get_regval(CLR_CAM, 0x1E, &flip_reg);
//      flip_reg &= ~(1 << 4);
        //camera_set_regval(CLR_CAM, 0x1E, 0x60); //set by kernel.
#endif

#if 0
        InitSettings();
        ResetButtons();
        AddButton(BTN_ID_SETTINGS, 340, LCD_HEADER_HEIGHT + 20, 450, LCD_HEADER_HEIGHT + 130, ":/icons/ic_menu_nor.png", ":/icons/ic_menu_act.png", 0, 0);
#endif

        SetClrBrightness(nBright);
        SetClrSaturation(nSaturation);
        SetClrContrast(nContrast);
    }

//    usleep(800 * 1000);

    LCDTask::FB_Init();
    LCDTask::DispOn();
    LCDTask::LCD_MemClear(0);
    LCDTask::LCD_Update();
    LCDTask::VideoStart();

    int iFrameCount = 0;
    int iFirst = 0;
    int iLedOnFlag = 0;
    int iErrorCount = 0;
    int iOldReserved = 0;
    int iID = m_ID;
    if(iID == IR_CAM1)
        iID = IR_CAM;

    int iOldPicMode = g_xCS.x.bPictureMode;

    LCDTask* pLCDTask = new LCDTask(NULL);

    UpdateLCD();

    while(m_fRunning)
    {
        float rNow = Now();
        if(rNow - rStart > m_iTimeOut && m_iTimeOut != -1)
            break;

        int iRet = wait_camera_ready (iID);
        if (iRet < 0)
        {
            iCamError = 1;
            break;
        }

        memset(&buf, 0, sizeof(struct v4l2_buffer));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_USERPTR;
        ioctl(cam_fd[iID], VIDIOC_DQBUF, &buf);

        if(iID == IR_CAM)
        {
            int id = (buf.index >> 16);
            int reserved = (buf.index >> 8) & 0xFF;
            buf.index = buf.index & 0xFF;

            if (iFrameCount > 1)
            {
                if((id == 0/* && m_ID == IR_CAM*/) || (id == 1/* && m_ID == IR_CAM1*/))
                {
                    if(iLedOnFlag == 0 && reserved == 1 && iOldReserved == 0)
                    {
                        iLedOnFlag ++;
                    }
                    else if(iLedOnFlag == 1)
                    {
                        unsigned char* pbLedOnSrc = (unsigned char*)buffers[IR_CAM][buf.index].start;
                        for(int i = 0; i < HEIGHT_720; i ++)
                            memcpy(g_irOnBayerL + i * WIDTH_960 * 2, pbLedOnSrc + (160 + i * WIDTH_1280) * 2, WIDTH_960 * 2);


                        Simple10Bayer2Yuv(g_irOnBayerL, g_irOnData, WIDTH_960, HEIGHT_720, 0);
                        memcpy(g_show_buf.start, g_irOnData, sizeof(g_irOnData));

                        LCDTask::VideoMap(WIDTH_960, HEIGHT_720, &g_show_buf, IR_CAM, id);

                        iLedOnFlag ++;
                    }
                    else if(iLedOnFlag > 4)
                        iLedOnFlag = 0;
                    else if(iLedOnFlag != 0)
                        iLedOnFlag ++;
                }
                else
                {
                    if(m_ID == IR_CAM && id == IR_CAM_SUB1)
                    {
                        camera_switch(IR_CAM, 0);
                        camera_set_irled_on(IR_CAM, 1);
                    }
                    else if(m_ID == IR_CAM1 && id == IR_CAM_SUB0)
                    {
                        camera_switch(IR_CAM, 1);
                        camera_set_irled_on(IR_CAM, 1);
                    }

                    iErrorCount ++;
                }
            }

            if(iIndex > 8)
            {
                iIndex = 0;
                camera_set_irled_on(IR_CAM, 1);
            }

            iOldReserved = reserved;
        }
        else
        {
            ConvertClrBuffer(&buffers[CLR_CAM][buf.index], NULL, &tmp_yuv_buf);
            memcpy(g_clrYuvData, tmp_yuv_buf.start, tmp_yuv_buf.length);

#if 0
            if(iFirst == 0)
            {
                int iMyError = 1;
                if(g_iForceClrCam == 0)
                {
                    unsigned char* pbBuf = (unsigned char*)g_clrYuvData;
                    int iCheck = 1; //CheckYUVBuffer(pbBuf, HEIGHT_240, WIDTH_320);

                    int iAverage = 0;
                    iMyError = getImageAverageLight(pbBuf, HEIGHT_240, WIDTH_320, &iAverage);
                    printf("[Clr] Average Light = %d, Check=%d\n", iAverage, iCheck);
                    if (iAverage == 255) //may be error on image.
                    {
                        ioctl(cam_fd[CLR_CAM], VIDIOC_QBUF, &buf);
                        continue;
                    }
                    if ((iAverage < 35 || iCheck == 0) && g_xCS.x.bShowCam == 2)
                    {
                        g_xSS.iShowIrCamera = 2;
                    }
                    else
                    {
                        int nEntireValue = 0;
                        float rNonSatPixelRate = 0;
                        GetYAVGValueOfClr_Entire_SAT(pbBuf, &nEntireValue, &rNonSatPixelRate);
#if 0
                        if ((rNonSatPixelRate > 0.95 && nEntireValue < 170))
                        {
                            g_nExposureMode = 0;
                        }
                        else if (rNonSatPixelRate < 0.825 || (rNonSatPixelRate > 0.825 && nEntireValue > 140))
                        {
                            g_nExposureMode = 1;
                        }
#endif
                        if (iAverage > 230)
                        {
                            if (g_nExposureMode)
                            {
                                g_nExposureClr = 520;
                                g_nGainClr = INIT_CLR_GAIN_1;
                                camera_clr_set_exp(g_nExposureClr);
                                camera_set_regval(CLR_CAM, 0x87, (unsigned char)g_nGainClr);
                            }
                            else
                            {
                                //g_nExposureClr = INIT_CLR_EXP_1;
                                g_nGainClr = INIT_CLR_GAIN_1;
                                camera_set_regval(CLR_CAM, 0x87, (unsigned char)g_nGainClr);
                                g_nClrFramePassCount = -1;

                            }
                        }
                        else if (iAverage > 180)
                        {
                            if (g_nExposureMode)
                            {
                                g_nExposureClr = INIT_CLR_EXP_1;
                                g_nGainClr = INIT_CLR_GAIN_1;
                                camera_clr_set_exp(g_nExposureClr);
                                camera_set_regval(CLR_CAM, 0x87, (unsigned char)g_nGainClr);
                                g_nClrFramePassCount = -1;
                            }
                            else
                            {
                                //g_nExposureClr = INIT_CLR_EXP_1;
                                g_nGainClr = INIT_CLR_GAIN_1;
                                camera_set_regval(CLR_CAM, 0x87, (unsigned char)g_nGainClr);
                                g_nClrFramePassCount = -1;

                            }
                        }
                    }
                }
                else
                {
                    iMyError = getImageAverageLight((unsigned char*)g_clrYuvData, HEIGHT_240, WIDTH_320);
                }

                iFirst = 1;
                if(iMyError == 1)
                {
                    ioctl(cam_fd[CLR_CAM], VIDIOC_QBUF, &buf);
                    continue;
                }
            }
#endif

            iFirst = 2;
            if (iFrameCount > 2)
                LCDTask::VideoMap(WIDTH_640, HEIGHT_480, &tmp_yuv_buf, CLR_CAM, 0, g_xSS.iVDBMode);

            CalcClrNextExposure();

            unsigned char iB, iC, iS;
            GetColorCtrlParam(&iB, &iC, &iS);
            if(iB != nBright)
            {
                nBright = iB;
                SetClrBrightness(nBright);
            }

            if(iC != nContrast)
            {
                nContrast = iC;

                SetClrContrast(nContrast);
            }
            if (g_xCS.x.bPictureMode == PM_WARM)
            {
                SetClrWarmMode();
            }
            else if (g_xCS.x.bPictureMode == PM_COOL)
            {
                SetClrCoolMode();
            }
            else
            {
                if(iS != nSaturation || iOldPicMode != g_xCS.x.bPictureMode)
                {
                    nSaturation = iS;

                    SetClrSaturation(nSaturation);

                }
            }

            iOldPicMode = g_xCS.x.bPictureMode;
        }

        if(!m_bIsDevTest)
        {            
            if (iID == IR_CAM || iFrameCount > 1)
            {
//                LCDTask::DrawMemFillRect(0, 0, MAX_X, LCD_HEADER_HEIGHT, C_Black);
//                pLCDTask->LCD_DrawText(0, 0, MAX_X - 1, LCD_HEADER_HEIGHT - 1,
//                             My_Camera_Test,
//                             Canvas::C_TA_Center | Canvas::C_TA_Middle,
//                             LCD_FOOTER_FONT_SIZE, C_White, g_HEADER_BG_COLOR[g_xCS.x.bTheme]);

//                LCDTask::DrawMemFillRect(0, MAX_Y - LCD_FOOTER_HEIGHT, MAX_X, MAX_Y, C_Black);
//                pLCDTask->LCD_DrawText(10, MAX_Y - LCD_FOOTER_HEIGHT, MAX_X - 20, MAX_Y - 1,
//                             My_Str_Tap_To_Cancel,
//                             Canvas::C_TA_Center | Canvas::C_TA_Middle,
//                             LCD_FOOTER_FONT_SIZE, C_White, C_NoBack);

//                LCDTask::LCD_Update(0, 0, MAX_X, LCD_HEADER_HEIGHT);
//                LCDTask::LCD_Update(0, MAX_Y - LCD_FOOTER_HEIGHT, MAX_X, MAX_Y);
            }
        }

        ioctl(cam_fd[iID], VIDIOC_QBUF, &buf);

        if(m_iTimeOut == -1)
            w->GetROK()->InitTime();

        iIndex ++;
        iFrameCount ++;
    }

#if 1
    if(iID == IR_CAM)
    {
        camera_switch(IR_CAM, 1);
        usleep(100 * 1000);

        CLEAR(buf);
        memset(&buf, 0, sizeof(struct v4l2_buffer));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_USERPTR;
        ioctl(cam_fd[IR_CAM], VIDIOC_DQBUF, &buf);
    }

#endif

    camera_release(iID);

    if(g_show_buf.start)
    {
        memset(g_show_buf.start, 0, g_show_buf.length);
        memset((unsigned char*)g_show_buf.start + WIDTH_640 * HEIGHT_480, 0x80, WIDTH_640 * HEIGHT_480 / 2);

        LCDTask::VideoMap(WIDTH_640, HEIGHT_480, &g_show_buf, IR_CAM);
    }

    LCDTask::VideoStop();
    LCDTask::FB_Release();

    delete_buffer(&tmp_yuv_buf);
    delete_buffer(&g_show_buf);

    if(pLCDTask)
    {
        delete pLCDTask;
        pLCDTask = NULL;
    }
        
    GPIO_fast_setvalue(IR_LED, OFF);
    printf("[Cam Test] End\n");

    if(!FormBase::QuitFlag)
        emit SigBack(iCamError);
}

void CamTestForm::mousePressEvent(QMouseEvent* e)
{
    QWidget::mousePressEvent(e);
#if 0
    if(m_ID == CLR_CAM && m_iSetClrCam)
    {
        if(m_iSettings == 0)
        {
            int iPressed = CheckBtnState(e->pos(), 0);
            if(iPressed == -1)
            {
                m_fRunning = 0;
                e->ignore();
                return;
            }
        }
        else
        {
            int iPressed = CheckBtnState(e->pos(), 0);
            if(iPressed != BTN_ID_RESET && iPressed != BTN_ID_WARM && iPressed != BTN_ID_COOL)
            {                
                int iExists = -1;
                for(int i = 0; i < 3; i ++)
                {
                    if(m_axProcessSlider[i].contains(e->pos()))
                    {
                        iExists = i;
                        break;
                    }
                }

                if(iExists >= 0)
                {
                    m_iSelectSlider = iExists;

                    int i = m_iSelectSlider;
                    int iX = e->pos().x();
                    m_iImageProcess[i] = 10 + (iX - SLIDER_LEFT) / (float)(SLIDER_RIGHT - SLIDER_LEFT) * 90;
                    m_iImageProcess[i] = __max(10, m_iImageProcess[i]);
                    m_iImageProcess[i] = __min(100, m_iImageProcess[i]);

                    if(i == 0)
                        g_xCS.x.bBrightValue = m_iImageProcess[i] / 10;
                    else if(i == 1)
                        g_xCS.x.bContrastValue = m_iImageProcess[i] / 10;
                    else if(i == 2)
                        g_xCS.x.bSaturationValue = m_iImageProcess[i] / 10;
                }
                else
                {
                    printf("clear scene %d, %d\n", e->pos().x(), e->pos().y());
                    m_iSelectSlider = -1;
                    m_iSettings = 0;

                    ResetButtons();
                    AddButton(BTN_ID_SETTINGS, 340, LCD_HEADER_HEIGHT + 20, 450, LCD_HEADER_HEIGHT + 130, ":/icons/ic_menu_nor.png", ":/icons/ic_menu_act.png", 0, 0);

                    UpdateLCD();
                    e->ignore();
                    return;
                }
            }
        }
        UpdateLCD();
    }
    else
    {
        m_fRunning = 0;
        e->ignore();
        return;
    }

    e->accept();
#endif
    m_fRunning = 0;
}

void CamTestForm::mouseMoveEvent(QMouseEvent* e)
{
    QWidget::mouseMoveEvent(e);

    if(m_ID == CLR_CAM && m_iSetClrCam)
    {
        if(m_iSettings == 0)
        {
//            CheckBtnState(e->pos(), 1);
        }
        else
        {
//            CheckBtnState(e->pos(), 1);

            if(m_iSelectSlider >= 0)
            {
                int i = m_iSelectSlider;
                int iX = e->pos().x();
                m_iImageProcess[i] = 10 + (iX - SLIDER_LEFT) / (float)(SLIDER_RIGHT - SLIDER_LEFT) * 90;
                m_iImageProcess[i] = __max(10, m_iImageProcess[i]);
                m_iImageProcess[i] = __min(100, m_iImageProcess[i]);

                if(i == 0)
                    g_xCS.x.bBrightValue = m_iImageProcess[i] / 10;
                else if(i == 1)
                    g_xCS.x.bContrastValue = m_iImageProcess[i] / 10;
                else if(i == 2)
                    g_xCS.x.bSaturationValue = m_iImageProcess[i] / 10;
            }
        }
        UpdateLCD();
    }
}


void CamTestForm::mouseReleaseEvent(QMouseEvent* e)
{
    QWidget::mouseReleaseEvent(e);

    if(m_ID == CLR_CAM && m_iSetClrCam)
    {
        if(m_iSettings == 0)
        {
            int iPressed = CheckBtnState(e->pos(), 2);
            if(iPressed == BTN_ID_SETTINGS)
            {
                m_iSettings = 1;
                ResetButtons();
                AddButton(BTN_ID_RESET, 340, 670 - LCD_FOOTER_HEIGHT, 450, 780 - LCD_FOOTER_HEIGHT, ":/icons/ic_reset_nor.png", ":/icons/ic_reset_act.png", 0, 0);

                AddButton(BTN_ID_WARM, 185, 670 - LCD_FOOTER_HEIGHT, 295, 780 - LCD_FOOTER_HEIGHT, ":/icons/ic_btn_warm.png", ":/icons/ic_btn_warm.png", 0, 0, g_xCS.x.bPictureMode == PM_WARM ? BTN_STATE_PRESSED : BTN_STATE_NONE);
                AddButton(BTN_ID_COOL, 30, 670 - LCD_FOOTER_HEIGHT, 140, 780 - LCD_FOOTER_HEIGHT, ":/icons/ic_btn_cool.png", ":/icons/ic_btn_cool.png", 0, 0, g_xCS.x.bPictureMode == PM_COOL ? BTN_STATE_PRESSED : BTN_STATE_NONE);
            }
        }
        else
        {
            int iPressed = CheckBtnState(e->pos(), 1);
            if(iPressed == BTN_ID_RESET)
            {
                g_xCS.x.bBrightValue = 5;
                g_xCS.x.bContrastValue = 5;
                g_xCS.x.bSaturationValue = 5;
                g_xCS.x.bPictureMode = PM_NORMAL;
                UpdateCommonSettings();

                m_iImageProcess[0] = 50;
                m_iImageProcess[1] = 50;
                m_iImageProcess[2] = 50;
            }
            else if(iPressed == BTN_ID_WARM)
            {
                g_xCS.x.bBrightValue = 5;
                g_xCS.x.bContrastValue = 5;
                g_xCS.x.bSaturationValue = 5;
                g_xCS.x.bPictureMode = PM_WARM;
                UpdateCommonSettings();
                m_iImageProcess[0] = 50;
                m_iImageProcess[1] = 50;
                m_iImageProcess[2] = 50;
            }
            else if(iPressed == BTN_ID_COOL)
            {
                g_xCS.x.bBrightValue = 5;
                g_xCS.x.bContrastValue = 5;
                g_xCS.x.bSaturationValue = 5;
                g_xCS.x.bPictureMode = PM_COOL;
                UpdateCommonSettings();
                m_iImageProcess[0] = 50;
                m_iImageProcess[1] = 50;
                m_iImageProcess[2] = 50;
            }
            else if(m_iSelectSlider >= 0)
            {
                int i = m_iSelectSlider;
                int iX = e->pos().x();
                m_iImageProcess[i] = 10 + (iX - SLIDER_LEFT) / (float)(SLIDER_RIGHT - SLIDER_LEFT) * 90;
                m_iImageProcess[i] = __max(10, m_iImageProcess[i]);
                m_iImageProcess[i] = __min(100, m_iImageProcess[i]);

                if(i == 0)
                    g_xCS.x.bBrightValue = m_iImageProcess[i] / 10;
                else if(i == 1)
                    g_xCS.x.bContrastValue = m_iImageProcess[i] / 10;
                else if(i == 2)
                    g_xCS.x.bSaturationValue = m_iImageProcess[i] / 10;
            }

            m_iSelectSlider = -1;
        }
        UpdateLCD();
    }
}

bool CamTestForm::event(QEvent* e)
{
    if(e->type() == EV_KEY_EVENT)
    {
        KeyEvent* pEvent = static_cast<KeyEvent*>(e);
        qDebug() << "CamTestForm:KeyEvent" << pEvent->m_iKeyID << pEvent->m_iEvType;
        if (pEvent->m_iKeyID == E_BTN_FUNC)
        {
            switch(pEvent->m_iEvType)
            {
            case KeyEvent::EV_CLICKED:
                m_fRunning = 0;
                break;
            case KeyEvent::EV_DOUBLE_CLICKED:
                break;
            case KeyEvent::EV_LONG_PRESSED:
                qApp->exit(QAPP_RET_OK);
                break;
            }
        }
    }    

    return QWidget::event(e);
}

void CamTestForm::InitSettings()
{
    m_iSettings = 0;    
    m_iImageProcess[0] = (float)g_xCS.x.bBrightValue * 100 / 10;
    m_iImageProcess[1] = (float)g_xCS.x.bContrastValue * 100 / 10;
    m_iImageProcess[2] = (float)g_xCS.x.bSaturationValue * 100 / 10;

    m_axProcessSlider[0] = QRect(0, 0, MAX_X, LCD_HEADER_HEIGHT + 116);
    m_axProcessSlider[1] = QRect(0, LCD_HEADER_HEIGHT + 116, MAX_X, 96);
    m_axProcessSlider[2] = QRect(0, LCD_HEADER_HEIGHT + 212, MAX_X, 96);
    m_iSelectSlider = -1;
}

void CamTestForm::ResetButtons()
{
    m_xMutex.lock();
    m_iBtnCount = 0;
    memset(m_axBtns, 0, sizeof(m_axBtns));
    m_xMutex.unlock();
}

void CamTestForm::AddButton(int iID, int iX1, int iY1, int iX2, int iY2, const char* szNormal, const char* szPress, unsigned int iNormalColor, int iPressColor, int iState)
{
    if(m_iBtnCount > MAX_BUTTON_CNT)
        return;

    BUTTON xBtn = { 0 };
    xBtn.iID = iID;
    xBtn.iX1 = iX1;
    xBtn.iY1 = iY1;
    xBtn.iX2 = iX2;
    xBtn.iY2 = iY2;

    if(szNormal)
        strcpy(xBtn.szNormal, szNormal);

    if(szPress)
        strcpy(xBtn.szPress, szPress);

    xBtn.iNormalColor = iNormalColor;
    xBtn.iPressColor = iPressColor;

    m_xMutex.lock();

    int iExist = -1;
    for(int i = 0; i < m_iBtnCount; i ++)
    {
        if(m_axBtns[i].iID == xBtn.iID)
        {
            iExist = i;
            break;
        }
    }

    if(iExist >= 0)
    {
        m_axBtns[iExist] = xBtn;
        m_xMutex.unlock();
        return;
    }

    xBtn.iState = iState;
    m_axBtns[m_iBtnCount] = xBtn;
    m_iBtnCount ++;
    m_xMutex.unlock();
}

void CamTestForm::UpdateLCD()
{
    QImage xImage((unsigned char*)LCDTask::m_piFBMem, MAX_X, MAX_Y, QImage::Format_ARGB32);

    QPainter painter;
    painter.begin(&xImage);

    LCDTask::LCD_MemClear(0);

#if 0
    if(m_iSettings == 1)
    {
        int iSlideWidth = SLIDER_RIGHT - SLIDER_LEFT;

        painter.fillRect(QRect(0, LCD_HEADER_HEIGHT, MAX_X, 320), QColor(0, 0, 0, 0x60));

        int aiIconY[3] = {LCD_HEADER_HEIGHT + 40, LCD_HEADER_HEIGHT + 136, LCD_HEADER_HEIGHT + 232};
        char* iconPaths[] = {":/icons/ic_brightness.png", ":/icons/ic_contrast.png", ":/icons/ic_saturation.png"};
        for(int i = 0; i < 3; i ++)
        {
            painter.drawImage(40, aiIconY[i], QImage(iconPaths[i]));

            painter.save();

            if(i == m_iSelectSlider)
            {
                painter.setPen(QColor(112, 146, 190));
                painter.setBrush(QColor(112, 146, 190));
            }
            else
            {
                painter.setPen(Qt::white);
                painter.setBrush(Qt::white);
            }

            int iCenterX = SLIDER_LEFT + (float)(m_iImageProcess[i] - 10) / 90 * iSlideWidth;

            painter.setRenderHint(QPainter::Antialiasing);
            painter.drawLine(QPoint(SLIDER_LEFT, aiIconY[i] + 24), QPoint(SLIDER_RIGHT, aiIconY[i] + 24));
            painter.drawRect(QRect(iCenterX - 12, aiIconY[i] + 24 - 20, 24, 40));

            painter.restore();
        }

//        LCDTask::LCD_Update(0, LCD_HEADER_HEIGHT, MAX_X, LCD_HEADER_HEIGHT + 320);
    }
    for(int i = 0; i < m_iBtnCount; i ++)
    {
        if(m_axBtns[i].iState == BTN_STATE_NONE)
        {
            painter.fillRect(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2 - m_axBtns[i].iX1 + 1, m_axBtns[i].iY2 - m_axBtns[i].iY1 + 1, QColor::fromRgba(m_axBtns[i].iNormalColor));
            if (strstr(m_axBtns[i].szNormal, ".png"))
            {
                QImage xImage(m_axBtns[i].szNormal);
//                unsigned char* pbData = xImage.bits();
//                for(int i = 0; i < xImage.width() * xImage.height(); i ++)
//                {
//                    pbData[i * 4] = pbData[i * 4 + 3];
//                    pbData[i * 4 + 1] = pbData[i * 4 + 3];
//                    pbData[i * 4 + 2] = pbData[i * 4 + 3];
//                }

                painter.drawImage(QRect(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2 - m_axBtns[i].iX1 + 1, m_axBtns[i].iY2 - m_axBtns[i].iY1 + 1), xImage);
            }
            else
                painter.drawText(QRect(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2 - m_axBtns[i].iX1 + 1, m_axBtns[i].iY2 - m_axBtns[i].iY1 + 1), QString::fromUtf8(m_axBtns[i].szNormal));
        }
        else
        {
            painter.fillRect(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2 - m_axBtns[i].iX1 + 1, m_axBtns[i].iY2 - m_axBtns[i].iY1 + 1, QColor::fromRgba(m_axBtns[i].iPressColor));
            if (strstr(m_axBtns[i].szPress, ".png"))
            {
                QImage xImage(m_axBtns[i].szPress);

                if (strcmp(m_axBtns[i].szPress, m_axBtns[i].szNormal) == 0)
                    painter.setOpacity(0.5);
                painter.drawImage(QRect(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2 - m_axBtns[i].iX1 + 1, m_axBtns[i].iY2 - m_axBtns[i].iY1 + 1), xImage);
                if (strcmp(m_axBtns[i].szPress, m_axBtns[i].szNormal) == 0)
                    painter.setOpacity(1.0);
            }
            else
                painter.drawText(QRect(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2 - m_axBtns[i].iX1 + 1, m_axBtns[i].iY2 - m_axBtns[i].iY1 + 1), QString::fromUtf8(m_axBtns[i].szPress));

        }

//        LCDTask::LCD_Update(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2, m_axBtns[i].iY2);
    }

    if(!m_bIsDevTest)
    {
        LCDTask* pLCDTask = new LCDTask(NULL);
        LCDTask::DrawMemFillRect(0, 0, MAX_X, LCD_HEADER_HEIGHT, C_Black);
        pLCDTask->LCD_DrawText(0, 0, MAX_X - 1, LCD_HEADER_HEIGHT - 1,
                     My_Camera_Test,
                     Canvas::C_TA_Center | Canvas::C_TA_Middle,
                     LCD_FOOTER_FONT_SIZE, C_White, g_HEADER_BG_COLOR[g_xCS.x.bTheme]);

        LCDTask::DrawMemFillRect(0, MAX_Y - LCD_FOOTER_HEIGHT, MAX_X, MAX_Y, C_Black);
        pLCDTask->LCD_DrawText(10, MAX_Y - LCD_FOOTER_HEIGHT, MAX_X - 20, MAX_Y - 1,
                     My_Str_Tap_To_Cancel,
                     Canvas::C_TA_Center | Canvas::C_TA_Middle,
                     LCD_FOOTER_FONT_SIZE, C_White, C_NoBack);
    }
#endif

    painter.end();
    LCDTask::LCD_Update();
}


int CamTestForm::CheckBtnState(QPoint pos, int mode)
{
    int iPressed = -1;
    for(int i = 0; i < m_iBtnCount; i ++)
    {
        QRect btnRect(m_axBtns[i].iX1, m_axBtns[i].iY1, m_axBtns[i].iX2 - m_axBtns[i].iX1 + 1, m_axBtns[i].iY2 - m_axBtns[i].iY1 + 1);
        if(btnRect.contains(pos))
        {
            iPressed = m_axBtns[i].iID;
            if(mode == 2)
            {
                m_axBtns[i].iState = BTN_STATE_NONE;
            }
            else
                m_axBtns[i].iState = BTN_STATE_PRESSED;
        }
        else
            m_axBtns[i].iState = BTN_STATE_NONE;
    }

    printf("m_iBtnCount = %d, pressed = %d\n", m_iBtnCount, iPressed);

    return iPressed;
}
