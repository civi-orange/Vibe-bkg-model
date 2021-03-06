#include "Detect_Parameter.h"
#define TS 10 //航迹初始

Detect_Parameter::Detect_Parameter()
{
    bbkg_type = false; //目标分离方法选择0：纯天空背景，直接分离目标；1：地空结合背景，分离出天空目标； 2...：其他新增办法
	bDetectFlag = false;
	bTargetColor = false;

	bmethod = 0;////处理方法：0：默认关联外推方式；1：直接做二值化做阈值分割；其他更多待完善

	m_nDilate_1 = 7; //膨胀参数1
	m_nErosion_1 = 7;//腐蚀参数1
	m_nDilate_2 = 7;//膨胀参数2
	m_nErosion_2 = 7;//腐蚀参数2
	m_nOpen = 16;
	m_nClose = 5;

	ImgROI = cv::Rect(0,0,0,0);//处理区域
	m_graythrd = 29;//分割阈值
	m_Areathrdmin = 5;//面积阈值
	m_Areathrdmax = 300;
	m_Lengththrd = 1000;

	/*********************/
	dTS = 1; //扫描时间
	dRateMax_x = 20;
	dRateMax_y = 20;
	dRateMin_x = -20;
	dRateMin_y = -20;
	OriF = (cv::Mat_<double>(4, 4) << 1, 0, TS, 0, 0, 1, 0, TS, 0, 0, 1, 0, 0, 0, 0, 1);
	OriH = (cv::Mat_<double>(2, 4) << 1, 0, 0, 0, 0, 1, 0, 0);
	OriGama = (cv::Mat_<double>(4, 1) << TS*TS / 2, TS*TS / 2, TS, TS);

	m_nFrNum = 1;
	
	m_OriThreshold = 0.1;
    m_OriTs = TS;

	m_nOriFrameLength = 5;//起始帧长度
    m_nPredLendth = 5;//外推帧长度
	m_nEmptyFrameThrd = 3; //连续n帧为空，重置限制
	m_nEraseFrameLength = 5;
	_target_detect_thrd = 20;//稳定检测目标阈值

	bFindTarget = false;
	stable_detect_ = false;

	m_vctOriginFrameinfo.clear();
	m_vctSaveFrameinfo.clear();
	m_vctBornFrameinfo.clear();
	m_vctEvent.clear();
	m_vctpairs.clear();
	m_vctTrackingFrameinfo.clear();
	m_vctEmptyFrameCount.clear();


}

Detect_Parameter::~Detect_Parameter()
{

}

int Detect_Parameter::Clone(Detect_Parameter& src)
{
	int nRet = true;

	src.bTargetColor = bTargetColor;
	src.bDetectFlag = bDetectFlag;
    src.bbkg_type = bbkg_type;
	src.bmethod = bmethod;

	src.m_vctOriginFrameinfo.assign(m_vctOriginFrameinfo.begin(), m_vctOriginFrameinfo.end());
	src.m_vctSaveFrameinfo.assign(m_vctSaveFrameinfo.begin(), m_vctSaveFrameinfo.end());
	src.m_vctBornFrameinfo.assign(m_vctBornFrameinfo.begin(), m_vctBornFrameinfo.end());
	src.m_vctTrackingFrameinfo.assign(m_vctTrackingFrameinfo.begin(), m_vctTrackingFrameinfo.end());
	src.m_vctEmptyFrameCount.assign(m_vctEmptyFrameCount.begin(), m_vctEmptyFrameCount.end());
	src.m_vctEvent.assign(m_vctEvent.begin(), m_vctEvent.end());
	src.m_vctpairs.assign(m_vctpairs.begin(), m_vctpairs.end());
	src.m_curFrameInfo = m_curFrameInfo;

	src.bFindTarget = bFindTarget;
	src.stable_detect_ = stable_detect_;
	src.m_nFrNum = m_nFrNum;
	src.m_nEmptyFrameThrd = m_nEmptyFrameThrd;
	src.dRateMax_x = dRateMax_x;
	src.dRateMax_y = dRateMax_y;
	src.dRateMin_x = dRateMin_x;
	src.dRateMin_y = dRateMin_y;
	src.OriF = OriF;
	src.OriH = OriH;
	src.OriGama = OriGama;
	src.m_OriTs = m_OriTs;
	src.m_OriThreshold = m_OriThreshold;
	src.m_nOriFrameLength = m_nOriFrameLength;
	src.m_nPredLendth = m_nPredLendth;
	src._target_detect_thrd = _target_detect_thrd;
	src.m_nEraseFrameLength = m_nEraseFrameLength;

	src.m_nDilate_1 = m_nDilate_1; //膨胀参数1
	src.m_nErosion_1 = m_nErosion_1; //腐蚀参数1
	src.m_nDilate_2 = m_nDilate_2;//膨胀参数2
	src.m_nErosion_2 = m_nErosion_2; //腐蚀参数2
	src.m_nOpen = m_nOpen;
	src.m_nClose = m_nClose;

	src.ImgROI = ImgROI;//处理区域
	src.m_graythrd = m_graythrd;//分割阈值
	src.m_Areathrdmin = m_Areathrdmin;//面积阈值
	src.m_Areathrdmax = m_Areathrdmax;// 面积阈值
	src.m_Lengththrd = m_Lengththrd;

	return nRet;

}

//查看图像，debug使用
void showImage(cv::String str, cv::Mat img)
{
	imshow(str, img);
	cvWaitKey(1);
	return;
}

