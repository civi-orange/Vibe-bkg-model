#pragma once
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/core.hpp>
#include <opencv.hpp>
#include <string.h>

//using namespace cv;
using namespace std;

//其他事件
struct Track_Event
{
	int nType;
	int nLabel;
	double dArea;

};

//序号，目标在疑似目标中编号
struct Target_index
{
	int nIndex;
	int nLabel;
	Target_index()
	{
		nIndex = 0;
		nLabel = 0;
	}
};

//目标特征
struct Target_Feature
{
	//其他特征N个，未定义
	Target_Feature(){}

};

//目标信息
struct Target_info
{
	vector<Target_Feature> vctFature;
	vector<cv::Point> vctposition;
	vector<double> vctWidth;
	vector<double> vctHeight;
	vector<double> vctRegionArea;
	vector<double> vctLength;
    vector<double> vctw_hrate;
	int nframeNum;
	Target_info()
	{
		nframeNum = 0;
		vctFature.clear();
		vctposition.clear();
		vctRegionArea.clear();
		vctWidth.clear();
		vctHeight.clear();
        vctw_hrate.clear();
	}
};

//跟踪信息
struct Tracking_info
{
	vector<Target_Feature> vctfeature;
	//外接矩信息
	vector<cv::Point> vctposition;
	double dWidth;
	double dHeight;
	//跟踪目标数量
	int nLabel;
	//帧数
	int nFrNum;
	//跟踪方法
	bool bLocalTrack;
	bool bBlindTrack;
	bool bBayesTrack;
	bool bMemoryTrack;
	bool bComplexTrack;



	Tracking_info() {
		vctfeature.clear();
		vctposition.clear();
		dWidth = 0;
		dHeight = 0;
		nFrNum = 0;
		nLabel = 0;
		bLocalTrack = false;
		bBlindTrack = false;
		bBayesTrack = false;
		bMemoryTrack = false;
		bComplexTrack = false;
	}

};

//跟踪参数
class Detect_Parameter
{
public:
	Detect_Parameter();

	~Detect_Parameter();

	int Clone(Detect_Parameter& src);

public:
	/*****************检测目标参数******************/


	bool bDetectFlag;//是否检测目标
	bool bTargetColor;//目标颜色

    int bbkg_type;//目标分离方法选择 0：纯天空背景，直接分离目标；1：地空结合背景，分离出天空目标； 2：其他新增办法

	int	bmethod;//处理方法：0：默认关联外推方式；1：直接做二值化做阈值分割；其他更多待完善

	int m_nDilate_1; //膨胀参数1
	int m_nErosion_1; //腐蚀参数1——分离目标
	int m_nDilate_2; //膨胀参数2
	int m_nErosion_2; //腐蚀参数2
	int m_nOpen;//开操作1——寻找天空
	int m_nClose;//闭操作1——去噪声（填充目标）

	cv::Rect ImgROI;//处理区域
	double m_graythrd;//分割阈值
	double m_Areathrdmin;//面积阈值
	double m_Areathrdmax;//面积阈值
	double m_Lengththrd;

	/****************定位目标参数*******************/
	//初始目标信息
	vector<Target_info> m_vctOriginFrameinfo;

	//迭代目标信息（保存当前帧向前推N帧，N为MN法检测长度
	vector<Target_info> m_vctSaveFrameinfo;

	//新生目标信息
	vector<Target_info> m_vctBornFrameinfo;

	//跟踪目标信息（叠加记录
	vector<Tracking_info> m_vctTrackingFrameinfo;

	//突发事件
	vector<Track_Event> m_vctEvent;

	//目标对信息()  最后一个为目标最后位置
	vector<Target_index> m_vctpairs;

	//上一帧无目标帧编号
	vector<int> m_vctEmptyFrameCount;

	//当前帧的信息
	Tracking_info  m_curFrameInfo;

	//稳定关联
	bool stable_detect_;
	
	//找到目标标志
	bool bFindTarget;
	
	//帧数
	int m_nFrNum;
	
	//没有目标帧---帧号
	int m_nEmptyFrameThrd;
	
	//检测初始
	double  dTS;

	//最大最小速度限制
	double dRateMax_x;
	double dRateMax_y;
	double dRateMin_x;
	double dRateMin_y;

	//系统模型,MN,卡尔曼模型
	cv::Mat OriF;//4*4
	cv::Mat OriH;//2*4
	cv::Mat OriGama;//4*1
	//扫描时间
	double m_OriTs;

	//卡尔曼滤波门限值
	double m_OriThreshold;

	//航迹关联起始帧数长度,关联帧数长度
	int m_nOriFrameLength;

	//外推长度
	int m_nPredLendth;
	int _target_detect_thrd = 20;//稳定检测目标阈值

	//删除航迹丢失帧数长度 5 连续丢失目标帧数
	int m_nEraseFrameLength;

};

void showImage(cv::String str, cv::Mat img);


