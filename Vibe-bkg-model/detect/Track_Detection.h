#pragma once
#include "Detect_Parameter.h"
#include "MNLogic.h"
#include <math.h>

class Track_Detection
{
public:
	Track_Detection();

	~Track_Detection();

	int Initialize();

	int SetParam(Detect_Parameter& param);

	int InputSample(cv::Mat* ImageIn, int nFrNum);

	//检测函数入口，传入图像，帧数，处理方式，返回目标的最小外接矩形
    int DetectProc(cv::Mat& img, cv::Rect& target_box, int nFrNum);

	void GetParameter(Detect_Parameter& param);

	int Uninitialize();

public:

	Detect_Parameter* m_Param;

	vector<cv::Point> vctTar_center;

	int target_count_valib;


};

