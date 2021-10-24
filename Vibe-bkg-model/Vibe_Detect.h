#pragma once
#include "Detect_Parameter.h"
#include "MNLogic.h"
#include <vibe-background-sequential.h>
class Vibe_Detect
{
public:
	Vibe_Detect();
	~Vibe_Detect();
	
	void Vibe_init(cv::Mat& frame);

	void RegionGrow(cv::Mat& src, cv::Mat& matDst, cv::Point2i pt, int th);

	int FindROI(cv::Mat& Img, cv::Mat& Img_ROI);

	int bkgVibeModel(cv::Mat& Img, cv::Mat& segmentationMap, int frame_num);

	int findTarget(cv::Mat& targetMat, Target_info& curInfo, int frame_num);

	int LinkFunc(Target_info& tar_Info, Detect_Parameter& parameter);

	int DetectProc(cv::Mat& Img, Detect_Parameter& parameter, cv::Rect& target_box, int frame_num);

	int FrameDiff(vector<Target_info>& vct_info);

	int detectdiff(cv::Mat& frame/*, vector<cv::Rect>& target_box*/);

public:

	cv::Mat _ROI_; //ºÏ≤‚«¯”Ú
	bool ROI_is_exist;
	int target_count_valib;
	vector<cv::Point> vctTar_center;
	
	cv::Mat segmentationMap;
	vibeModel_Sequential_t *model = NULL;

	//test
	vector<cv::Mat> vct_frame;
	vector<Target_info> vct_info;

};

