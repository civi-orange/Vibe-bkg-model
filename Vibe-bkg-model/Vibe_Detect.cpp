#include "stdafx.h"
#include "Vibe_Detect.h"


Vibe_Detect::Vibe_Detect()
{
	vctTar_center.clear();
	target_count_valib = 0;
	ROI_is_exist = false;

	//test
	vct_frame.clear();
	vct_info.clear();
	
}


Vibe_Detect::~Vibe_Detect()
{
	libvibeModel_Sequential_Free(model);
}


void Vibe_Detect::Vibe_init(cv::Mat& frame)
{		
	double t3 = clock();
	FindROI(frame, _ROI_);
	double t4 = clock();
	std::cout << "ROI time:" << t4 - t3 << std::endl;
	ROI_is_exist = true;
	cv::Mat Img_roi;
	frame.copyTo(Img_roi, _ROI_);

	
	cvtColor(Img_roi, Img_roi, CV_BGR2GRAY);
	segmentationMap = cv::Mat(frame.rows, frame.cols, CV_8UC1);
	
	double t1 = clock();
	model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
	libvibeModel_Sequential_AllocInit_8u_C1R(model, Img_roi.data, Img_roi.cols, Img_roi.rows);
	double t2 = clock();
	std::cout<< "vibe init time:" << t2 - t1 << std::endl;

	return;
}

/*阈值分割方法*//*输入：起始坐标，分割阈值差*/
void Vibe_Detect::RegionGrow(cv::Mat& src, cv::Mat& matDst, cv::Point2i pt, int th)
{
	cv::Point2i ptGrowing;						//待生长点位置
	int nGrowLable = 0;								//标记是否生长过
	int nSrcValue = 127;								//生长起点灰度值，起始灰度定为中值灰度
	int nCurValue = 0;								//当前生长点灰度值
	matDst = cv::Mat::zeros(src.size(), CV_8UC1);	//创建一个空白区域，填充为黑色											
	int DIR[8][2] = { { -1, -1 },{ 0, -1 },{ 1, -1 },{ 1, 0 },{ 1, 1 },{ 0, 1 },{ -1, 1 },{ -1, 0 } };//生长方向顺序数据//loop
	std::vector<cv::Point2i> vcGrowPt;						//生长点栈
	vcGrowPt.push_back(pt);							//将生长点压入栈中
	matDst.at<uchar>(pt.y, pt.x) = 255;				//标记生长点
	nSrcValue = src.at<uchar>(pt.y, pt.x);		  //记录生长点的灰度值，不以图像为基准，以中值为基准							
	while (!vcGrowPt.empty())						//生长栈不为空则生长
	{
		pt = vcGrowPt.back();						//取出一个生长点
		vcGrowPt.pop_back();
		//分别对八个方向上的点进行生长
		for (int i = 0; i < 8; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//检查是否是边缘点//loop
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;
			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);		//当前待生长点的灰度值
			if (nGrowLable == 0)					//如果标记点还没有被生长
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				//if (abs(nSrcValue - nCurValue) < th)					//在阈值范围内则生长
				if (nSrcValue - nCurValue < th)	//修改为大于阈值？
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;		//标记为白色
					vcGrowPt.push_back(ptGrowing);					//将下一个生长点压入栈中
				}
			}
		}
	}

}

//分离地空区域
int Vibe_Detect::FindROI(cv::Mat& Img, cv::Mat& Img_ROI)
{
	cv::Mat image_shift;
	cv::pyrMeanShiftFiltering(Img, image_shift, 50, 50, 1);//半径越大效果越好，时间越长
	cv::Mat gray_src;
	cv::cvtColor(image_shift, gray_src, CV_BGR2GRAY);
	cv::Mat im_out;
	RegionGrow(gray_src, im_out, cv::Point(100, 100), 30);
	cv::imshow("im_out", im_out);

	cv::Mat im_filter_1;
	cv::Mat f1 = getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
	morphologyEx(im_out, im_filter_1, cv::MORPH_ERODE, f1);
	cv::Mat im_filter_2;
	cv::Mat f2 = getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
	morphologyEx(im_filter_1, im_filter_2, cv::MORPH_OPEN, f2);

	vector<vector<cv::Point>> contours;
	vector<cv::Vec4i> hierarchy;
	findContours(im_filter_2, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, cv::Point());
	cv::Mat mask_ROI(Img.size(), CV_8U, cv::Scalar(0));
	drawContours(mask_ROI, contours, -1, cv::Scalar(255), CV_FILLED);
	showImage("mask", mask_ROI);
	Img_ROI = mask_ROI.clone();

	return true;
}

//背景建模与更新，输出前景，图像分割
int Vibe_Detect::bkgVibeModel(cv::Mat& Img, cv::Mat& segment_out, int frame_num)
{		
	double t1 = clock();
	libvibeModel_Sequential_Segmentation_8u_C1R(model, Img.data, segmentationMap.data);
	libvibeModel_Sequential_Update_8u_C1R(model, Img.data, segmentationMap.data);
	double t2 = clock();
	std::cout << "vibe update time:" << t2 - t1 << std::endl;

	/* Shows the current frame and the segmentation map. */
	showImage("Segmentation by ViBe", segmentationMap);//foreground picture
	//cv::Mat im_f0;
	//cv::Mat f0 = getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
	//morphologyEx(segmentationMap, im_f0, cv::MORPH_ERODE, f0);
	//showImage("im_f0", im_f0);
	//cv::Mat im_f1;
	//cv::Mat f1 = getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));
	//morphologyEx(im_f0, im_f1, cv::MORPH_DILATE, f1);
	//showImage("im_f1", im_f1);
	cv::Mat im_f2;
	cv::Mat f2 = getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
	morphologyEx(segmentationMap, im_f2, cv::MORPH_CLOSE, f2);
	showImage("im_f2", im_f2);
	segment_out = im_f2.clone();
	return true;
}

//目标信息统计
int Vibe_Detect::findTarget(cv::Mat& targetMat, Target_info& curInfo, int frame_num)
{
	if (targetMat.empty())
	{
		return false;
	}
	vector<vector<cv::Point>> vctContours;
	vector<cv::Vec4i>  vctHierarchy;//后、前、子、父轮廓的编号与轮廓一一对应
	vctContours.clear();
	vctHierarchy.clear();
	cv::Moments moment;
	cv::Point p;
	cv::Point pt[255];
	findContours(targetMat, vctContours, vctHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	if (vctContours.size() > 0 && vctContours.size() < 30)
	{
		vector<cv::Rect> boundRect(vctContours.size());  //定义外接矩形集合
		vector<cv::RotatedRect> box(vctContours.size()); //定义最小外接矩形集合
		cv::Point2f rect[4];//四个顶点
		for (int kk = 0; kk < int(vctContours.size()); kk++)
		{
			double dLength = 0;
			double dArea = 0;
			dArea = contourArea(vctContours[kk], false);//目标面积
			cv::Mat mask_ann = cv::Mat(targetMat.rows, targetMat.cols, CV_8UC1, cv::Scalar(0));
			cv::drawContours(mask_ann, vctContours, kk, cv::Scalar(255), cv::FILLED, 8);
			int nCount_White = 0;//白
			int nCount_Black = 0;//黑
			cv::Mat_<uchar>::iterator itor = mask_ann.begin<uchar>();
			cv::Mat_<uchar>::iterator itorEnd = mask_ann.end<uchar>();
			for (; itor != itorEnd; ++itor)	//通过迭代器访问图像的像素点
			{
				if ((*itor) > 0)
				{
					nCount_White += 1;//白：像素值 ptr:255
				}
				else
				{
					nCount_Black += 1;//黑：像素值 ptr:0
				}
			}
			dArea = nCount_White;
			dLength = arcLength(vctContours[kk], false);// 轮廓长度
			double w_hrate = 0;//长宽比
			box[kk] = minAreaRect(cv::Mat(vctContours[kk]));  //计算每个轮廓最小外接矩形
			boundRect[kk] = boundingRect(cv::Mat(vctContours[kk]));
			box[kk].points(rect);  //把最小外接矩形四个端点复制给rect数组
			cv::Mat temp(vctContours.at(kk));
			moment = moments(temp, false);
			if (moment.m00 != 0)//除数不能为0
			{
				pt[kk].x = cvRound(moment.m10 / moment.m00);//计算重心横坐标
				pt[kk].y = cvRound(moment.m01 / moment.m00);//计算重心纵坐标
			}
			p = cv::Point(pt[kk].x, pt[kk].y);//重心坐

			if (boundRect[kk].width > boundRect[kk].height)
			{
				w_hrate = (double)boundRect[kk].width / boundRect[kk].height;
			}
			else
			{
				w_hrate = (double)boundRect[kk].height / boundRect[kk].width;
			}
			if (dArea > 2 && dArea < 2000 && w_hrate < 15 && boundRect[kk].area() < 4000/* && p.y - 30 > 0 && frame.rows - p.y > 30*/)
			{
				curInfo.vctWidth.push_back(boundRect[kk].width);
				curInfo.vctHeight.push_back(boundRect[kk].height);
				curInfo.vctRegionArea.push_back(dArea);//面积
				curInfo.vctw_hrate.push_back(w_hrate);//长宽比
				curInfo.vctLength.push_back(dLength);//长度
				curInfo.vctposition.push_back(p);//重心坐标
				curInfo.nframeNum = frame_num;
			}
		}
	}

	return true;
}

//目标关联
int Vibe_Detect::LinkFunc(Target_info& tar_Info, Detect_Parameter& parameter)
{
	int nRet = true;

	int nFrNum = parameter.m_nFrNum;

	MNLogic m_mnlogic;


	if (tar_Info.vctposition.size() == 0)
	{
		if (parameter.m_vctOriginFrameinfo.size() > 0)
		{
			parameter.m_vctEmptyFrameCount.push_back(nFrNum);

			if ((int)parameter.m_vctEmptyFrameCount.size() > parameter.m_nEmptyFrameThrd)
			{
				if (parameter.m_vctEmptyFrameCount.back() - parameter.m_vctEmptyFrameCount.front() + 1 > parameter.m_nEmptyFrameThrd)
				{
					parameter.m_vctEmptyFrameCount.clear();
					parameter.bFindTarget = false;
					nRet = false;
					return nRet;
				}
			}
			else if (parameter.m_vctEmptyFrameCount.size() > 0 && parameter.m_vctEmptyFrameCount.size() <= 2)
			{
				//如果不是连续为空，则重新记录
				if (parameter.m_vctEmptyFrameCount.back() != nFrNum)
				{
					parameter.m_vctEmptyFrameCount.clear();
				}
				return false;
			}
			else
			{
				return false;
			}
		}
		else
		{
			return false;
		}
	}
	else if (tar_Info.vctposition.size() > 0 && tar_Info.vctposition.size() < 15)
	{
		if (parameter.m_vctEmptyFrameCount.size() > 0 && (int)parameter.m_vctEmptyFrameCount.size() <= parameter.m_nEmptyFrameThrd)
		{
			//如果不是连续为空，则重新记录
			if (parameter.m_vctEmptyFrameCount.back() != nFrNum)
			{
				parameter.m_vctEmptyFrameCount.clear();
			}
		}
		parameter.m_vctOriginFrameinfo.push_back(tar_Info);

		if ((int)parameter.m_vctSaveFrameinfo.size() < parameter.m_nOriFrameLength + 2)
		{
			parameter.m_vctSaveFrameinfo.push_back(tar_Info);
		}
		else
		{
			parameter.m_vctSaveFrameinfo.erase(parameter.m_vctSaveFrameinfo.begin());
			parameter.m_vctSaveFrameinfo.push_back(tar_Info);
		}


		//满足外推条件

		if ((int)parameter.m_vctOriginFrameinfo.size() > parameter.m_nOriFrameLength + 2)
		{
			if ((int)parameter.m_vctOriginFrameinfo.size() <= parameter.m_nOriFrameLength + 2 + parameter.m_nPredLendth)
			{
				nRet = m_mnlogic.inputSample(parameter, nFrNum);
			}
			else
			{
				parameter.m_vctOriginFrameinfo.clear();
				parameter.m_vctOriginFrameinfo.assign(parameter.m_vctSaveFrameinfo.begin(), parameter.m_vctSaveFrameinfo.end());
				nRet = m_mnlogic.inputSample(parameter, nFrNum);
			}
			return nRet;
		}
		else
		{
			nRet = false;
			return nRet;
		}
	}
	else if (tar_Info.vctposition.size() > 15)
	{
		nRet = false;
		return nRet;
		//目标杂乱
	}

}

//外部接口函数
int Vibe_Detect::DetectProc(cv::Mat& Img, Detect_Parameter& parameter, cv::Rect& target_box, int frame_num)
{
	int nRet = false;

	if (Img.empty()){
		return false;
	}

	cv::Mat Img_ROI, image_roi_gray, fontMat;

	Img.copyTo(Img_ROI, _ROI_);

	cvtColor(Img_ROI, image_roi_gray, CV_BGR2GRAY);

	//当前帧号
	parameter.m_nFrNum = frame_num;

	Target_info curInfo;

 	if (ROI_is_exist == true)
	{

		nRet = bkgVibeModel(image_roi_gray, fontMat, frame_num);

		nRet = findTarget(fontMat, curInfo, frame_num);

		nRet = LinkFunc(curInfo, parameter);

		//关联目标信息索引
		if (nRet == true && parameter.bFindTarget == true)
		{
			int label = parameter.m_vctpairs.back().nLabel;
			int index = parameter.m_vctpairs.back().nIndex;

			vctTar_center.push_back(parameter.m_vctOriginFrameinfo[index - 1].vctposition[label]);
			vctTar_center.back().x += parameter.ImgROI.x;
			vctTar_center.back().y += parameter.ImgROI.y;

			if (abs(vctTar_center.back().x - vctTar_center[vctTar_center.size() - 10].x) > 0 ||
				abs(vctTar_center.back().y - vctTar_center[vctTar_center.size() - 10].y) > 0)
			{
				target_box.width = parameter.m_vctOriginFrameinfo[index - 1].vctWidth[label];
				target_box.height = parameter.m_vctOriginFrameinfo[index - 1].vctHeight[label];
				target_box.width *= 3;
				target_box.height *= 5;
				int x = vctTar_center.back().x - target_box.width / 2;
				int y = vctTar_center.back().y - target_box.height / 2;
				if (x > 0) { target_box.x = x; }
				else { target_box.x = 0; }
				if (y > 0) { target_box.y = y; }
				else { target_box.y = 0; }

				///关于目标框越界的问题 只考虑开始位置，结束位置默认不越界。结果查看
				cv::rectangle(Img, target_box, cv::Scalar(0, 255, 0), 1, 8);
				//cv::circle(img, vctTar_center.back(), 1, cv::Scalar(255, 255, 0), 1, 8);
				target_count_valib++;
				imshow("target0", Img);
				cvWaitKey(1);
			}
			else
			{
				target_count_valib = 0;
				//cv::rectangle(img, target_box, cv::Scalar(0, 0, 255), 3, 8);
			}

		}
		else
		{
			//std::cout << "frame number: " << parameter.m_vctOriginFrameinfo.back().nframeNum << std::endl;
			target_count_valib = 0;
			cv::rectangle(Img, target_box, cv::Scalar(0, 0, 255), 1, 8);
		}
		showImage("target", Img);
		std::cout << "target count: " << target_count_valib << std::endl;
		if (target_count_valib >= parameter._target_detect_thrd)
		{
			parameter.stable_detect_ = true;
			parameter.bDetectFlag = false;///集成算法时，去掉本行注释；注释本行是为了单独测试本算法,将if 判断修改成 >=
			std::cout << "------------detect success--------------" << std::endl;
			imshow("target1", Img);
			cvWaitKey();
		}
	
		return nRet;
	}
	else
	{
		std::cout << "vibe need init!" << std::endl;
		return false;
	}

}

//need image is cv_8uc1
int Vibe_Detect::FrameDiff(vector<Target_info>& vct_info)
{
	vct_info.clear();
	if (vct_frame.size() == 0)
	{
		return false;
	}
	vector<cv::Mat> vct_frame_diff;

	for (int i = (int)vct_frame.size() - 1; i > 0; i--)
	{
		cv::Mat diff_temp(vct_frame[0].rows, vct_frame[0].cols, CV_8UC1);
		diff_temp = vct_frame[i - 1] - vct_frame.back();

		vct_frame_diff.push_back(diff_temp);
		//showImage("diff_temp", diff_temp);

	}

	for (int j = 0; j < (int)vct_frame_diff.size(); j++)
	{
		cv::Mat temp;
		Target_info info;
		vct_frame_diff[j].copyTo(temp);
		cv::threshold(temp, temp, 20, 255, CV_THRESH_BINARY);
		//showImage("temp", temp);
		findTarget(temp, info, j);
		vct_info.push_back(info);
	}
	vct_frame_diff.clear();
	return true;
}

int Vibe_Detect::detectdiff(cv::Mat& frame)
{

	cv::Mat frameGray;
	cv::cvtColor(frame, frameGray, CV_BGR2GRAY);

	if (vct_frame.size() < 4)
	{
		vct_frame.push_back(frameGray);
	}

	else
	{
		vct_frame.erase(vct_frame.begin());
		vct_frame.push_back(frameGray);
		FrameDiff(vct_info);
		for (int i = 0; i < (int)vct_info.size(); i++)
		{
			cv::Mat framedd;
			framedd = frame.clone();
			if (vct_info[i].vctposition.size() != 0)
			{
				for (int k = 0; k < (int)vct_info[i].vctposition.size(); k++)
				{
					cv::circle(framedd, vct_info[i].vctposition[k], 1, cv::Scalar(0, 0, 255), 8);
				}
			}
			//showImage(std::to_string(i),framedd);
		}
	}

	return true;
}
