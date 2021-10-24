#include <opencv2/opencv.hpp>
#include <iostream>
#include <vibe-background-sequential.h>
#include "Vibe_Detect.h"
using namespace std;
using namespace cv;

/*��ֵ�ָ��*//*���룺��ʼ���꣬�ָ���ֵ��*/
void RegionGrow(cv::Mat& src, cv::Mat& matDst, cv::Point2i pt, int th)
{
	cv::Point2i ptGrowing;						//��������λ��
	int nGrowLable = 0;								//����Ƿ�������
	int nSrcValue = 128;								//�������Ҷ�ֵ����ʼ�Ҷȶ�Ϊ��ֵ�Ҷ�
	int nCurValue = 0;								//��ǰ������Ҷ�ֵ
	matDst = cv::Mat::zeros(src.size(), CV_8UC1);	//����һ���հ��������Ϊ��ɫ											
	int DIR[8][2] = { { -1, -1 },{ 0, -1 },{ 1, -1 },{ 1, 0 },{ 1, 1 },{ 0, 1 },{ -1, 1 },{ -1, 0 } };//��������˳������//loop
	std::vector<cv::Point2i> vcGrowPt;						//������ջ
	vcGrowPt.push_back(pt);							//��������ѹ��ջ��
	matDst.at<uchar>(pt.y, pt.x) = 255;				//���������
	//nSrcValue = src.at<uchar>(pt.y, pt.x);			//��¼������ĻҶ�ֵ������ͼ��Ϊ��׼������ֵΪ��׼							
	while (!vcGrowPt.empty())						//����ջ��Ϊ��������
	{
		pt = vcGrowPt.back();						//ȡ��һ��������
		vcGrowPt.pop_back();
		//�ֱ�԰˸������ϵĵ��������
		for (int i = 0; i < 8; ++i)
		{
			ptGrowing.x = pt.x + DIR[i][0];
			ptGrowing.y = pt.y + DIR[i][1];
			//����Ƿ��Ǳ�Ե��//loop
			if (ptGrowing.x < 0 || ptGrowing.y < 0 || ptGrowing.x >(src.cols - 1) || (ptGrowing.y > src.rows - 1))
				continue;
			nGrowLable = matDst.at<uchar>(ptGrowing.y, ptGrowing.x);		//��ǰ��������ĻҶ�ֵ
			if (nGrowLable == 0)					//�����ǵ㻹û�б�����
			{
				nCurValue = src.at<uchar>(ptGrowing.y, ptGrowing.x);
				//if (abs(nSrcValue - nCurValue) < th)					//����ֵ��Χ��������
				if (nSrcValue - nCurValue < th)	//�޸�Ϊ������ֵ��
				{
					matDst.at<uchar>(ptGrowing.y, ptGrowing.x) = 255;		//���Ϊ��ɫ
					vcGrowPt.push_back(ptGrowing);					//����һ��������ѹ��ջ��
				}
			}
		}
	}

}

int main()
{
	cv::Mat frame;
	int frame_num = 0;
	//cv::VideoCapture cap("E:/target track/mianyangvideo/sample1.mp4");
	//cv::VideoCapture cap("E:/target track/��Ƶ/�ֻ���Ƶ/�޸���.mp4");
	cv::VideoCapture cap("C:/Users/lenovo/Desktop/cxd.mp4");


	if (!cap.isOpened())
	{
		return 0;
	}

	Detect_Parameter parameter;
	parameter.dTS = 10;
	parameter.m_OriThreshold = 0.01;

	cv::Rect t_box;

	Vibe_Detect detector;
	int b_vibe_init = false;
	while (cap.read(frame))
	{
		frame_num++;

		if (frame.cols > 2000)
		{
			cv::resize(frame, frame, cv::Size(1920, 1080));
			cv::resize(frame, frame, cv::Size(frame.cols / 2 ,frame.rows / 2));
		}

		//showImage("frame_all", frame);

		if ( b_vibe_init == false){
			detector.Vibe_init(frame);
			b_vibe_init = true;
			frame_num = 0;
		}
		else {
			detector.DetectProc(frame, parameter, t_box, frame_num);
		}
		
	}



	return 1;
}

int main_model()
{
	double t1, t2, t3, t4;
	vibeModel_Sequential_t *model = NULL;
	cv::Mat segmentationMap;

	cv::Mat image;
	int frame_num = 0;
	

	cv::VideoCapture cap("E:/target track/mianyangvideo/label1.mp4");
	//cv::VideoCapture cap("E:/target track/��Ƶ/�ֻ���Ƶ/�޸���.mp4");

	if (!cap.isOpened())
	{
		return 0;
	}
	
	cv::Mat _ROI_;
	while (cap.read(image))
	{

		cv::Mat image_roi, image_roi_gray;
		frame_num++;
		cv::imshow("Image", image);
		if (frame_num == 1)
		{
			if (image.cols > 2300)//temp
			{
				cv::resize(image, image, cv::Size(image.cols / 4, image.rows / 4));
			}

			cv::Mat image_shift;
			cv::pyrMeanShiftFiltering(image, image_shift, 50, 50, 1);//�뾶Խ��Ч��Խ�ã�ʱ��Խ��
			cv::imshow("image_shift", image_shift);
			cv::Mat gray_src;
			cv::cvtColor(image_shift, gray_src, CV_BGR2GRAY);
			cv::imshow("gray_src", gray_src);
			//cv::imwrite("E:/g.bmp", gray_src);
						
			cv::Mat im_out;
			RegionGrow(gray_src, im_out, cv::Point(100, 100), 85);
			cv::imshow("im_out", im_out);
			cv::Mat im_filter_1;
			cv::Mat f1 = getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
			morphologyEx(im_out, im_filter_1, cv::MORPH_ERODE, f1);
			cv::imshow("im_filter_1", im_filter_1);
			cv::Mat im_filter_2;
			cv::Mat f2 = getStructuringElement(cv::MORPH_RECT, cv::Size(21, 21));
			morphologyEx(im_filter_1, im_filter_2, cv::MORPH_OPEN, f2);
			cv::imshow("im_filter_2", im_filter_2);

			vector<vector<Point>> contours;
			vector<Vec4i> hierarchy;
			findContours(im_filter_2, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point());
			cv::Mat mask_ROI(image.size(), CV_8U, cv::Scalar(0));
			drawContours(mask_ROI, contours, -1, cv::Scalar(255), CV_FILLED);
			cv::imshow("mask", mask_ROI);
			_ROI_ = mask_ROI.clone();//��������
		}

		/*----------------------�ָ���--------------------*/
		image.copyTo(image_roi, _ROI_);
		cvtColor(image_roi, image_roi_gray, CV_BGR2GRAY);
		cv::Mat image_roi_bot, image_roi_bot_out;
		cv::Mat bothat = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		morphologyEx(image_roi_gray, image_roi_bot, cv::MORPH_BLACKHAT, bothat);	
		cv::imshow("image_roi_bot", image_roi_bot);
		threshold(image_roi_bot, image_roi_bot_out, 27, 255, CV_THRESH_BINARY);

		cv::Mat sssss;
		cv::bitwise_not(image_roi_bot_out, sssss);
		cv::imshow("image_roi_bot_out", sssss);

		vector<vector<cv::Point>> contours1;
		findContours(image_roi_bot_out, contours1, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
		cv::Mat mask1(image.size(), CV_8U, cv::Scalar(0));
		drawContours(mask1, contours1, -1, cv::Scalar(255), CV_FILLED);
		cv::bitwise_not(mask1, mask1);
		imshow("mask1", mask1);
		/*----------------------�ָ���--------------------*/

		if (frame_num == 1)
		{
			image.copyTo(image_roi, _ROI_);
			cv::imshow("image_roi", image_roi);
			cvtColor(image_roi, image_roi_gray, CV_BGR2GRAY);
			cv::imshow("image_roi_gray", image_roi_gray);
			t1 = clock();
			segmentationMap = cv::Mat(image.rows, image.cols, CV_8UC1);
			model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
			libvibeModel_Sequential_AllocInit_8u_C1R(model, image_roi_gray.data, image_roi_gray.cols, image_roi_gray.rows);
			t2 = clock();
			std::cout << t2 - t1 << std::endl;
			cv::imshow("segmentationMap111", segmentationMap);
			cvWaitKey();
		}
		else if (frame_num <= 100)
		{
			//sky_setment(frame, frame_t, frame_bound);
			t3 = clock();
			image.copyTo(image_roi, _ROI_);
			cvtColor(image_roi, image_roi_gray, CV_BGR2GRAY);
			libvibeModel_Sequential_Segmentation_8u_C1R(model, image_roi_gray.data, segmentationMap.data);
			libvibeModel_Sequential_Update_8u_C1R(model, image_roi_gray.data, segmentationMap.data);
			//medianBlur(segmentationMap, segmentationMap, 3); /* 3x3 median filtering */
			/* Shows the current frame and the segmentation map. */
			cv::imshow("Frame", image_roi);
			cv::Mat setmentshow;
			cv::bitwise_not(segmentationMap, setmentshow);
			cv::imshow("Segmentation by ViBe", setmentshow);//foreground picture
			cv::imwrite("E:/a.bmp", setmentshow);
			t4 = clock();
			std::cout << t4 - t3 << std::endl;
			cvWaitKey();
		}


	}

	cap.release();
	libvibeModel_Sequential_Free(model);
	return 1;
}

