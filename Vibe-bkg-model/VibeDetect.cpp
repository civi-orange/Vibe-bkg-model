// VibeDetect.cpp : 定义控制台应用程序的入口点。
//
#include <vibe-background-sequential.h>
#include <opencv2/core.hpp>
#include <opencv.hpp>
#include <highgui.h>
#include <cv.h>

using namespace std;

void showImage(std::string str, cv::Mat img)
{
	imshow(str, img);
	cvWaitKey(1);
}

struct target_ {
	int nFrameNum;
	cv::Rect vct_target_rect;//目标信息
	int target_count;//目标累积出现次数
};

//阈值信息
struct thrd_ {
	int target_thrd = 3;//目标关联成功次数阈值
	int distance_min = 1;//关联距离阈值
	int distance_max = 15;
	int width_diff = 5;//关联宽高变化阈值
	int height_diff = 5;
};


int sky_setment(cv::Mat& img_in, cv::Mat& img_out)
{
	cv::Mat Kernelx, Kernely;

	Kernelx = (cv::Mat_<double>(3, 3) << 1, 1, 1, 0, 0, 0, -1, -1, -1);
	Kernely = (cv::Mat_<double>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1);
	cv::Mat grad_x, grad_y;
	filter2D(img_in, grad_x, CV_16S, Kernelx, cv::Point(-1, -1));
	filter2D(img_in, grad_y, CV_16S, Kernely, cv::Point(-1, -1));
	cv::Mat abs_grad_x, abs_grad_y, gray;
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gray);
	showImage("gray", gray);

	cv::Mat gray_out, gray_f1, gray_f2;
	threshold(gray, gray_out, 0, 255, CV_THRESH_OTSU);
	showImage("gray_out", gray_out);
	cv::Mat f1 = getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
	cv::Mat f2 = getStructuringElement(cv::MORPH_RECT, cv::Size(6, 6));
	morphologyEx(gray_out, gray_f1, cv::MORPH_DILATE, f2);
	morphologyEx(gray_f1, gray_f2, cv::MORPH_ERODE, f1);

	showImage("gray_f2", gray_f2);
	cvWaitKey();


	return 1;
}


int main666()
{
	int frame_num = 0;
	cv::Mat frame;
	cv::VideoCapture cap("E:/target track/mianyangvideo/track1.mp4");
	//cv::VideoCapture cap("E:/target track/视频/手机视频/地面干扰.mp4");
	if (!cap.isOpened())
	{
		return false;
	}
	double t1, t2, t3, t4;
	cv::Mat frame_t, frame_bound;
	cv::Mat frame_model, frame_input;
	vibeModel_Sequential_t *model = NULL;
	cv::Mat segmentationMap;
	while (cap.read(frame))
	{
		
		frame_num++;
		cv::Mat frame_gray;
		cvtColor(frame, frame_gray, CV_BGR2GRAY);

		//cv::Mat frame_gray_test;//聚集度拟合
		//threshold(frame_gray, frame_gray_test, 60, 255, CV_THRESH_BINARY);
		//bitwise_not(frame_gray_test, frame_gray_test);
		//showImage("frame_gray_test", frame_gray_test);


		if (frame_num ==1)
		{
			t1 = clock();
			sky_setment(frame_gray, frame_t);
			//frame_model = frame_t.mul(frame_bound);

			frame_gray.copyTo(frame_input, frame_t);

			segmentationMap = cv::Mat(frame_input.rows, frame_input.cols, CV_8UC1);
			model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
			libvibeModel_Sequential_AllocInit_8u_C1R(model, frame_input.data, frame_input.cols, frame_input.rows);
			t2 = clock();
			std::cout << t2 - t1 << std::endl;
		}
		else if (frame_num <= 100)
		{
			//sky_setment(frame, frame_t, frame_bound);
			t3 = clock();
			frame_gray.copyTo(frame_input, frame_model);

			libvibeModel_Sequential_Segmentation_8u_C1R(model, frame_input.data, segmentationMap.data);
			libvibeModel_Sequential_Update_8u_C1R(model, frame_input.data, segmentationMap.data);

			//medianBlur(segmentationMap, segmentationMap, 3); /* 3x3 median filtering */
			/* Shows the current frame and the segmentation map. */
			//showImage("Frame", frame);
			showImage("Segmentation by ViBe", segmentationMap);//foreground picture
			t4 = clock();
			std::cout << t4 - t3<< std::endl;
		}

		else if (frame_num > 5)
		{
			frame_gray.copyTo(frame_input, frame_model);
			libvibeModel_Sequential_Segmentation_8u_C1R(model, frame_input.data, segmentationMap.data);
			libvibeModel_Sequential_Update_8u_C1R(model, frame_input.data, segmentationMap.data);
		}

	}			

	cap.release();
	libvibeModel_Sequential_Free(model);
	system("pause");
    return 0;

}

int EdgeFit(cv::Mat& img, int cut_num)
{
	int nRet = true;
	std::vector<cv::Mat> vct_img_cut;
	vct_img_cut.clear();

	int width = img.cols;
	int height = img.rows;
	cv::Mat gray_img, binary_img;
	cvtColor(img, gray_img, CV_BGR2GRAY);
	threshold(gray_img, binary_img, 60, 255, CV_THRESH_BINARY);//调节阈值直接分离地空
	showImage("binary_img", binary_img);

	for (int i=0; i<cut_num; i++)
	{
		cv::Rect rect(i*width / cut_num, 0, width / cut_num, height);
		cv::Mat temp;
		temp = binary_img(rect);
		vct_img_cut.push_back(temp);
	}





	return nRet;
}

int coor_opt(cv::Mat Obj_all)//外推策略
{
	int nRet = true;
	//cv::Mat mask()


	return nRet;
}