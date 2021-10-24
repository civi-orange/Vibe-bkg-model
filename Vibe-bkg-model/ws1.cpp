#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

Vec3b RandomColor(int value);  //���������ɫ����

int main2222(char argc, char* argv[])
{
	cv::Mat image;
	cv::VideoCapture cap("E:/target track/mianyangvideo/sample1.mp4");
	cap.read(image);
	imshow("Source Image", image);
	//�ҶȻ����˲���Canny��Ե���
	Mat imageGray;
	cvtColor(image, imageGray, CV_RGB2GRAY);//�Ҷ�ת��
	GaussianBlur(imageGray, imageGray, Size(5, 5), 2);   //��˹�˲�
	imshow("Gray Image", imageGray);
	Canny(imageGray, imageGray, 1, 180);
	imshow("Canny Image", imageGray);

	//��������
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imageGray, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
	Mat imageContours = Mat::zeros(image.size(), CV_8UC1);  //����
	Mat marks(image.size(), CV_32S);   //Opencv��ˮ��ڶ����������
	marks = Scalar::all(0);
	int index = 0;
	int compCount = 0;
	for (; index >= 0; index = hierarchy[index][0], compCount++)
	{
		//��marks���б�ǣ��Բ�ͬ������������б�ţ��൱������עˮ�㣬�ж������������ж���עˮ��
		drawContours(marks, contours, index, Scalar::all(compCount + 1), 1, 8, hierarchy);
		drawContours(imageContours, contours, index, Scalar(255), 1, 8, hierarchy);
	}

	//��������һ�´���ľ���marks����ʲô����
	Mat marksShows;
	convertScaleAbs(marks, marksShows);
	imshow("marksShow", marksShows);
	imshow("����", imageContours);
	watershed(image, marks);

	//����������һ�·�ˮ���㷨֮��ľ���marks����ʲô����
	Mat afterWatershed;
	convertScaleAbs(marks, afterWatershed);	
	imshow("After Watershed", afterWatershed);

	//vector<vector<Point>> contours1;
	//vector<Vec4i> hierarchy1;
	//findContours(PerspectiveImage, contours1, hierarchy1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point());
	//cv::Mat mask1(image.size(), CV_8U, cv::Scalar(0));
	//drawContours(mask1, contours, -1, cv::Scalar(255), CV_FILLED);
	//imshow("111", mask1);



	//��ÿһ�����������ɫ���
	Mat PerspectiveImage = Mat::zeros(image.size(), CV_8UC3);
	for (int i = 0; i < marks.rows; i++)
	{
		for (int j = 0; j < marks.cols; j++)
		{
			int index = marks.at<int>(i, j);
			if (marks.at<int>(i, j) == -1)
			{
				PerspectiveImage.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
			}
			else
			{
				PerspectiveImage.at<Vec3b>(i, j) = RandomColor(index);
			}
		}
	}
	imshow("After ColorFill", PerspectiveImage);




	//�ָ�����ɫ�Ľ����ԭʼͼ���ں�
	Mat wshed;
	addWeighted(image, 0.4, PerspectiveImage, 0.6, 0, wshed);
	imshow("AddWeighted Image", wshed);

	waitKey();
}

Vec3b RandomColor(int value)
{
	value = value % 255;  //����0~255�������
	RNG rng;
	int aa = rng.uniform(0, value);
	int bb = rng.uniform(0, value);
	int cc = rng.uniform(0, value);
	return Vec3b(aa, bb, cc);
}