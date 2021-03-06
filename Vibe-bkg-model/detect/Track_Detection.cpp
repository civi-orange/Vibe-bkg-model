#include "Track_Detection.h"

Track_Detection::Track_Detection()
{
	m_Param = new Detect_Parameter;
	target_count_valib = 0;
}

Track_Detection::~Track_Detection()
{
	if (m_Param != NULL)
	{
		delete m_Param;
	}
}

int Track_Detection::Initialize()
{
	int nRet = true;

	return nRet;
}

int Track_Detection::SetParam(Detect_Parameter& param)
{
	int nRet = false;	
	param.Clone(*m_Param);

	return nRet;
}

int Track_Detection::InputSample(cv::Mat* ImageIn, int nFrNum)
{
    int nRet = true;
	Target_info  curInfo;//记录当前帧目标信息
	cv::Mat frame;
    m_Param->m_nFrNum = nFrNum;
    if (ImageIn == NULL){return false;}

    //区域检测
    if (m_Param->ImgROI.area() == 0){
		frame = *ImageIn;
    }
    else{
		frame = (*ImageIn)(m_Param->ImgROI);// 截取图像
	}

	//分离背景与目标
    cv::Mat framegray; //灰度图
	cvtColor(frame, framegray, CV_BGR2GRAY);	
	cv::Mat targetMat;


    if (m_Param->bbkg_type == 0)
	{
		///纯天空背景，分离目标测试
		cv::Mat frame_filter;
		cv::Mat filterSize = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		if (m_Param->bTargetColor == true)
		{
			morphologyEx(framegray, frame_filter, cv::MORPH_TOPHAT, filterSize);//顶帽
			//showImage("ding", frame_filter);
		}
		else
		{
			morphologyEx(framegray, frame_filter, cv::MORPH_BLACKHAT, filterSize);//底帽
            //showImage("bottom", frame_filter);
		}
		
		cv::Mat frame_filter_thrd, frame_filter_ostu_open, frame_filter_ostu_close;
		if (1)
		{
			//目标二值化分离;
			threshold(frame_filter, frame_filter_thrd, m_Param->m_graythrd, 255, CV_THRESH_BINARY);
			//showImage("binary", frame_filter_thrd);
		}
		else
		{
			//目标OSTU分离
			threshold(frame_filter, frame_filter_thrd, 0, 255, CV_THRESH_OTSU);
			//showImage("OSTU", frame_filter_thrd);
		}
		//开运算
		cv::Mat bottom_open = getStructuringElement(cv::MORPH_RECT, cv::Size(m_Param->m_nOpen, m_Param->m_nOpen));
		morphologyEx(frame_filter_thrd, frame_filter_ostu_open, cv::MORPH_OPEN, bottom_open);
		//showImage("open", frame_filter_ostu_open);
		//闭运算
		cv::Mat bottom_clsoe = getStructuringElement(cv::MORPH_RECT, cv::Size(m_Param->m_nClose, m_Param->m_nClose));
		morphologyEx(frame_filter_thrd, frame_filter_ostu_close, cv::MORPH_CLOSE, bottom_clsoe);
        //showImage("close", frame_filter_ostu_close);
		targetMat = frame_filter_ostu_close;
    }

    else if(m_Param->bbkg_type == 1)
	{
		if (m_Param->bTargetColor == false)
		{
            ///分离天空区域，带有地面物体
            cv::Mat frame_angle, frame_Otsu, frame_Otsu_1;
			double thrd1 = 0;
			double thrd2 = 0;
			threshold(framegray, frame_angle, thrd1, 255, CV_THRESH_TRIANGLE);
			threshold(framegray, frame_Otsu, thrd2, 255, CV_THRESH_OTSU);
			//showImage("otsu_angle", frame_Otsu);
			bitwise_not(frame_Otsu, frame_Otsu_1);//反色	
			//showImage("otsu_1", frame_Otsu_1);
			cv::Mat dilateSize, erosionSize, out_d, out_e, out_o;
			cv::Mat open = getStructuringElement(cv::MORPH_RECT, cv::Size(m_Param->m_nOpen, m_Param->m_nOpen));
			morphologyEx(frame_Otsu_1, out_o, cv::MORPH_OPEN, open);
			//showImage("sky_o", out_o);
			cv::Mat result, result_c, result_e;
			result = frame_Otsu_1 - out_o;
			//showImage("result", result);
			cv::Mat close = getStructuringElement(cv::MORPH_RECT, cv::Size(m_Param->m_nClose, m_Param->m_nClose));
			morphologyEx(result, result_c, cv::MORPH_CLOSE, close);
			//showImage("result_c", result_c);
			cv::Mat erode = getStructuringElement(cv::MORPH_RECT, cv::Size(m_Param->m_nErosion_1, m_Param->m_nErosion_1));
			morphologyEx(result_c, result_e, cv::MORPH_ERODE, erode);
			//showImage("result_e", result_e);
			//天空轮廓区域
			vector<vector<cv::Point>> contours;
			findContours(result_e, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
			cv::Mat mask(frame.size(), CV_8U, cv::Scalar(0));
			drawContours(mask, contours, -1, cv::Scalar(255), CV_FILLED);
			//showImage("mask", mask);
			targetMat = mask;
		}
		else
		{
			//白色目标未添加
		}
	}

    else if(m_Param->bbkg_type == 2)
    {
        cv::Mat frame_thrd;
        threshold(framegray, frame_thrd, 65, 255, CV_THRESH_BINARY);
        cv::Mat opening = getStructuringElement(cv::MORPH_RECT, cv::Size(30, 30));
        morphologyEx(frame_thrd, frame_thrd, cv::MORPH_OPEN, opening);//去除噪声
        //showImage("frame_thrd", frame_thrd);
        vector<vector<cv::Point>> contours_2;
        findContours(frame_thrd, contours_2, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
        cv::Mat mask_2(frame.size(), CV_8U, cv::Scalar(0));
        drawContours(mask_2, contours_2, -1, cv::Scalar(255), CV_FILLED);
        //showImage("mask1", mask_2);
        cv::bitwise_not(mask_2, targetMat);
        //showImage("targetMat", targetMat);
    }

    cv::Mat frameddd;//临时状态查看 for test
	frame.copyTo(frameddd);

	//求轮廓的相关信息
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
        //double dWidth = 0;
        //double dHeight =0;//最小外接矩形长、宽
        for (int kk = 0; kk < int(vctContours.size()); kk++)
		{
			double dLength = 0;
			double dArea = 0;
			dArea = contourArea(vctContours[kk], false);//目标面积
			cv::Mat mask_ann = cv::Mat(frame.rows, frame.cols, CV_8UC1, cv::Scalar(0));
			cv::drawContours(mask_ann, vctContours, kk, cv::Scalar(255), cv::FILLED, 8);
			//fillPoly(mask_ann, vctContours[kk], cv::Scalar(255));
			int nCount_White = 0;//白
			int nCount_Black = 0;//黑
			//通过迭代器访问图像的像素点
			cv::Mat_<uchar>::iterator itor = mask_ann.begin<uchar>();
			cv::Mat_<uchar>::iterator itorEnd = mask_ann.end<uchar>();
			for (; itor != itorEnd; ++itor)
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
			//绘制最小外接矩形的中心点
			//circle(dstImg, cv::Point(box[i].center.x, box[i].center.y), 5, cv::Scalar(0, 255, 0), -1, 8); 
			box[kk].points(rect);  //把最小外接矩形四个端点复制给rect数组
			//绘制外接矩形
			//rectangle(dstImg, cv::Point(boundRect[i].x, boundRect[i].y), cv::Point(boundRect[i].x + boundRect[i].width, boundRect[i].y + boundRect[i].height), cv::Scalar(0, 255, 0), 2, 8);

            //dHeight= sqrt(pow(rect[0].x - rect[1].x, 2) + pow(rect[0].y - rect[1].y, 2));
            //dWidth = sqrt(pow(rect[1].x - rect[2].x, 2) + pow(rect[1].y - rect[2].y, 2));
			//求中心

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
				w_hrate =(double)boundRect[kk].width / boundRect[kk].height;
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
				curInfo.nframeNum = nFrNum;
				//绘制最小外接矩形每条边
				//for (int j = 0; j < 4; j++)
				//{
				//	line(frameddd, rect[j], rect[(j + 1) % 4], cv::Scalar(0, 0, 255), 2, 8);
				//}
				cv::circle(frameddd, p, 1, cv::Scalar(0, 255, 0), 1, 8);
				putText(frameddd, std::to_string(dArea), p, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0), 1, 8);
			}			
		}
	}
    putText(frameddd, std::to_string(curInfo.vctposition.size()), cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, 8);
    if(curInfo.vctRegionArea.size() > 0)
        putText(frameddd, std::to_string(curInfo.vctRegionArea.back()), cv::Point(100, 150), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1, 8);
    //imshow("huatu", frameddd);
    //cvWaitKey(1);


	if (curInfo.vctposition.size() == 0)
	{
		if (m_Param->m_vctOriginFrameinfo.size() > 0)
		{
			m_Param->m_vctEmptyFrameCount.push_back(nFrNum);

            if ((int)m_Param->m_vctEmptyFrameCount.size() > m_Param->m_nEmptyFrameThrd)
			{
				if (m_Param->m_vctEmptyFrameCount.back() - m_Param->m_vctEmptyFrameCount.front() + 1 > m_Param->m_nEmptyFrameThrd)
				{
					//m_Param->m_vctOriginFrameinfo.clear();
					m_Param->m_vctEmptyFrameCount.clear();
					m_Param->bFindTarget = false;
					nRet = false;
					return nRet;
				}
			}
			if (m_Param->m_vctEmptyFrameCount.size() > 0 && m_Param->m_vctEmptyFrameCount.size() <= 2)
			{
				//如果不是连续为空，则重新记录
				if (m_Param->m_vctEmptyFrameCount.back() != nFrNum)
				{
					m_Param->m_vctEmptyFrameCount.clear();
				}
				return false;
			}
		}
	}
	if (curInfo.vctposition.size() > 0 && curInfo.vctposition.size() < 15)
    {
        if (m_Param->m_vctEmptyFrameCount.size() > 0 && (int)m_Param->m_vctEmptyFrameCount.size() <= m_Param->m_nEmptyFrameThrd)
		{
			//如果不是连续为空，则重新记录
			if (m_Param->m_vctEmptyFrameCount.back() != nFrNum)
			{
				m_Param->m_vctEmptyFrameCount.clear();
			}
		}
        m_Param->m_vctOriginFrameinfo.push_back(curInfo);
        if ((int)m_Param->m_vctSaveFrameinfo.size() < m_Param->m_nOriFrameLength + 2)
		{
			m_Param->m_vctSaveFrameinfo.push_back(curInfo);
		}
		else
		{
			m_Param->m_vctSaveFrameinfo.erase(m_Param->m_vctSaveFrameinfo.begin());
			m_Param->m_vctSaveFrameinfo.push_back(curInfo);
		}

		MNLogic m_mnlogic;

        if ((int)m_Param->m_vctOriginFrameinfo.size() > m_Param->m_nOriFrameLength + 2)
        {
            if ((int)m_Param->m_vctOriginFrameinfo.size() <= m_Param->m_nOriFrameLength + 2 + m_Param->m_nPredLendth)
			{
				nRet = m_mnlogic.inputSample(*m_Param, nFrNum);
			}
			else
			{
				m_Param->m_vctOriginFrameinfo.clear();
				m_Param->m_vctOriginFrameinfo.assign(m_Param->m_vctSaveFrameinfo.begin(), m_Param->m_vctSaveFrameinfo.end());
				nRet = m_mnlogic.inputSample(*m_Param, nFrNum);
			}
			return nRet;
		}
		else
		{
			nRet = false;
			return nRet;
		}
	}
	else if (curInfo.vctposition.size() > 15)
	{
		nRet = false;
		return nRet;
		//目标杂乱
	}

	return nRet;
}


int Track_Detection::DetectProc(cv::Mat& img, cv::Rect& target_box, int nFrNum)
{
	int nRet = true;	

    if (m_Param->bmethod == 0)
	{
		nRet = InputSample(&img, nFrNum);

		if (m_Param->bFindTarget == true && nRet == true)
		{
			int label = m_Param->m_vctpairs.back().nLabel;
			int index = m_Param->m_vctpairs.back().nIndex;

			vctTar_center.push_back(m_Param->m_vctOriginFrameinfo[index - 1].vctposition[label]);
			vctTar_center.back().x += m_Param->ImgROI.x;
			vctTar_center.back().y += m_Param->ImgROI.y;

			if (abs(vctTar_center.back().x - vctTar_center[vctTar_center.size() - 10].x) > 0 ||
				abs(vctTar_center.back().y - vctTar_center[vctTar_center.size() - 10].y) > 0)
			{
				target_box.width = m_Param->m_vctOriginFrameinfo[index - 1].vctWidth[label];
				target_box.height = m_Param->m_vctOriginFrameinfo[index - 1].vctHeight[label];
                target_box.width *= 1.5;
                target_box.height *= 3;
				int x = vctTar_center.back().x - target_box.width / 2;
				int y = vctTar_center.back().y - target_box.height / 2;
				if (x > 0) { target_box.x = x; }
				else { target_box.x = 0; }
				if (y > 0) { target_box.y = y; }
				else { target_box.y = 0; }

                ///关于目标框越界的问题 只考虑开始位置，结束位置默认不越界。结果查看
                cv::rectangle(img, target_box, cv::Scalar(0, 255, 0), 1, 8);
                //cv::circle(img, vctTar_center.back(), 1, cv::Scalar(255, 255, 0), 1, 8);
				target_count_valib++;
			}
			else
			{
				target_count_valib = 0;
                //cv::rectangle(img, target_box, cv::Scalar(0, 0, 255), 3, 8);
			}

		}
		else
		{
            //std::cout << "frame number: " << m_Param->m_vctOriginFrameinfo.back().nframeNum << std::endl;
			target_count_valib = 0;
            cv::rectangle(img, target_box, cv::Scalar(0, 0, 255), 1, 8);
		}
        showImage("target", img);
        std::cout << "target count: " << target_count_valib << std::endl;
		if (target_count_valib == m_Param->_target_detect_thrd)
		{
            //m_Param->m_TrackFlag = true;
            //m_Param->bDetectFlag = false;///集成算法时，去掉本行注释；注释本行是为了单独测试本算法,将if 判断修改成 >=
            std::cout << "------------detect success--------------" << std::endl;
		}
	}

    else if (m_Param->bmethod == 1)
	{
		//求轮廓的相关信息//后、前、子、父轮廓的编号与轮廓一一对应
		vector<vector<cv::Point>> vctContours;		vctContours.clear();
		vector<cv::Vec4i>  vctHierarchy;		vctHierarchy.clear();
		cv::Moments moment;		cv::Point p;		cv::Point pt[255];
		//中间过程图像
		cv::Mat img_g, img_b, img_out;
        img = img(m_Param->ImgROI);

        cvtColor(img, img_g, CV_BGR2GRAY);
        //showImage("yuan", img_g);

        double thrd = threshold(img_g, img_b, 0, 255, CV_THRESH_OTSU);
        std::cout<< "OSTD threshold:" << thrd <<std::endl;

        cv::Mat close_ = getStructuringElement(cv::MORPH_RECT, cv::Size(17, 17));
		morphologyEx(img_b, img_out, cv::MORPH_CLOSE, close_);
        cv::bitwise_not(img_out,img_out);
        //showImage("binary", img_out);

		findContours(img_out, vctContours, vctHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
		if (vctContours.size() > 0 && vctContours.size() < 5)
		{
			vector<cv::Rect> boundRect(vctContours.size());  //定义外接矩形集合
			vector<cv::RotatedRect> box(vctContours.size()); //定义最小外接矩形集合
            cv::Point2f rect[4];//四个顶点
            for (int kk = 0; kk < (int)vctContours.size(); kk++)
			{
                //double dLength = 0;
				double dArea = 0;
				dArea = contourArea(vctContours[kk], false);//目标面积
                cv::Mat mask_ann = cv::Mat(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
				cv::drawContours(mask_ann, vctContours, kk, cv::Scalar(255), cv::FILLED, 8);
				//fillPoly(mask_ann, vctContours[kk], cv::Scalar(255));
				int nCount_White = 0;//白
				int nCount_Black = 0;//黑
				//通过迭代器访问图像的像素点
				cv::Mat_<uchar>::iterator itor = mask_ann.begin<uchar>();
				cv::Mat_<uchar>::iterator itorEnd = mask_ann.end<uchar>();
				for (; itor != itorEnd; ++itor)
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

                //dLength = arcLength(vctContours[kk], false);// 轮廓长度
				double w_hrate = 0;//长宽比
				box[kk] = minAreaRect(cv::Mat(vctContours[kk]));  //计算每个轮廓最小外接矩形
				boundRect[kk] = boundingRect(cv::Mat(vctContours[kk]));
				box[kk].points(rect);  //把最小外接矩形四个端点复制给rect数组
				//求中心
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

                if (dArea > 30 && dArea < 4000 && w_hrate < 15 && boundRect[kk].area() < 4000 && p.y - 30 > 0 && img.rows - p.y > 30)
				{
					cv::circle(img, p, 1, cv::Scalar(0, 255, 0), 3, 8);

                    //p.x += m_Param->ImgROI.x;
                    //p.y += m_Param->ImgROI.y;

                    //showImage("gray_binary_result", img);
                    std::cout<<"target_area:"<<dArea<<std::endl;
				}
			}
		}

	}

    else
	{
		nRet = false;
		return nRet;
	}
	return nRet;
}

void Track_Detection::GetParameter(Detect_Parameter& param)
{
	m_Param->Clone(param);
}

int Track_Detection::Uninitialize()
{
	int nRet = true;

	return nRet;
}
