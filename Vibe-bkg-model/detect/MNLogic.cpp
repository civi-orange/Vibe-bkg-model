#include "MNLogic.h"

MNLogic::MNLogic()
{
	
}


MNLogic::~MNLogic()
{

}


int MNLogic::inputSample(Detect_Parameter& m_Param, int nframeNum)
{
	int nRet = true;

	vector<int> vctpair;
	vctpair.clear();

	int nTargetNumThrd = m_Param.m_nOriFrameLength + m_Param.m_nPredLendth + 2;
	if (m_Param.m_vctOriginFrameinfo.size() > 0)
	{
		m_Param.m_vctpairs = MNLogicProc(m_Param.m_vctOriginFrameinfo, m_Param);
	}

	if (!m_Param.m_vctpairs.empty())
	{
        for (int i=0; i<(int)m_Param.m_vctpairs.size();i++)
		{
            if ((int)m_Param.m_vctpairs[i].nIndex > (int)m_Param.m_vctOriginFrameinfo.size())
			{
				m_Param.bFindTarget = false;
				return false;
			}
			vctpair.push_back(m_Param.m_vctpairs[i].nIndex);
			vctpair.push_back(m_Param.m_vctpairs[i].nLabel);
		}

        if ((int)vctpair.size()  > nTargetNumThrd * 3)
		{
			m_Param.bFindTarget = false;
			return false;
		}

        if ((int)vctpair.size() / 2 > m_Param.m_nOriFrameLength && (int)vctpair.size()  <= nTargetNumThrd * 3)
		{			
			m_Param.bFindTarget = true;
			return nRet;
		}
	}
	else
	{
		return false;
	}

}


vector<Target_index> MNLogic::MNLogicProc(vector<Target_info>& vctframe_info, Detect_Parameter& m_Param)
{
	vector<Target_index> vctpairs;
	vctpairs.clear();
	Target_index temp;

	//出现新的目标（暂不用）
	//vector<Target_index> vctpairsNew;
	//vctpairsNew.clear();
	//Target_index newTemp;

    if ((int)m_Param.m_vctOriginFrameinfo.size() >= m_Param.m_nOriFrameLength + 2)
	{
		//取出第一第二帧的中检测目标位置
		vector<cv::Point> vctposition1;
		vector<cv::Point> vctposition2;
		vctposition1.clear();
		vctposition2.clear();
		vctposition1 = vctframe_info[0].vctposition;
		vctposition2 = vctframe_info[1].vctposition;
		int count1 = vctposition1.size();
		int count2 = vctposition2.size();
		//卡尔曼滤波进行关联2*2
		cv::Mat err_ = (cv::Mat_<float>(2, 2) << vctposition1[0].x, 0, 0, vctposition1[0].y);
		cv::Mat err_Cov = err_ * err_;
		double eQ = cv::norm(err_);
		int k = 1;
		cv::Mat d = cv::Mat(cv::Size(1, 2), CV_32FC1);
		
		vector<cv::Mat> vctxInit;
		vector<cv::Mat> vctSS;
		vector<cv::Mat> vctPP;
		vector<cv::Mat> vctKx;
		vector<cv::Mat> vctPx0;
		vector<cv::Mat> vctxForest;
		vector<cv::Mat> vctOutForest;
		vector<cv::Mat> vctOutside;

		for (int m=0; m<count1; m++)
		{
			for (int n=0; n<count2; n++)
			{
				if (vctposition1[m].x != -1 && vctposition1[m].y != -1 &&
					vctposition2[n].x != -1 && vctposition2[n].y != -1)
				{

					double dx1 = max(0.0, (vctposition2[n].x - vctposition1[m].x - m_Param.dRateMax_x*m_Param.m_OriTs));
					double dy1 = max(0.0, (-vctposition2[n].x + vctposition1[m].x + m_Param.dRateMin_x*m_Param.m_OriTs));
					double d1 = dx1 + dy1;
					double dx2 = max(0.0, (vctposition2[n].y - vctposition1[m].y - m_Param.dRateMax_y*m_Param.m_OriTs));
					double dy2 = max(0.0, (-vctposition2[n].y + vctposition1[m].y + m_Param.dRateMin_y*m_Param.m_OriTs));
					double d2 = dx2 + dy2;
					cv::Mat d = (cv::Mat_<double>(1, 2) << d1, d2);

					cv::Mat err_1 = (cv::Mat_<double>(2, 2) << vctposition1[m].x, 0, 0, vctposition1[0].y);
					cv::Mat err_2 = (cv::Mat_<double>(2, 2) << vctposition2[n].x, 0, 0, vctposition2[0].y);

					cv::Mat err_Cov1 = err_1 * err_1;
					cv::Mat err_Cov2 = err_2 * err_2;

					cv::Mat d_mn = d * (err_Cov1 + err_Cov2).inv() * d.t();

					double seesee = 0;
					for (int h = 0; h < d_mn.rows; ++h)
					{
						for (int w = 0; w < d_mn.cols; ++w)
						{
							seesee = d_mn.at<double>(h, w);
						}
					}
					
					///////////修改使得矩阵乘法生效！！！
					if (seesee <= m_Param.m_OriThreshold)
					{
						temp.nIndex += 1;
						temp.nLabel = m;
						vctpairs.push_back(temp);
						temp.nIndex += 1;
						temp.nLabel = n;
						vctpairs.push_back(temp);

						//由前两个观测值外推
						cv::Mat xInit;
						cv::Mat x1 = (cv::Mat_<double>(1, 2) << vctposition2[n].x, vctposition2[n].y);
						cv::Mat x2 = (cv::Mat_<double>(1, 2) << (vctposition2[n].x - vctposition1[m].x) / m_Param.m_OriTs, (vctposition2[n].y - vctposition1[m].y) / m_Param.m_OriTs);
						cv::hconcat(x1, x2, xInit); //1*4
						
						vctxInit.push_back(xInit);
						//更新 初始化状态
						err_ = (cv::Mat_<double>(2, 2) << vctposition2[n].x, 0, 0, vctposition2[n].y);
                        err_Cov = err_ * err_;
                        double  errcov[2][2] = {0};
                        memset(errcov,0,sizeof (errcov));
						for (int h = 0; h < err_Cov.rows; ++h)
						{
							for (int w = 0; w < err_Cov.cols; ++w)
							{
								errcov[h][w] = err_Cov.at<double>(h, w);
							}
						}
						
						cv::Mat Px0 = (cv::Mat_<double>(4, 4) <<
							errcov[0][0], 0, errcov[0][0] / m_Param.m_OriTs, 0,
							0, errcov[1][1], 0, errcov[1][1] / m_Param.m_OriTs,
							errcov[0][0] / m_Param.m_OriTs, 0, errcov[0][0] / m_Param.m_OriTs / m_Param.m_OriTs, 0,
							0, errcov[1][1] / m_Param.m_OriTs, 0, errcov[1][1] / m_Param.m_OriTs / m_Param.m_OriTs
							);
						//状态一步预测
						cv::Mat xInit_t = xInit.t();
						cv::Mat xForest = m_Param.OriF * xInit_t;//4*1
						vctxForest.push_back(xForest);	
						//观测一步预测
						cv::Mat outForest = m_Param.OriH * xForest;//2*1///注意数值意义：（实际在matlab中被转置为1*2）
						vctOutForest.push_back(outForest);					
						//预测协方差
						cv::Mat PP = m_Param.OriF * Px0 * (m_Param.OriF.t()) + m_Param.OriGama*eQ*(m_Param.OriGama.t()); //4*4
						vctPP.push_back(PP);
						//计算信息协方差
						cv::Mat SS = m_Param.OriH * PP * (m_Param.OriH.t()) + err_Cov; //2*2
						vctSS.push_back(SS);
						//kalman滤波增益
						cv::Mat Kx = PP * (m_Param.OriH.t()) * (SS.inv());
						vctKx.push_back(Kx);
						//协方差更新
						Px0 = Px0 - Kx*SS*(Kx.t());
						vctPx0.push_back(Px0);
						//外推点
						cv::Mat outside = outForest;//2*1
						vctOutside.push_back(outside);	
						k = k + 1;
					}
				}
			}
		}

        for (int i=2;i< (int)m_Param.m_vctOriginFrameinfo.size();i++)
		{
			vector<cv::Point> vctpositions;
			vctpositions.clear();
			vctpositions = m_Param.m_vctOriginFrameinfo[i].vctposition;
			if (vctpositions.size() > 0)
			{
				for (int j = 0; j < k - 1; j++)
				{
					if (j == 0)
					{
						vector<double> vctPDA;
						vctPDA.clear();
                        for (int l = 0; l < (int)vctpositions.size(); l++)
						{
							cv::Mat x1 = (cv::Mat_<double>(1, 2) << vctpositions[l].x, vctpositions[l].y);
							cv::Mat dPDA = (x1 - (vctOutside[j].t()))*(vctSS[j].inv())*((x1 - vctOutside[j].t()).t());// 1*1						
							double dtemp = 0;
							for (int h = 0; h < dPDA.rows; ++h)
							{
								for (int w = 0; w < dPDA.cols; ++w)
								{
									dtemp = dPDA.at<double>(h, w);									
								}
							}
							vctPDA.push_back(dtemp);

						}

						std::vector <double>::iterator smallest = std::min_element(std::begin(vctPDA), std::end(vctPDA));

						if (*smallest < m_Param.m_OriThreshold)
						{
							temp.nIndex = i+1;
							temp.nLabel = std::distance(std::begin(vctPDA), smallest);
							vctpairs.push_back(temp);
						}
					}
					else
					{
						//出现新的目标，需要重新增加目标并调用MN关联
					}
				}
			}
		}


	}

	return vctpairs;
}


int MNLogic::GetResult(Detect_Parameter& result, Detect_Parameter& m_Param)
{
	
	m_Param.Clone(result);

	return true;
}
