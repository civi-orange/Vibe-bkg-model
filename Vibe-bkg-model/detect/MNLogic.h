#pragma once
#include "Detect_Parameter.h"

class MNLogic
{
public:
	MNLogic();

	~MNLogic();


	int inputSample(Detect_Parameter& m_Param, int nframeNum);

	vector<Target_index> MNLogicProc(vector<Target_info>& vctframe_info, Detect_Parameter& m_Param);

	int GetResult(Detect_Parameter& result, Detect_Parameter& m_Param);

public:

	//上一帧无目标帧编号
	//vector<int> m_vctEmptyFrameCount;

	//记录当前帧的上一帧是否为空
	//Target_info m_EmptyFrameRecord;
};

