#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>

using namespace cv;
using namespace std;

class PotholeDetection {

private:
	// YOLO cfg, weight, name file
	string cfg_path;
	string weight_path;
	string name_path;

	// Input size, output frame size, confidences
	Size input_size;
	Size frame_size;
	float min_confidence;
	float nms_confidence;

	// dnn model
	dnn::Net net;
	vector<Mat> predictions;

	// time check
	clock_t start, end;

	// YOLO data
	vector<string> class_names;
	vector<string> output_layers;
public:

	vector<pair<string, Rect>> outs;

	// constructor
	PotholeDetection();
	PotholeDetection(const string cfg, const string weight, const string name);

	// setting, getting method
	void setCfgFile(const string cfg);
	void setWeightFile(const string weight);
	void setNameFile(const string name);
	string getCfgFile();
	string getWeightFile();
	string getNameFile();

	void setSize(const Size target_size);
	Size getSize();
	
	void setMinConfidence(const float m_conf);
	float getMinConfidence();
	void setNmsConfidence(const float n_conf);
	float getNmsConfidence();

	void setYOLONames();
	vector<string> getYOLONames();
	void setOutPutLayers();
	vector<string> getOutputLayers();

	void setInputSize(const Size s);
	Size getInputSize();

	// setting dnn
	void initDnn();
	// predict
	void predict(Mat& frame, bool isGray = false, bool isFlip = false);
	
	//PostProcess
	void PostProcess(Mat& frame);
};