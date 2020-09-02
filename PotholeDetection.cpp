#include "PotholeDetection.h"

PotholeDetection::PotholeDetection() {
	this->frame_size = Size(640, 480);
	this->min_confidence = 0.5;
	this->nms_confidence = 0.5;
}

PotholeDetection::PotholeDetection(const string cfg, const string weight, const string name) {
	this->PotholeDetection::PotholeDetection();
	this->cfg_path = cfg;
	this->weight_path = weight;
	this->name_path = name;
}

void PotholeDetection::setCfgFile(const string cfg) {
	this->cfg_path = cfg;
}
void PotholeDetection::setWeightFile(const string weight) {
	this->weight_path = weight;
}
void PotholeDetection::setNameFile(const string name) {
	this->name_path = name;
}

string PotholeDetection::getCfgFile() {
	return this->cfg_path;
}
string PotholeDetection::getWeightFile() {
	return this->weight_path;
}
string PotholeDetection::getNameFile() {
	return this->name_path;
}

void PotholeDetection::setSize(const Size target_size) {
	this->frame_size = target_size;
}
Size PotholeDetection::getSize() {
	return this->frame_size;
}

void PotholeDetection::setMinConfidence(const float m_conf) {
	this->min_confidence = m_conf;
}
float PotholeDetection::getMinConfidence() {
	return this->min_confidence;
}
void PotholeDetection::setNmsConfidence(const float n_conf) {
	this->nms_confidence = n_conf;
}
float PotholeDetection::getNmsConfidence() {
	return this->nms_confidence;
}

void PotholeDetection::setYOLONames() {
	ifstream f;
	string line;
	f.open(this->name_path);
	if (!f) {
		std::cout << "YOLO-Name file is not opened" << std::endl;
		exit(0);
	}
	while (getline(f, line)) {
		this->class_names.push_back(line);
	}
	f.close();
}
vector<string> PotholeDetection::getYOLONames() {
	return this->class_names;
}
void PotholeDetection::setOutPutLayers() {
	if (this->output_layers.empty()) {
		std::vector<int> outLayers = this->net.getUnconnectedOutLayers();
		std::vector<String> layersNames = this->net.getLayerNames();
		this->output_layers.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); i++) {
			this->output_layers[i] = layersNames[outLayers[i] - 1];
		}
	}
}
vector<string> PotholeDetection::getOutputLayers() {
	return this->output_layers;
}

void PotholeDetection::setInputSize(const Size s) {
	this->input_size = s;
}

Size PotholeDetection::getInputSize() {
	return this->input_size;
}

void PotholeDetection::initDnn() {
	this->net = dnn::readNetFromDarknet(this->cfg_path, this->weight_path);
	this->net.setPreferableBackend(dnn::DNN_BACKEND_OPENCV);
	this->net.setPreferableTarget(dnn::DNN_TARGET_CPU);
}

void PotholeDetection::predict(Mat& frame, bool isGray, bool isFlip) {
	Mat blob;

	if (isGray) {
		cvtColor(frame, frame, COLOR_BGR2GRAY);
	}
	if (isFlip) {
		flip(frame, frame, 0);
		flip(frame, frame, 1);
	}
	resize(frame, frame, this->frame_size);
	dnn::blobFromImage(frame, blob, 1 / 255.0, this->input_size, Scalar(), true, false);
	net.setInput(blob);
	net.forward(this->predictions, this->output_layers);
	if (isGray) {
		cvtColor(frame, frame, COLOR_GRAY2BGR);
	}
}

void PotholeDetection::PostProcess(Mat& frame) {
	vector<int> classIds;
	std::vector<float> confidences;
	std::vector<Rect> boxes;

	for (size_t i = 0; i < this->predictions.size(); i++) {
		float* data = (float*)this->predictions[i].data;
		for (int j = 0; j < this->predictions[i].rows; ++j, data += this->predictions[i].cols)
		{
			Mat scores = this->predictions[i].row(j).colRange(5, this->predictions[i].cols);
			Point classIdPoint;
			double confidence;
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > this->min_confidence) {
				int center_x = (int)(data[0] * this->frame_size.width) < 0 ? 0 : (int)(data[0] * this->frame_size.width);
				int center_y = (int)(data[1] * this->frame_size.height) < 0 ? 0 : (int)(data[1] * this->frame_size.height);
				int w = center_x + int(data[2] * this->frame_size.width) > frame.cols ? frame.cols - center_x : int(data[2] * this->frame_size.width);
				int h = center_y + int(data[3] * this->frame_size.height) > frame.rows ? frame.rows - center_y : int(data[3] * this->frame_size.height);
				int left = center_x - w / 2 < 0 ? 0 : center_x - w / 2;
				int top = center_y - h / 2 < 0 ? 0 : center_y - h / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, w, h));
			}
		}
	}
	std::vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, this->min_confidence, this->nms_confidence, indices);
	for (size_t i = 0; i < indices.size(); i++) {
		int idx = indices[i];
		Rect box = boxes[idx];
		String label = this->class_names[classIds[idx]];
		this->outs.push_back(make_pair(label, box));
	}
}
