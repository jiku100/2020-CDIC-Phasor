#include "PotholeDetection.h"
#include <cstring>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
	std::string YOLO_CFG_PATH;
	std::string YOLO_WEIGHT_PATH;
	std::string YOLO_NAME_PATH;

	if (argc < 2) {
		cout << "No Option Error!!" << endl;
		return 0;
	}
	if (strcmp(argv[1], "real") == 0) {
		cout << "real" << endl;
		YOLO_CFG_PATH = "./phasor/phasor_real_local_gray.cfg";
		YOLO_WEIGHT_PATH = "./phasor/phasor_real_local_gray.weights";
		YOLO_NAME_PATH = "./phasor/phasor-classes.names";
	}
	else if (strcmp(argv[1], "woodRock") == 0) {
		cout << "woodROck" << endl;
		YOLO_CFG_PATH = "./phasor/phasor_woodRock_local_gray.cfg";
		YOLO_WEIGHT_PATH = "./phasor/phasor_woodRock_local_gray.weights";
		YOLO_NAME_PATH = "./phasor/phasor-classes.names";
	}
	
	PotholeDetection detector(YOLO_CFG_PATH, YOLO_WEIGHT_PATH, YOLO_NAME_PATH);
	detector.setMinConfidence(0.1);
	detector.setInputSize(Size(256, 256));
	
	VideoCapture cap(0);
	Mat frame;
	double fps = cap.get(CAP_PROP_FPS);
	int delay = cvRound(1000 / fps);

	clock_t start;
	while (true) {
		cap >> frame;
		if (frame.empty()) {
			break;
		}
		start = clock();
		Mat src = frame.clone();
		detector.predict(src, true);
		detector.PostProcess(src);
		for (int i = 0; i < detector.outs.size(); i++) {
			Rect box = detector.outs[i];
			rectangle(frame, box, Scalar(92, 92, 205), 2);
		}
		imshow("result", frame);
		detector.outs.clear();
		
		if (waitKey(delay) == 27) {
			break;
		}
	}

	cap.release();
	return 0;
}