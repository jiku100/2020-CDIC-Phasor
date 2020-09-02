#include "PotholeDetection.h"
#include <utility>

using namespace cv;
using namespace std;

int main()
{
	std::string YOLO_CFG_PATH = "./phasor-gray-train-yolo-v3.cfg";
	std::string YOLO_WEIGHT_PATH = "./phasor-gray-train-yolo-v3_final.weights";
	std::string YOLO_NAME_PATH = "./phasor-classes.names";

	PotholeDetection detector(YOLO_CFG_PATH, YOLO_WEIGHT_PATH, YOLO_NAME_PATH);
	detector.setMinConfidence(0.7);
	detector.setNmsConfidence(0.4);
	detector.setInputSize(Size(256, 256));
	detector.initDnn();
	detector.setYOLONames();
	detector.setOutPutLayers();
	
	VideoCapture cap(0);
	Mat frame;


	while (true) {
		cap >> frame;
		Mat src = frame.clone();
		detector.predict(src, true);
		detector.PostProcess(frame);
		for (int i = 0; i < detector.outs.size(); i++) {
			string label = detector.outs[i].first;
			Rect box = detector.outs[i].second;
			rectangle(frame, box, (0, 0, 255));
			putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0));
		}
		imshow("result", frame);
		detector.outs.clear();

		
		if (waitKey(10) == 27) {
			break;
		}
	}

	cap.release();
	return 0;
}