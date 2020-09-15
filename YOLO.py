import cv2
import numpy as np

YOLO_CFG_PATH = "./face-yolov3-tiny.cfg"
YOLO_WEIGHT_PATH = "./face-yolov3-tiny_41000.weights"
YOLO_NAMES_PATH = "./classes1.names"
net = cv2.dnn.readNetFromDarknet(YOLO_CFG_PATH, YOLO_WEIGHT_PATH)
classes = []
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    height, width, channels = frame.shape
    if ret is False: break
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # 좌표
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    font = cv2.FONT_HERSHEY_PLAIN
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            print(f"{x} {y} {w} {h}")
    if cv2.waitKey(10) is 27: break


