import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import time

# Load YOLOv8 segmentation model
model = YOLO("best.pt")  # Adjust path to your weights

# Load class labels
with open("coco1.txt", "r") as f:
    class_list = f.read().splitlines()

# Open local video file
cap = cv2.VideoCapture("input.mp4")

# Define output video properties
frame_width, frame_height = 1020, 600
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (frame_width, frame_height))

    results = model(frame)
    detections = results[0].boxes.data
    masks = results[0].masks.data if results[0].masks is not None else []

    # Draw masks
    for mask in masks:
        mask = mask.cpu().numpy().astype(np.uint8) * 255
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = frame.copy()
        cv2.drawContours(overlay, contours, -1, (255, 105, 180), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

    # Apply Non-Maximum Suppression (NMS) to remove duplicate boxes
    if detections is not None and len(detections) > 0:
        detections_np = detections.cpu().numpy()
        boxes = []
        scores = []
        class_ids = []

        for det in detections_np:
            x1, y1, x2, y2, conf, cls_id = det
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])  # Convert to x,y,w,h
            scores.append(float(conf))
            class_ids.append(int(cls_id))

        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.3, nms_threshold=0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes[i]
                x2, y2 = x + w, y + h
                class_id = class_ids[i]
                class_name = class_list[class_id]

                cv2.rectangle(frame, (x, y), (x2, y2), (255, 255, 255), 1)
                label_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(frame, (x, y - 20), (x + label_size[0] + 6, y), (255, 255, 255), -1)
                cv2.putText(frame, class_name, (x + 3, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display FPS and frame number
    elapsed = time.time() - start_time
    current_fps = int(frame_count / elapsed)
    cv2.putText(frame, f'FPS: {current_fps}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f'Frame: {frame_count}', (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow("YOLOv8 Enhanced Segmentation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
