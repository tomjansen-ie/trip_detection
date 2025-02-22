from ultralytics import YOLO
import cv2
import numpy as np


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = YOLO('yolov8n-pose.pt').to(device)  # You can use yolov8m-pose.pt or yolov8l-pose.pt for more accuracy

cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    for result in results:
        keypoints = result.keypoints.xy  # Extract (x, y) coordinates for keypoints
        # keypoints: [person][keypoint][x, y] -> Use this for tripping analysis

    cv2.imshow('Tripping Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

