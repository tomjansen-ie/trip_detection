from ultralytics import YOLO
import cv2
import numpy as np
import torch

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = YOLO('yolo11l-pose.pt').to(device)
cap = cv2.VideoCapture('/Users/tomjansen/Desktop/github_projects/trip_detection/input/full.mp4')

keypoint_labels = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

pairs = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

output_path = '/Users/tomjansen/Desktop/github_projects/trip_detection/output/stickman_pose.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 20.0
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


def detect_trip(keypoints, prev_keypoints):
    try:
        head_y = keypoints[0][1]
        hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
        knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
        ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2

        if prev_keypoints is not None:
            prev_hip_y = (prev_keypoints[11][1] + prev_keypoints[12][1]) / 2
            prev_knee_y = (prev_keypoints[13][1] + prev_keypoints[14][1]) / 2
            prev_ankle_y = (prev_keypoints[15][1] + prev_keypoints[16][1]) / 2

            if ankle_y < knee_y and (prev_ankle_y - ankle_y) < 2 \
               or head_y < hip_y and (prev_hip_y - hip_y) > 5 \
               or knee_y < hip_y and (prev_knee_y - knee_y) > 5:
                return True
        return False
    except:
        return False


prev_keypoints = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    blank_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    results = model.predict(frame)

    for result in results:
        keypoints = result.keypoints.xy
        if keypoints is not None and len(keypoints) > 0:
            for person_keypoints in keypoints:
                for idx, (x, y) in enumerate(person_keypoints):
                    if x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        cv2.circle(blank_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
                        cv2.putText(frame, keypoint_labels[idx], (int(x) + 5, int(y) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(blank_frame, keypoint_labels[idx], (int(x) + 5, int(y) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                for pair in pairs:
                    if pair[0] < len(person_keypoints) and pair[1] < len(person_keypoints):
                        pt1, pt2 = person_keypoints[pair[0]], person_keypoints[pair[1]]
                        if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), 2)
                            cv2.line(blank_frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), 2)

                if detect_trip(person_keypoints, prev_keypoints):
                    cv2.putText(frame, "TRIP DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(blank_frame, "TRIP DETECTED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                prev_keypoints = person_keypoints

    out.write(blank_frame)
    cv2.imshow('Tripping Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()