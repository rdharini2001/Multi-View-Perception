import cv2
import torch
import numpy as np
from ultralytics import YOLO

model = YOLO('/path/to/weights.pt') #replace with downloaded weights file

threed_points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])

video_capture = cv2.VideoCapture('/path/to/video.mp4') #replace with your video file

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    results = model(frame)
    # Extract keypoint coordinates (assuming keypoints are in results.xyxy format)
    keypoints = results.xyxy[0][:, -4:]
    # PnP (Perspective-n-Point) using OpenCV
    object_points = threed_points  # 3D points in the world coordinate system
    image_points = keypoints[:, :2]  # 2D points in the image plane
  
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) #replace with your camera matrix (if available).
    dist_coeffs = np.zeros((4, 1)) #replace with your camera's distortion coefficients (if available).
    retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
  
    # rvec and tvec are the rotation vector and translation vector respectively
    print("Rotation Vector (rvec):", rvec.flatten())
    print("Translation Vector (tvec):", tvec.flatten())

    # Draw keypoints on the frame (just for visualization purposes)
    for point in image_points.astype(int):
        cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow('Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
