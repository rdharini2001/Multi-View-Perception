import cv2
import torch
import numpy as np
from ultralytics import YOLO

model = YOLO('train46.pt') #replace with downloaded weights file

threed_points = three_d_points_new = np.array([[0.01206,0.235,-0.21776],[0.00005,0.0813,-0.26776],[0.00005, 0.171, 0.19024],[0.16505, 0.2642, 0.10624],[0.16506, 0.2655, -0.16966],[-0.16495, 0.2641, 0.10524],[-0.16495, 0.2633, -0.17016],[-0.06495, 0.17695, -0.30763],
        [-0.01994,0.17695,-0.27316],[0.06006, 0.18075, -0.28716],[0.06505, 0.13695, -0.27976],[0.00005, 0.037, -0.26736],[-0.11994, 0.19595, -0.27163],[-0.11994, 0.15995, -0.27063],[0.11805, 0.19595, -0.27063],
        [0.11805, 0.15795, -0.27163],[0.17996,0.2377,-0.16876],[0.18006, 0.1211, -0.16876],[0.18075,0.238,0.10694],[0.18026,0.1226,0.10524],[0.03766,0.2073,0.14424],[0.03756,0.207,0.22124],[-0.03694,0.2073,0.14424],
        [-0.03654,0.2073,0.22164],[0.13015,0.1208, 0.22314],[-0.12935,0.1206,0.22254],[0.17955,0.1211,0.17304],[-0.17975,0.1206,0.17334],[-0.16495,0.2671,-0.05846],[0.06505, 0.13695, -0.27976]]) #these are obtained from Unity 3D game simulator

video_capture = cv2.VideoCapture('/path/to/video.mp4') #replace with your video file

while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    results = model(frame)
    keypoints = results[0].keypoints.xy.cpu().numpy()
    # PnP (Perspective-n-Point) using OpenCV
    object_points = threed_points  # 3D points in the world coordinate system
    image_points = keypoints[:, :2]  # 2D points in the image plane

    camera_matrix = np.array([[970.13975699,   0.        , 661.05696322],
                                   [  0.        , 965.0683426 , 324.24867006],
                                   [  0.        ,    0.       ,   1.        ]]) #for camera 219, for other cameras refer to homography.txt

    dist_coeffs = np.array([-0.44779831, 0.21493212, 0.0086979, -0.00269077, 0.00281984]) #for camera 219, for other cameras refer to homography.txt

    retval, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
    rvec, tvec, inliers = cv2.solvePnPRANSAC(object_points, image_points, camera_matrix, dist_coeffs) ##with RANSAC
  
    # rvec and tvec are the rotation vector and translation vector respectively
    print("Rotation Vector (rvec):", rvec.flatten())
    print("Translation Vector (tvec):", tvec.flatten())
        
    for point in image_points.astype(int):
        cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)
    cv2.imshow('Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
