import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# Load YOLO model
model = YOLO('/path/to/your/model.pt')

# Initialize SORT tracker
tracker = Sort()

# Camera matrices and distortion coefficients
camera_matrices = [
    np.array([[970.13975699, 0, 661.05696322],
              [0, 965.0683426, 324.24867006],
              [0, 0, 1]]),  # Camera 1
    np.array([[960.0, 0, 650.0],
              [0, 960.0, 320.0],
              [0, 0, 1]])   # Camera 2
]
dist_coeffs = [
    np.array([-0.44779831, 0.21493212, 0.0086979, -0.00269077, 0.00281984]), 
    np.zeros(5)  
]

# Predefined 3D keypoints for the robot
keypoints_3d = np.array([[0.01206,0.235,-0.21776],[0.00005,0.0813,-0.26776],[0.00005, 0.171, 0.19024],[0.16505, 0.2642, 0.10624],[0.16506, 0.2655, -0.16966],[-0.16495, 0.2641, 0.10524],[-0.16495, 0.2633, -0.17016],[-0.06495, 0.17695, -0.30763],
        [-0.01994,0.17695,-0.27316],[0.06006, 0.18075, -0.28716],[0.06505, 0.13695, -0.27976],[0.00005, 0.037, -0.26736],[-0.11994, 0.19595, -0.27163],[-0.11994, 0.15995, -0.27063],[0.11805, 0.19595, -0.27063],
        [0.11805, 0.15795, -0.27163],[0.17996,0.2377,-0.16876],[0.18006, 0.1211, -0.16876],[0.18075,0.238,0.10694],[0.18026,0.1226,0.10524],[0.03766,0.2073,0.14424],[0.03756,0.207,0.22124],[-0.03694,0.2073,0.14424],
        [-0.03654,0.2073,0.22164],[0.13015,0.1208, 0.22314],[-0.12935,0.1206,0.22254],[0.17955,0.1211,0.17304],[-0.17975,0.1206,0.17334],[-0.16495,0.2671,-0.05846],[0.06505, 0.13695, -0.27976]]) #these are obtained from Unity 3D game simulator

# Open video captures for both cameras
cap1 = cv2.VideoCapture('/path/to/video_camera_1.mp4')
cap2 = cv2.VideoCapture('/path/to/video_camera_2.mp4')

# Function to estimate pose using PnP
def estimate_pose(keypoints_2d, camera_idx):
    keypoints_2d = np.array(keypoints_2d)
    _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(
        keypoints_3d, keypoints_2d, camera_matrices[camera_idx], dist_coeffs[camera_idx]
    )
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return translation_vector.flatten(), rotation_matrix
  
def pose_distance(pose1, pose2):
    t1, r1 = pose1
    t2, r2 = pose2
    t_dist = np.linalg.norm(t1 - t2)
    r_dist = np.linalg.norm(r1 - r2) 
    return t_dist, r_dist

# Track fusion logic
def fuse_poses(cam1_poses, cam2_poses, distance_threshold=0.5):
    fused_poses = {}
    for cam1_id, cam1_pose in cam1_poses.items():
        closest_dist = float('inf')
        closest_cam2_id = None

        for cam2_id, cam2_pose in cam2_poses.items():
            t_dist, r_dist = pose_distance(cam1_pose, cam2_pose)
            if t_dist < closest_dist and t_dist < distance_threshold:
                closest_dist = t_dist
                closest_cam2_id = cam2_id

        if closest_cam2_id:
            fused_poses[cam1_id] = cam2_poses.pop(closest_cam2_id)
        else:
            fused_poses[cam1_id] = cam1_pose

    fused_poses.update(cam2_poses)
    return fused_poses

# Main
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    # Detect robots in both cameras
    results1 = model(frame1)
    results2 = model(frame2)

    # Trackers and poses for Camera 1
    boxes1 = results1[0].boxes.xyxy.tolist()
    trackers1 = tracker.update(np.array(boxes1))
    poses_cam1 = {}

    for x1, y1, x2, y2, track_id in trackers1:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        keypoints_2d = [(center_x, center_y)]  
        translation, rotation = estimate_pose(keypoints_2d, camera_idx=0)
        poses_cam1[track_id] = (translation, rotation)

    # Trackers and poses for Camera 2
    boxes2 = results2[0].boxes.xyxy.tolist()
    trackers2 = tracker.update(np.array(boxes2))
    poses_cam2 = {}

    for x1, y1, x2, y2, track_id in trackers2:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        keypoints_2d = [(center_x, center_y)]  
        translation, rotation = estimate_pose(keypoints_2d, camera_idx=1)
        poses_cam2[track_id] = (translation, rotation)

    # Fuse poses from both cameras
    fused_poses = fuse_poses(poses_cam1, poses_cam2)

    # Visualize results (Camera 1 example)
    for track_id, pose in poses_cam1.items():
        t, _ = pose
        cv2.putText(frame1, f"ID: {track_id} Pose: {t}", (10, 50 + track_id * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display frames
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
