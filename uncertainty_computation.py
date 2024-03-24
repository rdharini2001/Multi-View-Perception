import cv2
import numpy as np
import torch
# Initialize YOLO model
model = YOLO('train46.pt')

# Camera parameters
camera_matrix = np.array([[970.13975699,   0.        , 661.05696322],
                                   [  0.        , 965.0683426 , 324.24867006],
                                   [  0.        ,    0.       ,   1.        ]]) #for camera 219, for other cameras refer to homography.txt
dist_coeffs = np.array([-0.44779831, 0.21493212, 0.0086979, -0.00269077, 0.00281984]) #for camera 219, for other cameras refer to homography.txt

# Function to estimate poses using PnP algorithm with 2D-3D correspondences
def estimate_pose(keypoints_2d, keypoints_3d):
    # Convert keypoints to numpy arrays
    keypoints_2d = np.array(keypoints_2d)
    keypoints_3d = np.array(keypoints_3d)
    # Perform PnP
    _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs)
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return translation_vector, rotation_matrix
    # Append pose estimate to list
   
def detect_and_estimate_pose(frame):
    try:
        # Perform object detection
        results = model(frame)
        # Extract keypoints from YOLO detections
        keypoints = results[0].keypoints.xy.cpu().numpy()
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        # Prepare 3D keypoints
        keypoints_3d = np.array([[0.01206,0.235,-0.21776],[0.00005,0.0813,-0.26776],[0.00005, 0.171, 0.19024],[0.16505, 0.2642, 0.10624],[0.16506, 0.2655, -0.16966],[-0.16495, 0.2641, 0.10524],[-0.16495, 0.2633, -0.17016],[-0.06495, 0.17695, -0.30763],
        [-0.01994,0.17695,-0.27316],[0.06006, 0.18075, -0.28716],[0.06505, 0.13695, -0.27976],[0.00005, 0.037, -0.26736],[-0.11994, 0.19595, -0.27163],[-0.11994, 0.15995, -0.27063],[0.11805, 0.19595, -0.27063],
        [0.11805, 0.15795, -0.27163],[0.17996,0.2377,-0.16876],[0.18006, 0.1211, -0.16876],[0.18075,0.238,0.10694],[0.18026,0.1226,0.10524],[0.03766,0.2073,0.14424],[0.03756,0.207,0.22124],[-0.03694,0.2073,0.14424],
        [-0.03654,0.2073,0.22164],[0.13015,0.1208, 0.22314],[-0.12935,0.1206,0.22254],[0.17955,0.1211,0.17304],[-0.17975,0.1206,0.17334],[-0.16495,0.2671,-0.05846],[0.06505, 0.13695, -0.27976]]) #these are obtained from Unity 3D game simulator
        # Estimate pose using PnP
        translation, rotation = estimate_pose(keypoints, keypoints_3d)
        # Combine rotation matrix and translation vector to form pose
        pose = np.hstack((rotation_matrix, translation_vector))
        poses.append(pose)
        return poses

# Function to compute covariance matrix
def compute_covariance(poses):
    # Compute mean pose
    mean_pose = np.mean(poses, axis=0)
    # Compute difference from mean
    diff = poses - mean_pose
    # Compute covariance matrix
    covariance = np.dot(diff.T, diff) / len(poses)
    return covariance

# Function to propagate covariance forward using the forward propagation theorem
def propagate_covariance(covariance):
    # Propagate covariance using forward propagation theorem
    return np.dot(covariance, covariance.T)

# Main function to process video feed
def process_video_feed():
    cap = cv2.VideoCapture('rtsp://admin:artpark123@192.168.0.219:554/Streaming/Channels/1/')  # Change to the appropriate video feed
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Detect keypoints
        keypoints = detect_keypoints(frame)
        if keypoints is not None:
            # Estimate poses using PnP algorithm
            poses = estimate_poses(frame, keypoints)
            # Compute covariance matrix
            covariance = compute_covariance(poses)
            # Propagate covariance forward
            propagated_covariance = propagate_covariance(covariance)
            # Print pose uncertainty for the current frame
            print("Pose uncertainty for frame:", propagated_covariance)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Run the main function
process_video_feed()
