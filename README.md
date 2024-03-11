# Leveraging Monocular Infrastructure Cameras for Collaborative Multi-View Perception for Indoor Autonomous Mobile Robots 
This repo contains code, data and step-by-step instructions to replicate the paper. This work can be divided into 5 phases - marker-less robot pose estimation, pose uncertainty estimation, camera placement optimziation, data association for multi-camera multi-robot navigation and sensor fusion.

# Marker-less Robot Pose Estimation
1. Download the weights file using this [link](https://drive.google.com/file/d/1scYfZa8a6hECXPae7nkQLXC1lbxKabC0/view?usp=sharing). If you wish to retrain the model, download the dataset from here: [Volta Pose](https://drive.google.com/drive/folders/1uBcb-0tSmQp2Nw9Y9dzLTH_DdySIXnbV?usp=sharing)
2. Put it in the same directory as markerless_cam_pose.py and execute the script.
3. The pose of the robot is estimated with respect to the camera's global origin.

# Pose Uncertainty Estimation
1. Install ultralytics and replace the default predict.py file with 'nms_predict.py'provided in this repository. The model is expected to return 300 bounding boxes during inference.
2. The 2D keypoints corresponding to every prediction along with the 3D keypoint coordinates is used for computing epistemic uncertainty (refer to the paper for the exact method).


# Camera Placement Optimization
1. Replace the path to your map file in the cam_placement_optimizer script
2. Tune the hyperparameters such as 'grid_size', 'weight_coverage' and 'weight'covariance' based on the nature of the operating environment.
3. Run the cam_placement_optimizer script, the corresponding camera locations are printed on the map.
4. Using the homography of the camera, project the camera positions in the image plane to the world plane.
NOTE: This algorithm only determines the x and y locations of the camera. The cameras are assumed to be fixed at a height d = 750 cm.

# Multi-Robot Tracking and Data Association 
1. Download the weights file in the same directory as that of the track_volta_annotated.py script and execute the script.

# Sensor Fusion
