# Zero-Shot Pose Estimation and Tracking of Autonomous Mobile Robots - A Multi-Camera Multi-Robot Perception Framework
# [Paper](https://drive.google.com/file/d/1QNR3rqcUCgoFsgf7qkK5IoRGjACY71ll/view?usp=sharing) | [Video](https://drive.google.com/file/d/1Wf-Vssxf6LgiQLftj3QVA6msW8QQP0EW/view?usp=sharing)
This repo contains code, data and instructions to replicate the paper - 'A Multi-Camera Perception Framework for Zero-Shot Pose Estimation and Tracking of Autonomous Mobile Robots'. This work is divided into 5 phases - zero-shot robot pose estimation, pose uncertainty estimation, camera placement optimziation, data association for multi-camera multi-robot navigation and sensor fusion. Below is the overall framework proposed. We also provide instructions for model-based instance-level training for pose estimation to compare it wiith the proposed zero-shot pipeline.

![alt text](https://github.com/rdharini2001/Multi-View-Perception/blob/main/Zero_Pose.JPG)

Please read the supplementary material for a comprehensive and more detailed explanation of various aspects of the framework that are not covered in the paper.

# Zero-Shot Pose Estimation and Uncertainty Quantification


# Instance Level Robot Pose Estimation (Optional)
Download the pre-trained weights using this [link](https://drive.google.com/file/d/1scYfZa8a6hECXPae7nkQLXC1lbxKabC0/view?usp=sharing). If you wish to retrain the model, download the dataset from here: [Volta Pose](https://drive.google.com/drive/folders/1uBcb-0tSmQp2Nw9Y9dzLTH_DdySIXnbV?usp=sharing). We fine-tune the YOLOv8n-pose model for keypoint detection. Refer to this [link](https://github.com/ultralytics/ultralytics/blob/4ac93d82faf3324d18a233090445e83cfac62ce2/ultralytics/nn/modules/head.py) for more details on the model architecture. 

# Model Training
```
from ultralytics import YOLO
# Load a model
model = YOLO('yolov8n-pose.pt')  # load a pretrained model
# Train the model
results = model.train(data='volta_pose.yaml', epochs=200, imgsz=640)
```
Place the trained model in the same directory as ```markerless_cam_pose.py``` and execute the script. The pose of the robot is estimated with respect to the camera's global origin.

# Instance Level Pose Uncertainty Estimation (Optional)
1. Install ultralytics and replace the default ```predict.py``` file with ```nms_predict.py``` provided in this repository. The model is expected to return 300 bounding boxes during inference. NOTE - You may have to rebuild the script for changes to take place.
2. Run the ```uncertainty_computation.py``` script to estimate epistemic model uncertainty and associated covariance. (refer to the paper for the exact method).
3. Sample covariance data is provided in ```cov_data.txt```.

# Camera Placement Optimization
1. Replace the path to your map file in the ```cam_placement_optimizer``` script.
2. Tune the hyperparameters such as ```covariance_threshold```, ```d_max``` and ```grid size``` based on the nature of the operating environment.
3. Run the ```cam_placement_optimizer``` script, the corresponding camera locations are printed on the map.
4. Using the homography of the camera, project the camera positions in the image plane to the world plane.
NOTE: This algorithm only determines the ```x``` and ```y``` locations of the camera. The cameras are assumed to be fixed at a height ```d = 750 cm```.

# Multi-Robot Tracking and Data Association 
1. Execute the ```track_volta_annotated.py``` script. This is used for tracking one or more robots within the field of view of the camera. The corresponding track locations/pose of the robot are printed on every frame.
2. The ```association.py``` script is used for associating multiple sensor measurements with respective robot tracks. It internally runs a Hungarian algorithm which further involves minimization of a cost function leading to the assignments of measurements to corresponding tracks. The cost function depends on the norm of the covariance associated with the camera. Both cases of overlapping views and non-overlapping views are handled separately. Refer to the paper for more details.

# Sensor Fusion
1. In this work, we consider RTAB Map as the base RGBD SLAM framework. We use the [robot_localization](https://github.com/cra-ros-pkg/robot_localization) package and implement an extended Kalman filter to communicate with the RTAB map node.
2. We consider 5 different sensors - onboard LiDAR, IMU, wheel encoders, onboard depth camera and external monocular camera.
3. Install RTAB Map using the instructions provided in the [RTABMap Repo](https://github.com/introlab/rtabmap) and launch RTAB Map with the custom launch file ```camera_sensor_fusion.launch``` provided in this repo. NOTE - The initial covariance and the noise covariance matrices might need some tuning based on the experimental setup and operating conditions.

We extend our thanks to the many wonderful works that were used in this project - 
1. [Ultralytics](https://github.com/ultralytics)
2. [robot_localization](https://github.com/cra-ros-pkg/robot_localization)
3. [RTABMap](https://github.com/introlab/rtabmap)

# Recommended citation
```
@inproceedings{raghavan2024multiviewperception,
  author = {Dharini Raghavan, Raghu Krishnapuram and Bharadwaj Amrutur},
  pages = {1--8},
  title = {{Leveraging Monocular Infrastructure Cameras for Collaborative Multi-View Perception for Indoor Autonomous Mobile Robots}},
  year = {2024}
}
```
