# Zero-Shot Pose Estimation and Tracking of Autonomous Mobile Robots - A Multi-Camera Multi-Robot Perception Framework
This repo contains code, data and instructions to replicate the paper - 'Zero-Shot Pose Estimation and Tracking of Autonomous Mobile Robots - A Multi-Camera Multi-Robot Perception Framework'. This work is divided into 5 phases - zero-shot robot pose estimation, pose uncertainty estimation, camera placement optimziation, data association for multi-camera multi-robot navigation and sensor fusion. Below is the overall framework proposed. We also provide instructions for model-based instance-level training for pose estimation to compare it wiith the proposed zero-shot pipeline.

![alt text](https://github.com/rdharini2001/Multi-View-Perception/blob/main/Zero_Pose.JPG)


# Zero-Shot Pose Estimation and Uncertainty Quantification
Instructions for stage-wise execution of the pose pipeline - 

Step 0 - Follow this [notebook](https://github.com/paulguerrero/lang-sam/blob/main/example_notebook/getting_started_with_lang_sam.ipynb) for segmenting out the target robot from the query image using a text prompt.

Step 1 - Follow the instructions provided in this [notebook](https://github.com/facebookresearch/segment-anything/blob/main/notebooks/automatic_mask_generator_example.ipynb) to generate segmentation masks and associated centroids for the target robot. Save the mask and centroid locations in a text file.

Step 2 - Run ALIKE to obtain 2D keypoints - 
```
git clone https://github.com/Shiaoming/ALIKE
cd ALIKE
pip install -r requirements.txt
python demo.py -h
usage: demo.py [-h] [--model {alike-t,alike-s,alike-n,alike-l}]
               [--device DEVICE] [--top_k TOP_K] [--scores_th SCORES_TH]
               [--n_limit N_LIMIT] [--no_display] [--no_sub_pixel]
               input

positional arguments:
  input                 Image directory or movie file or "camera0" (for
                        webcam0).

optional arguments:
  -h, --help            show this help message and exit
  --model {alike-t,alike-s,alike-n,alike-l}
                        The model configuration
  --device DEVICE       Running device (default: cuda).
  --top_k TOP_K         Detect top K keypoints. -1 for threshold based mode,
                        >0 for top K mode. (default: -1)
  --scores_th SCORES_TH
                        Detector score threshold (default: 0.2).
  --n_limit N_LIMIT     Maximum number of keypoints to be detected (default:
                        5000).
  --no_display          Do not display images to screen. Useful if running
                        remotely (default: False).
  --no_sub_pixel        Do not detect sub-pixel keypoints (default: False).
```
Step 3 - Run ```keypointness.py``` to filter 2D keypoints based on confidence scores and assign semantic labels.

Step 4 - Run ZoeDepth using the below instructions to obtain a depth map for the query image. This depth map is used to create an initial point cloud of the target robot. The correponding 3D keypoint locations can also be computed using the pinhole camera model.

1. It is recommended to fetch the latest [MiDaS repo](https://github.com/isl-org/MiDaS) via torch hub.
   
```
import torch
torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)  # Triggers fresh download of MiDaS repo
```

2. Use ZoeDepth to predict depth in the query image

```
##### sample prediction
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
zoe = model_zoe_n.to(DEVICE)

# Local file
from PIL import Image
image = Image.open("/path/to/image.jpg").convert("RGB")  # load
depth_numpy = zoe.infer_pil(image)  # as numpy

depth_pil = zoe.infer_pil(image, output_type="pil")  # as 16-bit PIL Image

depth_tensor = zoe.infer_pil(image, output_type="tensor")  # as torch tensor

# Tensor 
from zoedepth.utils.misc import pil_to_batched_tensor
X = pil_to_batched_tensor(image).to(DEVICE)
depth_tensor = zoe.infer(X)

# From URL
from zoedepth.utils.misc import get_image_from_url

# Example URL
URL = "path_to_image"

image = get_image_from_url(URL)  # fetch
depth = zoe.infer_pil(image)

# Save raw
from zoedepth.utils.misc import save_raw_16bit
fpath = "/path/to/output.png"
save_raw_16bit(depth, fpath)

# Colorize output
from zoedepth.utils.misc import colorize

colored = colorize(depth)

# save colored output
fpath_colored = "/path/to/output_colored.png"
Image.fromarray(colored).save(fpath_colored)
```
Generate a point cloud using the above depth map following [these](https://github.com/HarendraKumarSingh/stereo-images-to-3D-model-generation/blob/master/depth-map-to-3D-point-cloud.ipynb) instructions. This serves as the initial pointcloud.

Step 5 - Follow [these](https://colmap.github.io/install.html) instructions and install colmap to build a 3D SfM representation of the target object. This serves as the final accurate point cloud.

Step 6 - Run ```icp.py``` to align both the pointclouds and obtain a transformation matrix ```T```. Use ```T``` to refine the locations of the 3D keypoints and get more accurate estimates.

Step 7 - Use ```cv2.solvePnP``` to estimate the robot's 6D pose. Refer [here](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html) for more details.

Step 8 - Run ```zero_uncertain.py``` to estimate the covariance associated with the above pipeline.

We are working on providing docker capabilities for user-friendly usage. Stay tuned!

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
4. [Colmap](https://colmap.github.io/install.html)
5. [Segment Anything Model](https://github.com/facebookresearch/segment-anything)
6. [ALIKE](https://github.com/Shiaoming/ALIKE)
7. [ZoeDepth](https://github.com/isl-org/ZoeDepth)

Additional details can be found [here](https://github.com/rdharini2001/Camera-Based-6D-Robot-Pose/tree/main).

# Recommended citation
```
@inproceedings{raghavan2024multiviewperception,
  author = {Dharini Raghavan, Raghu Krishnapuram and Bharadwaj Amrutur},
  pages = {1--8},
  title = {{Zero-Shot Pose Estimation and Tracking of Autonomous Mobile Robots - A Multi-Camera Multi-Robot Perception Framework}},
  year = {2024}
}
```
