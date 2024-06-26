import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial import distance
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from tf import transformations as tf_transformations

# Initialize ROS node
rospy.init_node('pose_estimation')
# Initialize ROS publisher for images
image_pub = rospy.Publisher('/camera/image_processed', Image, queue_size=10)
bridge = CvBridge()
# Initialize ROS publisher for pose
pose_pub = rospy.Publisher('/pose_estimation', PoseStamped, queue_size=10)
# Initialize YOLO model
model = YOLO('train46.pt') #replace with downloaded weights file

# Camera parameters (update these with actual camera parameters)
camera_matrix = np.array([[970.13975699,   0.        , 661.05696322],
                                   [  0.        , 965.0683426 , 324.24867006],
                                   [  0.        ,    0.       ,   1.        ]]) #for camera 219, for other cameras refer to homography.txt
dist_coeffs = np.array([-0.44779831, 0.21493212, 0.0086979, -0.00269077, 0.00281984]) #for camera 219, for other cameras refer to homography.txt

# Function to estimate pose using PnP
def estimate_pose(keypoints_2d, keypoints_3d):
    # Convert keypoints to numpy arrays
    keypoints_2d = np.array(keypoints_2d)
    keypoints_3d = np.array(keypoints_3d)
    # Perform PnP
    _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(keypoints_3d, keypoints_2d, camera_matrix, dist_coeffs)
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    return translation_vector, rotation_matrix

# ROS callback function for receiving camera images
def camera_callback(frame):
    try:
        # Publish processed image
        image_msg = bridge.cv2_to_imgmsg(frame, "bgr8")
        image_pub.publish(image_msg)
        # Perform object detection
        results = model(frame)
        # Extract keypoints from YOLO detections
        keypoints = results[0].keypoints.xy.cpu().numpy()
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        # Show the frame with keypoints
        cv2.imshow("KeyPoints", frame)
        cv2.waitKey(1)

        # Prepare 3D keypoints
        keypoints_3d = np.array([[0.01206,0.235,-0.21776],[0.00005,0.0813,-0.26776],[0.00005, 0.171, 0.19024],[0.16505, 0.2642, 0.10624],[0.16506, 0.2655, -0.16966],[-0.16495, 0.2641, 0.10524],[-0.16495, 0.2633, -0.17016],[-0.06495, 0.17695, -0.30763],
        [-0.01994,0.17695,-0.27316],[0.06006, 0.18075, -0.28716],[0.06505, 0.13695, -0.27976],[0.00005, 0.037, -0.26736],[-0.11994, 0.19595, -0.27163],[-0.11994, 0.15995, -0.27063],[0.11805, 0.19595, -0.27063],
        [0.11805, 0.15795, -0.27163],[0.17996,0.2377,-0.16876],[0.18006, 0.1211, -0.16876],[0.18075,0.238,0.10694],[0.18026,0.1226,0.10524],[0.03766,0.2073,0.14424],[0.03756,0.207,0.22124],[-0.03694,0.2073,0.14424],
        [-0.03654,0.2073,0.22164],[0.13015,0.1208, 0.22314],[-0.12935,0.1206,0.22254],[0.17955,0.1211,0.17304],[-0.17975,0.1206,0.17334],[-0.16495,0.2671,-0.05846],[0.06505, 0.13695, -0.27976]]) #these are obtained from Unity 3D game simulator

        # Estimate pose using PnP
        translation, rotation = estimate_pose(keypoints, keypoints_3d)
        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position.x = translation[0]
        pose_msg.pose.position.y = translation[1]
        pose_msg.pose.position.z = translation[2]
        quaternion = tf_transformations.quaternion_from_matrix(rotation)
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        pose_pub.publish(pose_msg)
    except CvBridgeError as e:
        print(e)

# Initialize video capture from network camera
rtsp_url = 'rtsp://admin:artpark123@192.168.0.219:554/Streaming/Channels/1/'
gstreamer_exe = 'gst-launch-1.0' 
p = Popen(shlex.split(f'{gstreamer_exe} --quiet rtspsrc location={rtsp_url} latency=0 ! queue2 ! rtph264depay ! avdec_h264 ! videoconvert ! capsfilter caps="video/x-raw, format=BGR" ! timeoverlay ! fdsink'), stdout=PIPE)
width = 1280
height  = 720
# Start ROS main loop
while not rospy.is_shutdown():
   raw_image = p.stdout.read(width * height * 3)
   frame = np.frombuffer(raw_image, np.uint8).reshape((height, width, 3))
    if ret:
        camera_callback(frame)
    else:
        rospy.logerr("Failed to capture frame from camera.")
# Release video capture
if rospy.is_shutdown():
            p.release()
            cv2.destroyAllWindows()
