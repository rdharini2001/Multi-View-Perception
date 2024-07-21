import rospy
import numpy as np
from scipy.optimize import linear_sum_assignment
from sensor_msgs.msg import Imu, PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

# Global variables to store sensor measurements and covariances
imu_measurements = []
lidar_measurements = []
camera_measurements = []
wheel_encoder_measurements = []
external_camera_measurements = []
track_predictions = []
camera_covariances = []  # Store covariance matrices for cameras

def imu_callback(msg):
    global imu_measurements
    imu_data = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
    imu_measurements.append(imu_data)

def lidar_callback(msg):
    global lidar_measurements
    lidar_data = np.array([point.x for point in msg.points])
    lidar_measurements.append(lidar_data)

def camera_callback(msg):
    global camera_measurements, camera_covariances
    camera_data = np.array(msg.data)  # Assuming data is a flat array of features
    camera_covariances.append(camera_zero_cov) 
    camera_measurements.append(camera_data)

def wheel_encoder_callback(msg):
    global wheel_encoder_measurements
    wheel_data = np.array([msg.data])
    wheel_encoder_measurements.append(wheel_data)

def external_camera_callback(msg):
    global external_camera_measurements
    pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
    covariance = np.array(msg.covariance).reshape(6, 6)
    external_camera_measurements.append((pose, covariance))

def stat_distance(prediction, measurement, covariance):
    innovation = measurement - prediction
    inv_covariance = np.linalg.inv(covariance)
    mahalanobis_dist = np.sqrt(np.dot(np.dot(innovation.T, np.norm(inv_covariance)), innovation))
    return mahalanobis_dist

def stat_gating(track_predictions, measurements, covariances, gating_threshold):
    num_tracks = len(track_predictions)
    num_measurements = len(measurements)
    gated_associations = []
    for i in range(num_tracks):
        for j in range(num_measurements):
            stat_dist = stat_distance(track_predictions[i], measurements[j], covariances[i])
            if stat_dist < gating_threshold:
                gated_associations.append((i, j))
    return gated_associations

def associate_measurements(track_predictions, measurements, covariances, gating_threshold):
    gated_associations = stat_gating(track_predictions, measurements, covariances, gating_threshold)
    if len(gated_associations) > 0:
        # Create a cost matrix for Hungarian algorithm
        cost_matrix = np.ones((len(track_predictions), len(measurements)))
        for i, j in gated_associations:
            cost_matrix[i, j] = np.linalg.norm(measurements[j] - track_predictions[i])
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        final_associations = [(row_ind[i], col_ind[i]) for i in range(len(row_ind))]
        return final_associations
    else:
        return []

def handle_non_overlapping_views(track_predictions, imu_measurements, lidar_measurements, camera_measurements, wheel_encoder_measurements, external_camera_measurements, gating_threshold=5.0):
    measurements = np.concatenate((imu_measurements, lidar_measurements, camera_measurements, wheel_encoder_measurements, external_camera_measurements), axis=0)
    covariances = [np.eye(measurements.shape[1]) for _ in range(len(track_predictions))]
    return associate_measurements(track_predictions, measurements, covariances, gating_threshold)

def handle_overlapping_views(track_predictions, imu_measurements, lidar_measurements, camera_measurements, wheel_encoder_measurements, external_camera_measurements, gating_threshold=5.0):
    # First stage: Associate multiple camera measurements
    camera_meas_combined = np.vstack(camera_measurements)
    camera_cov_combined = np.vstack(camera_covariances)
    camera_associations = associate_measurements(track_predictions, camera_meas_combined, camera_cov_combined, gating_threshold)
    
    # Second stage: Associate all other sensor measurements
    imu_meas_combined = np.vstack(imu_measurements)
    lidar_meas_combined = np.vstack(lidar_measurements)
    wheel_encoder_meas_combined = np.vstack(wheel_encoder_measurements)
    external_camera_meas_combined = np.vstack([ec[0] for ec in external_camera_measurements])
    external_camera_cov_combined = np.vstack([ec[1] for ec in external_camera_measurements])
    
    combined_measurements = np.vstack((imu_meas_combined, lidar_meas_combined, wheel_encoder_meas_combined, external_camera_meas_combined))
    combined_covariances = np.vstack((np.eye(imu_meas_combined.shape[1]), np.eye(lidar_meas_combined.shape[1]), np.eye(wheel_encoder_meas_combined.shape[1]), np.eye(external_camera_meas_combined.shape[1])))

    return associate_measurements(track_predictions, combined_measurements, combined_covariances, gating_threshold)

def multi_robot_tracker_node():
    rospy.init_node('multi_robot_tracker_node')

    # Subscriptions
    rospy.Subscriber('imu_topic', Imu, imu_callback)
    rospy.Subscriber('lidar_topic', PointCloud2, lidar_callback)
    rospy.Subscriber('camera_topic', Float32MultiArray, camera_callback)
    rospy.Subscriber('wheel_encoder_topic', Float32MultiArray, wheel_encoder_callback)
    rospy.Subscriber('external_camera_topic', PoseStamped, external_camera_callback)

    # Publisher
    pub_associations = rospy.Publisher('robot_associations', Float32MultiArray, queue_size=10)

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        if imu_measurements and lidar_measurements and camera_measurements and wheel_encoder_measurements and external_camera_measurements:
            if overlapping_views:
                associations = handle_overlapping_views(
                    track_predictions,
                    imu_measurements,
                    lidar_measurements,
                    camera_measurements,
                    wheel_encoder_measurements,
                    external_camera_measurements
                )
            else:
                associations = handle_non_overlapping_views(
                    track_predictions,
                    imu_measurements,
                    lidar_measurements,
                    camera_measurements,
                    wheel_encoder_measurements,
                    external_camera_measurements
                )
            
            # Publish associations
            msg = Float32MultiArray()
            msg.data = [float(a) for a in associations]
            pub_associations.publish(msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        multi_robot_tracker_node()
    except rospy.ROSInterruptException:
        pass
