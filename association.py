import numpy as np
from scipy.optimize import linear_sum_assignment

def mahalanobis_distance(prediction, measurement, covariance):
    innovation = measurement - prediction
    inv_covariance = np.linalg.inv(covariance)
    mahalanobis_dist = np.sqrt(np.dot(np.dot(innovation.T, inv_covariance), innovation))
    return mahalanobis_dist

def mahalanobis_gating(track_predictions, measurements, covariances, gating_threshold):
    num_tracks = len(track_predictions)
    num_measurements = len(measurements)

    gated_associations = []

    for i in range(num_tracks):
        for j in range(num_measurements):
            mahalanobis_dist = mahalanobis_distance(track_predictions[i], measurements[j], covariances[i])

            if mahalanobis_dist < gating_threshold:
                gated_associations.append((i, j))

    return gated_associations

def multi_sensor_data_association(track_predictions, imu_measurements, lidar_measurements, camera_measurements, wheel_encoder_measurements, external_camera_measurements, gating_threshold=5.0):
    all_gated_associations = []

    for track_pred, imu_meas, lidar_meas, camera_meas, wheel_encoder_meas, external_camera_meas in zip(track_predictions, imu_measurements, lidar_measurements, camera_measurements, wheel_encoder_measurements, external_camera_measurements):
        measurements = np.concatenate((imu_meas, lidar_meas, camera_meas, wheel_encoder_meas, external_camera_meas), axis=0)

        # Assuming that covariances for each track are known or can be estimated
        covariances = [np.eye(measurements.shape[1]) for _ in range(len(track_pred))]

        gated_associations = mahalanobis_gating(track_pred, measurements, covariances, gating_threshold)
        all_gated_associations.append(gated_associations)

    # Apply the Hungarian algorithm for assignment across all tracks and measurements
    all_associations = np.concatenate(all_gated_associations)
    row_ind, col_ind = linear_sum_assignment(np.ones_like(all_associations))

    final_associations = [(row_ind[i], col_ind[i]) for i in range(len(row_ind))]

    return final_associations


'''track_predictions_sensor = np.array([[1, 1, 1, 1, 1], [5, 5, 5, 5, 5], [10, 10, 10, 10, 10]])
imu_measurements_sensor = np.array([[1.1, 1.2, 1.3], [5.1, 5.2, 5.3]])
lidar_measurements_sensor = np.array([[1.01, 1.02, 1.03], [5.01, 5.02, 5.03]])
camera_measurements_sensor = np.array([[1.001, 1.002], [5.001, 5.002]])
wheel_encoder_measurements_sensor = np.array([[0.1], [0.5]])
external_camera_measurements_sensor = np.array([[0.9, 0.8], [4.9, 4.8]])

track_predictions = [track_predictions_sensor]
imu_measurements = [imu_measurements_sensor]
lidar_measurements = [lidar_measurements_sensor]
camera_measurements = [camera_measurements_sensor]
wheel_encoder_measurements = [wheel_encoder_measurements_sensor]
external_camera_measurements = [external_camera_measurements_sensor]

associations = multi_sensor_data_association(track_predictions, imu_measurements, lidar_measurements, camera_measurements, wheel_encoder_measurements, external_camera_measurements)

print("Associations:", associations)'''
