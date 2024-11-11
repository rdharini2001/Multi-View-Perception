import numpy as np
import pandas a pd

import numpy as np

def compute_centroids_and_covariances(segmentation_parts):
    results = {}
    
    for part_id, segmentation_mask in segmentation_parts.items():
        # Find pixel coordinates of the segmented part
        coords = np.argwhere(segmentation_mask == 1)  # Assuming 1 represents the part
        
        # Check if there are any coordinates in the segmentation
        if coords.size == 0:
            print(f"Warning: No segmented area found for part ID {part_id}.")
            continue
        
        # Compute the centroid (mean)
        centroid = np.mean(coords, axis=0)

        # Compute the covariance matrix of the pixel coordinates
        covariance_matrix = np.cov(coords.T)
        
        # Store the centroid and covariance matrix in the results dictionary
        results[part_id] = {
            'centroid': centroid,
            'covariance': covariance_matrix
        }
    
    return results

from sklearn.metrics import pairwise_distances

def compute_keypoint_uncertainty(keypoints, confidences, sigma_o, beta):
    """
    Compute uncertainty for each keypoint based on the detection confidence and position.
    """
    uncertainties = []
    for i, (keypoint, confidence) in enumerate(zip(keypoints, confidences)):
        # Distance from the center (assumed to be the center of the image)
        distance = np.linalg.norm(keypoint - np.array([keypoint.shape[1] / 2, keypoint.shape[0] / 2]))
        
        # Compute the uncertainty covariance
        uncertainty = sigma_o**2 * (1 + beta * distance**2) * np.eye(2) / confidence
        uncertainties.append(uncertainty)
    
    return uncertainties

def compute_3d_unprojection(pixel_coords, depth, intrinsic_matrix):
    """
    Compute 3D unprojection of pixel coordinates using depth and camera intrinsic matrix.
    """
    u, v = pixel_coords
    d = depth

    # Pinhole camera model inverse projection
    inv_K = np.linalg.inv(intrinsic_matrix)
    point_3d = inv_K @ np.array([u * d, v * d, d])

    return point_3d

def propagate_uncertainty(pixel_uncertainty, depth_uncertainty, jacobian):
    """
    Propagate uncertainty through the Jacobian matrix.
    """
    total_uncertainty = jacobian @ np.block([[pixel_uncertainty, np.zeros((2, 1))],
                                             [np.zeros((1, 2)), depth_uncertainty]]) @ jacobian.T
    return total_uncertainty


from sklearn.neighbors import NearestNeighbors
import open3d as o3d

def compute_3d_pose(keypoints_2d, keypoints_3d, depth_image, intrinsic_matrix):
    """
    Estimate 6D pose using 2D and 3D keypoints and depth information.
    """
    # Unproject keypoints from 2D to 3D using depth information
    keypoints_3d_computed = []
    for k in keypoints_2d:
        depth = depth_image[k[1], k[0]]  # Get depth value from the depth image
        keypoints_3d_computed.append(compute_3d_unprojection(k, depth, intrinsic_matrix))

    # Perform ICP to align 2D and 3D keypoints
    # Assuming keypoints_3d_computed are 3D points in camera space
    source_points = np.array(keypoints_3d_computed)
    target_points = np.array(keypoints_3d)

    pcd_source = o3d.geometry.PointCloud()
    pcd_target = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source_points)
    pcd_target.points = o3d.utility.Vector3dVector(target_points)

    # Apply ICP to align point clouds
    reg_icp = o3d.pipelines.registration.registration_icp(
        pcd_source, pcd_target, 0.05, np.eye(4), 
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    transformation = reg_icp.transformation
    return transformation  # This gives the 6D pose (rotation + translation)


def combine_uncertainty(uncertainty_sam, uncertainty_alike, uncertainty_depth, uncertainty_centroids, uncertainty_keypoints):
    """
    Combine uncertainty from all stages and propagate it.
    """
    total_uncertainty = uncertainty_sam + uncertainty_alike + uncertainty_depth + uncertainty_centroids + uncertainty_keypoints
    return total_uncertainty


def full_pipeline(input_image, segmentation_model, keypoint_model, depth_model, intrinsic_matrix):
    # Apply SAM for segmentation and compute uncertainty
    segmentation_mask, uncertainty_sam = mc_dropout_inference(segmentation_model, input_image)
    
    # Compute part-wise centroids and associated uncertainties
    centroids, uncertainty_centroids = [], []
    for part_mask in segmentation_mask:
        centroid, centroid_cov = compute_centroid_and_uncertainty(part_mask)
        centroids.append(centroid)
        uncertainty_centroids.append(centroid_cov)
    
    # Apply ALIKE for keypoint detection and compute uncertainty
    keypoints, confidences = keypoint_model(input_image)
    uncertainty_keypoints = compute_keypoint_uncertainty(keypoints, confidences, sigma_o=0.1, beta=0.01)
    
    # Estimate depth using ZoeDepth and unproject 2D to 3D
    depth_map = depth_model(input_image)
    uncertainty_depth = np.var(depth_map)  # Simple variance for depth uncertainty
    # Propagate depth uncertainty through unprojection (Jacobian)
    uncertainty_depth_propagated = propagate_uncertainty(uncertainty_keypoints, uncertainty_depth, jacobian_matrix)
    
    # Combine uncertainty from all models
    total_uncertainty = combine_uncertainty(uncertainty_sam, uncertainty_keypoints, uncertainty_depth_propagated, uncertainty_centroids)
    
    # Estimate 3D pose using SfM, ICP, and combine uncertainties
    keypoints_3d = get_3d_keypoints_from_model()  # Placeholder: function to get 3D keypoints from model
    pose_estimation = compute_3d_pose(keypoints, keypoints_3d, depth_map, intrinsic_matrix)
    
    return pose_estimation, total_uncertainty


