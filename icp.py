import open3d as o3d
import numpy as np

# Load point clouds
initial_pcd = o3d.io.read_point_cloud("initial_point_cloud.pcd")
final_pcd = o3d.io.read_point_cloud("final_point_cloud.pcd")

# Visualize the initial and final point clouds
o3d.visualization.draw_geometries([initial_pcd, final_pcd], window_name='Initial and Final Point Clouds')

# Perform ICP alignment
threshold = 0.02  # Distance threshold for point matching
trans_init = np.eye(4)  # Initial transformation matrix (identity matrix)

print("Initial alignment")
evaluation = o3d.pipelines.registration.evaluate_registration(initial_pcd, final_pcd, threshold, trans_init)
print(evaluation)

print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    initial_pcd, final_pcd, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())

print("Transformation matrix:")
print(reg_p2p.transformation)

# Transform the initial point cloud to align with the final point cloud
transformed_initial_pcd = initial_pcd.transform(reg_p2p.transformation)

# Save the transformed initial point cloud
o3d.io.write_point_cloud("transformed_initial_point_cloud.pcd", transformed_initial_pcd)

# Visualize the final point cloud and the transformed initial point cloud
o3d.visualization.draw_geometries([transformed_initial_pcd, final_pcd], window_name='Transformed Initial and Final Point Clouds')
