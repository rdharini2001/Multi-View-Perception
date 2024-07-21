import numpy as np
from scipy.spatial import cKDTree

# Define file paths
segmentation_file = 'centroids.txt'
keypoints_file = 'keypoints.txt'
output_file = 'semantic_keypoints.txt'

# Define a confidence threshold
confidence_threshold = 0.75

# Read centroid locations from the segmentation file
centroid_locations = np.loadtxt(segmentation_file, delimiter=',')

# Read keypoints with confidence scores
keypoints_data = np.loadtxt(keypoints_file, delimiter=',')
keypoints = keypoints_data[:, :2]  # Assuming first two columns are keypoint locations
confidence_scores = keypoints_data[:, 2]  # Assuming third column is confidence score

# Filter keypoints based on confidence scores
filtered_keypoints = keypoints[confidence_scores >= confidence_threshold]

# Assign semantic labels using nearest neighbor search
kdtree = cKDTree(centroid_locations)
distances, indices = kdtree.query(filtered_keypoints)

# Prepare the final semantic keypoints with labels
semantic_keypoints = np.hstack((filtered_keypoints, indices.reshape(-1, 1)))

# Save the final semantic keypoints to a separate text file
np.savetxt(output_file, semantic_keypoints, delimiter=',', fmt='%f')

print(f"Final semantic keypoints saved to {output_file}")
