import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from scipy.optimize import minimize

def generate_binary_map(rgb_image, threshold=128):
    grayscale_image = np.mean(rgb_image, axis=-1)
    binary_map = (grayscale_image > threshold).astype(int)
    return binary_map

def coverage_matrix(camera_position, grid_size, fov_angle):
    coverage_matrix = np.zeros((grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            angle = np.arctan2(i - camera_position[0], j - camera_position[1])
            distance = np.sqrt((i - camera_position[0])**2 + (j - camera_position[1])**2)
            if np.abs(angle) < fov_angle / 2 and distance <= grid_size / 2:
                coverage_matrix[i, j] = 1
    return coverage_matrix

def total_coverage(positions, grid_size, fov_angle):
    num_cameras = len(positions) // 2
    coverage_matrices = [coverage_matrix(positions[i:i+2], grid_size, fov_angle) for i in range(0, len(positions), 2)]
    total_coverage = np.sum(np.stack(coverage_matrices), axis=0)
    return total_coverage

def covariance_penalty(covariance_matrix):
    covariance_penalty = np.trace(covariance_matrix)
    return covariance_penalty

def objective_function(positions, grid_size, fov_angle, weight_coverage, weight_covariance):
    total_coverage_value = np.sum(total_coverage(positions, grid_size, fov_angle))
    covariance_matrix = np.cov(positions.reshape((-1, 2)).T)
    covariance_penalty_value = covariance_penalty(covariance_matrix)
    combined_objective = -weight_coverage * total_coverage_value + weight_covariance * covariance_penalty_value
    return combined_objective

def optimize_camera_placement(rgb_image, num_cameras, grid_size, fov_angle, weight_coverage=1, weight_covariance=1):
    # Initial guess for camera positions
    initial_positions = np.random.rand(num_cameras * 2) * grid_size

    # Optimization using scipy minimize
    result = minimize(
        objective_function,
        initial_positions,
        args=(grid_size, fov_angle, weight_coverage, weight_covariance),
        method='L-BFGS-B',  # Limited-memory Broyden-Fletcher-Goldfarb-Shanno
        bounds=[(0, grid_size)] * num_cameras * 2
    )

    optimal_positions = result.x

    return optimal_positions

def plot_map_with_cameras(rgb_image, optimal_positions, grid_size, fov_angle):
    plt.imshow(rgb_image)

    num_cameras = len(optimal_positions) // 2
    for i in range(0, len(optimal_positions), 2):
        camera_position = optimal_positions[i:i+2]
        coverage_matrix_data = coverage_matrix(camera_position, grid_size, fov_angle)
        covered_cells = np.transpose(np.where(coverage_matrix_data == 1))
        plt.scatter(covered_cells[:, 1], covered_cells[:, 0], c='red', marker='.', alpha=0.3)

    plt.scatter(optimal_positions[0::2], optimal_positions[1::2], c='red', marker='x', label='Cameras')
    plt.title('Optimal Camera Placement')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.show()

# Example usage
image_path = 'path_to_your_image.jpg'
rgb_image = io.imread(image_path)
num_cameras = 3
grid_size = 50
fov_angle = np.radians(30)
weight_coverage = 0.5
weight_covariance = 1

binary_map = generate_binary_map(rgb_image)
optimal_positions = optimize_camera_placement(rgb_image, num_cameras, grid_size, fov_angle, weight_coverage, weight_covariance)
plot_map_with_cameras(rgb_image, optimal_positions, grid_size, fov_angle)
