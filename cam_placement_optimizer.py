import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

def visibility_model(x, y, d, a):
    """
    Visibility model function.
    x: x-coordinate of the point
    y: y-coordinate of the point
    d: maximum allowable covariance
    a: field of view angle
    """
    return (x <= d) and (y <= (a * x) / (2 * d))

def objective_function(x, n, m, beta):
    """
    Objective function to minimize the number of cameras.
    x: array containing the positions and orientations of the cameras
    n: number of rows
    m: number of columns
    beta: field of view angle (in degrees)
    """
    c = x[:n*m].reshape((n, m))
    d = x[n*m:-1].reshape((n, m))
    a = x[-1]
    return np.sum(c)

def constraint_function(x, n, m, beta, d_max, covariance_threshold):
    """
    Constraint function to ensure coverage and maximum allowable covariance.
    x: array containing the positions and orientations of the cameras
    n: number of rows
    m: number of columns
    beta: field of view angle (in degrees)
    d_max: maximum allowable distance
    covariance_threshold: maximum allowable covariance threshold
    """
    c = x[:n*m].reshape((n, m))
    d = x[n*m:-1].reshape((n, m))
    a = x[-1]
    constraints = []
    for i in range(n):
        for j in range(m):
            for i1 in range(n):
                for j1 in range(m):
                    if np.sqrt((i-i1)**2 + (j-j1)**2) <= d_max:
                        if (i, j) != (i1, j1):
                            cov = np.sqrt((i - i1) ** 2 + ((j - j1) / 2) ** 2)
                            if cov <= d[i, j]:
                                if visibility_model(i - i1, j - j1, d[i, j], a * np.pi / 180):
                                    constraints.append(1 - c[i, j] + c[i, j] * c[i1, j1])
                                else:
                                    constraints.append(c[i, j] * c[i1, j1])
                            else:
                                constraints.append(c[i, j] * c[i1, j1])
    return np.array(constraints)

def camera_placement_algorithm(n, m, d_max, covariance_threshold, beta):
    """
    Camera placement algorithm using Lagrange multipliers.
    n: number of rows
    m: number of columns
    d_max: maximum allowable distance
    covariance_threshold: maximum allowable covariance threshold
    beta: field of view angle (in degrees)
    """
    # Initial guess for the positions and orientations of the cameras
    x0 = np.ones(n * m + n * m + 1)

    # Bounds for the positions and orientations of the cameras
    bounds = [(0, 1) for _ in range(n * m)] + \
             [(0, d_max) for _ in range(n * m)] + \
             [(0, 90)]

    # Constraint for each camera to cover at least one grid cell
    cons = ({'type': 'ineq', 'fun': lambda x: np.sum(x[:n*m].reshape((n, m)), axis=1) - 1})

    # Solve the optimization problem
    res = minimize(objective_function, x0, args=(n, m, beta), constraints=cons,
                   bounds=bounds, method='SLSQP')

    if res.success:
        print("Optimization successful.")
        c = res.x[:n*m].reshape((n, m))
        print("Number of cameras:", np.sum(c))
        print("Camera positions and orientations:")
        print(c)
        return c
    else:
        print("Optimization failed.")
        return None

def divide_into_grids(img, n, m):
    """
    Divide the image into n x m occupancy grids.
    img: PIL Image object
    n: number of rows
    m: number of columns
    """
    width, height = img.size
    grid_width = width // m
    grid_height = height // n
    grids = []
    for i in range(n):
        for j in range(m):
            left = j * grid_width
            top = i * grid_height
            right = left + grid_width
            bottom = top + grid_height
            grids.append((left, top, right, bottom))
    return grids

def plot_camera_positions(img, camera_positions, grids):
    """
    Plot the camera positions on the map.
    img: PIL Image object
    camera_positions: 2D array indicating camera positions
    grids: list of tuples representing grid coordinates
    """
    fig, ax = plt.subplots()
    ax.imshow(img)
    for i in range(len(camera_positions)):
        for j in range(len(camera_positions[0])):
            if camera_positions[i][j] == 1:
                rect = Rectangle((grids[i * len(camera_positions[0]) + j][0], grids[i * len(camera_positions[0]) + j][1]),
                                 grids[i * len(camera_positions[0]) + j][2] - grids[i * len(camera_positions[0]) + j][0],
                                 grids[i * len(camera_positions[0]) + j][3] - grids[i * len(camera_positions[0]) + j][1],
                                 linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
    ax.set_aspect('equal')
    plt.show()

# Load the map image
img = Image.open("map.jpg")  # Replace "map.jpg" with the path to your map image

# Parameters
n = 5  # Number of rows
m = 5  # Number of columns
d_max = 5  # Maximum allowable distance
covariance_threshold = 0.1  # Maximum allowable covariance threshold
field_of_view = 45  # Field of view angle (in degrees)

# Run the camera placement algorithm
camera_positions = camera_placement_algorithm(n, m, d_max, covariance_threshold, field_of_view)

# Divide the image into occupancy grids
grids = divide_into_grids(img, n, m)

# Plot the camera positions on the map
plot_camera_positions(img, camera_positions, grids)
