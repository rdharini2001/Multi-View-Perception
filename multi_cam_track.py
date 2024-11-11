import cv2
import torch
from sort import Sort
import numpy as np
import math

# Load YOLO model
model = YOLO('/path/to/your/model.pt')

# Initialize SORT tracker
tracker = Sort()

# Camera matrices and distortion coefficients (for each camera)
camera_matrices = [
    np.array([[970.13975699, 0, 661.05696322],
              [0, 965.0683426, 324.24867006],
              [0, 0, 1]]),  # Camera 1 matrix
    np.array([[960.0, 0, 650.0],
              [0, 960.0, 320.0],
              [0, 0, 1]])   # Camera 2 matrix (example)
]

# Homography matrices (for each camera)
homographies = [
    np.matrix([[ 1.36836082e+00,  9.75493567e-01,  9.59715657e+02],
               [-2.81827277e-02, -4.62794652e-01,  7.48434225e+01],
               [-1.79816570e-04,  1.24865401e-03,  1.00000000e+00]]),  # Camera 1
    np.matrix([[ 1.0,  0.5,  950.0],
               [0.5, 1.0, 300.0],
               [0, 0, 1]])   # Camera 2 matrix
]

# Open video captures for both cameras
cap1 = cv2.VideoCapture('/path/to/video_camera_1.mp4')
cap2 = cv2.VideoCapture('/path/to/video_camera_2.mp4')

# Get video dimensions
width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize video writers
writer1 = cv2.VideoWriter('camera1_output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20, (width1, height1))
writer2 = cv2.VideoWriter('camera2_output.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20, (width2, height2))

# Trackers for each camera
track_lines_cam1 = []
track_lines_cam2 = []

# Initialize lists to store the tracks of robots
robots_tracks_cam1 = {}
robots_tracks_cam2 = {}

# Define parameters for track fusion and handoff
fusion_distance_threshold = 50  # max distance to consider objects as the same (set based on Mahalanobis distance)
handoff_distance_threshold = 75  # distance to trigger handoff of track between cameras
handoff_frame_buffer = 5  # number of frames to consider for handoff

# Helper function to calculate Euclidean distance
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Track fusion logic (fusion based on distance and ID association)
def fuse_tracks(cam1_tracks, cam2_tracks):
    fused_tracks = {}

    for cam1_id, cam1_position in cam1_tracks.items():
        closest_distance = float('inf')
        closest_cam2_id = None

        # Find the closest track from camera 2
        for cam2_id, cam2_position in cam2_tracks.items():
            dist = euclidean_distance(cam1_position, cam2_position)
            if dist < closest_distance and dist < fusion_distance_threshold:
                closest_distance = dist
                closest_cam2_id = cam2_id

        if closest_cam2_id is not None:
            fused_tracks[cam1_id] = cam2_tracks.pop(closest_cam2_id)
        else:
            fused_tracks[cam1_id] = cam1_position

    # Add remaining tracks from camera 2
    fused_tracks.update(cam2_tracks)
    return fused_tracks

# Handle track handoff (based on proximity and frame buffer)
def handle_handoff(cam1_tracks, cam2_tracks, handoff_frame_buffer=5):
    handoff_tracks = {}

    for cam1_id, cam1_position in cam1_tracks.items():
        for cam2_id, cam2_position in cam2_tracks.items():
            dist = euclidean_distance(cam1_position, cam2_position)
            if dist < handoff_distance_threshold:
                handoff_tracks[cam2_id] = cam1_position
                break  # Handoff track to camera 2

    # Add handoff tracks to camera 2's tracking
    cam2_tracks.update(handoff_tracks)
    return cam2_tracks

# Process each frame
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Detect objects (robots) in both cameras
    results1 = model(frame1)
    results2 = model(frame2)

    # Extract boxes and update trackers for Camera 1
    boxes1 = results1[0].boxes.xyxy.tolist()
    trackers1 = tracker.update(np.array(boxes1))

    # Process each tracked object in Camera 1
    for x1, y1, x2, y2, track_id in trackers1:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(frame1, (center_x, center_y), 4, (0, 255, 0), -1)
        cv2.putText(frame1, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        track_lines_cam1.append((center_x, center_y))
        robots_tracks_cam1[track_id] = (center_x, center_y)

    # Extract boxes and update trackers for Camera 2
    boxes2 = results2[0].boxes.xyxy.tolist()
    trackers2 = tracker.update(np.array(boxes2))

    # Process each tracked object in Camera 2
    for x1, y1, x2, y2, track_id in trackers2:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        cv2.circle(frame2, (center_x, center_y), 4, (0, 255, 0), -1)
        cv2.putText(frame2, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        track_lines_cam2.append((center_x, center_y))
        robots_tracks_cam2[track_id] = (center_x, center_y)

    # **Track Fusion**: Fuse the tracks from both cameras
    fused_tracks = fuse_tracks(robots_tracks_cam1, robots_tracks_cam2)

    # **Handle Handoff**: Handoff tracks from one camera to another
    robots_tracks_cam2 = handle_handoff(robots_tracks_cam1, robots_tracks_cam2, handoff_frame_buffer)

    # Write to video files
    writer1.write(frame1)
    writer2.write(frame2)

    # Display the result (optional)
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release video capture and writer objects
cap1.release()
cap2.release()
writer1.release()
writer2.release()
cv2.destroyAllWindows()
