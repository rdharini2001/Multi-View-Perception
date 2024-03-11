import cv2 
import torch
from sort import Sort
from pathlib import Path
from ultralytics import YOLO
import imutils
import math
import numpy as np 
import random
model = YOLO('/home/rbccps/Desktop/pose estimation/train46.pt')

# Initialize SORT tracker
tracker = Sort()
homo_mat = np.matrix([[ 1.36836082e+00,  9.75493567e-01,  9.59715657e+02],
                                        [-2.81827277e-02, -4.62794652e-01,  7.48434225e+01],
                                        [-1.79816570e-04,  1.24865401e-03,  1.00000000e+00]])
cam_mat = np.array([[970.13975699,   0.        , 661.05696322],
                                   [  0.        , 965.0683426 , 324.24867006],
                                   [  0.        ,    0.       ,   1.        ]])

dist_mat = np.array([-0.44779831, 0.21493212, 0.0086979, -0.00269077, 0.00281984])

# Open video capture
cap = cv2.VideoCapture('/home/rbccps/Desktop/pose estimation/cam219vid.mp4')  # Replace with the path to your video file
width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer= cv2.VideoWriter('CAMERA219_march1.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20, (width,height))
# Initialize variables for tracking lines
track_lines = []

while True:
    ret, frame = cap.read()
    #frame = imutils.resize(frame, width=800, height=800)
    if not ret:
        break
    results = model(frame)
    box = results[0].boxes.xyxy.tolist()
    for x1, y1, x2, y2 in box:
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            yaw1 = math.atan2((y2-y1),(x2-x1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Update tracking line with the center of the bounding box
            center_x = int((x1 + x2) / 2.0)
            center_y = int((y1 + y2) / 2.0)
            cv2.circle(frame, (center_x, center_y), 4, (0, 255, 0), -1)
            track_lines.append((center_x, center_y))
            inv_homo_mat = np.linalg.inv(homo_mat)
            point = np.array([center_x, center_y, 1])
            world_coord=np.dot(inv_homo_mat,point)
            scalar=world_coord[0,2]
            frame_text = "3D Pose x: {:.2f}, y: {:.2f}, yaw: {:.2f}".format(world_coord[0,0]/scalar, world_coord[0,1]/scalar, yaw1)
            cv2.putText(frame,frame_text,(6, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
         # Draw tracking lines
    if len(track_lines) > 1:
            for i in range(1, len(track_lines)):
                    cv2.line(frame, track_lines[i - 1], track_lines[i], (255, 0, 0), 2)
    writer.write(frame)
    # Display the result
    #cv2.imshow('Object Tracking', frame)
    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
                         break
cap.release()
cv2.destroyAllWindows()
