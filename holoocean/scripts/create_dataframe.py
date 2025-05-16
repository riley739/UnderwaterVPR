import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


def rotation_matrix_to_rpy(rotations):
    """
    Converts a 3x3 rotation matrix to roll, pitch, yaw (in radians)
    Using ZYX convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    rolls = []
    pitchs = [] 
    yaws = []

    for R in rotations:
        if abs(R[2, 0]) != 1:
            pitch = -np.arcsin(R[2, 0])
            roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
            yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
        else:
            # Gimbal lock: pitch = ±90°
            yaw = 0
            if R[2, 0] == -1:
                pitch = np.pi / 2
                roll = np.arctan2(R[0, 1], R[0, 2])
            else:
                pitch = -np.pi / 2
                roll = np.arctan2(-R[0, 1], -R[0, 2])
        rolls.append(roll)
        pitchs.append(pitch)
        yaws.append(yaw)

    return rolls, pitchs, yaws

# Function to read the 4x4 rotation matrices from a file
def read_file(file_path):
    poses = []  # List to store the extracted positions
    rotations = [] 
    names = []  # List to store the names
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix_values = []  # Temporary list to store matrix values
        for i,line in enumerate(lines):
            if line.startswith("Image: "):
                img_name =  line.replace("Image: ", "").strip()
                names.append(os.path.basename(img_name))
                continue
            line = line.replace("Pose: ", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            
            # Clean the line (remove unwanted spaces and newline)
            line = line.strip()
            
            # Convert the string into a list of floats
            matrix_values += map(float, line.split())
            if len(matrix_values) == 16:
                # Reshape into a 4x4 matrix
                matrix = np.array(matrix_values).reshape(4, 4)
                # Extract the position (translation) from the last column (tx, ty, tz)
                position = matrix[:3, 3]  # Extract translation vector [tx, ty, tz]
                rotation = matrix[:3, :3]
                poses.append(position)
                rotations.append(rotation)
                matrix_values = []
                
    return names, np.array(poses), rotations

# Read the poses (translations) from the file
file_path = "/home/rbeh9716/Desktop/UnderwaterVPR/holoocean/logs/2025-05-02_16-05-24/images.log"  # Your file path here
names, positions, rotations = read_file(file_path)

roll, pitch, yaw = rotation_matrix_to_rpy(rotations)

# Extract the x and y components of the positions for 2D plotting
x_positions = positions[:, 0]  # X positions
y_positions = positions[:, 1]  # Y positions
z_positions = positions[:, 2]  # Z positions

df = pd.DataFrame({
    'name': names,
    'x': x_positions,
    'y': y_positions,
    'z': z_positions,
    'roll': roll,
    'pitch': pitch,
    'yaw': yaw
})


df.to_csv('images.csv', index=False)