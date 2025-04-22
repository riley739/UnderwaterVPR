import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# Function to read the 4x4 rotation matrices from a file
def read_file(file_path):
    poses = []  # List to store the extracted positions
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
                poses.append(position)
                matrix_values = []
    return names, np.array(poses)

# Read the poses (translations) from the file
file_path = '/home/rbeh9716/Desktop/holoocean/logs/2025-04-07_16-03-36/images.log'  # Your file path here
names, positions = read_file(file_path)

# Extract the x and y components of the positions for 2D plotting
x_positions = positions[:, 0]  # X positions
y_positions = positions[:, 1]  # Y positions
z_positions = positions[:, 2]  # Z positions

df = pd.DataFrame({
    'name': names,
    'x': x_positions,
    'y': y_positions,
    'z': z_positions
})


df.to_csv('holoocean.csv', index=False)