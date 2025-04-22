import numpy as np
import matplotlib.pyplot as plt

# Function to read the 4x4 rotation matrices from a file
def read_rotation_matrices(file_path):
    poses = []  # List to store the extracted positions
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix_values = []  # Temporary list to store matrix values
        for i,line in enumerate(lines):
            line = line.replace("Pose: ", "")
            line = line.replace("[", "")
            line = line.replace("]", "")
            
            # Clean the line (remove unwanted spaces and newline)
            line = line.strip()
            print(line)
            
            # Convert the string into a list of floats
            matrix_values += map(float, line.split())
            print
            if len(matrix_values) == 16:
                # Reshape into a 4x4 matrix
                matrix = np.array(matrix_values).reshape(4, 4)
                # Extract the position (translation) from the last column (tx, ty, tz)
                position = matrix[:3, 3]  # Extract translation vector [tx, ty, tz]
                poses.append(position)
                matrix_values = []
    return np.array(poses)

# Read the poses (translations) from the file
file_path = '/home/rbeh9716/Desktop/holoocean/logs/2025-04-07_16-03-36/pose.log'  # Your file path here
positions = read_rotation_matrices(file_path)

# Extract the x and y components of the positions for 2D plotting
x_positions = positions[:, 0]  # X positions
y_positions = positions[:, 1]  # Y positions

# Plot the 2D trajectory
plt.figure(figsize=(8, 6))
plt.plot(x_positions, y_positions, marker='o', linestyle='-', color='b', label='Trajectory')
plt.title("2D Trajectory of the Object")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.legend()
plt.show()
