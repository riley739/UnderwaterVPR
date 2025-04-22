
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
import pandas as pd  

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

def assign_place_ids(positions, distance_threshold=5):
    place_ids = []
    centroids = []
    current_id = 0

    for pos in positions:
        found = False
        for i, centroid in enumerate(centroids):
            if np.linalg.norm(pos - centroid) < distance_threshold:
                place_ids.append(i)
                found = True
                break
        if not found:
            centroids.append(pos)
            place_ids.append(current_id)
            current_id += 1

    return np.array(place_ids)

file_path = '/home/rbeh9716/Desktop/holoocean/logs/2025-04-07_16-03-36/images.log' 
names, poses = read_file(file_path) 


place_ids = assign_place_ids(poses[:,:2])
# # Method 3: Cluster using DBSCAN
# db = DBSCAN(eps=1, min_samples=4)  # eps is the distance threshold
# place_ids = db.fit_predict(poses[:,:2])

# Plot the results, color-coded by place_id
plt.figure(figsize=(8, 6))
scatter = plt.scatter(poses[:, 0], poses[:, 1], c=place_ids, cmap='tab20', s=60)
plt.title("Clustered Places (Color-coded by Place ID)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.colorbar(scatter, label="Place ID")
plt.tight_layout()
plt.savefig("scripts/places.png")

x_positions = poses[:, 0]  # X positions
y_positions = poses[:, 1]  # Y positions
z_positions = poses[:, 2]  # Z positions


df = pd.DataFrame({
    'image_name': names,
    'x': x_positions,
    'y': y_positions,
    'z': z_positions,
    "place_id" : place_ids
})


df.to_csv('scripts/holoocean.csv', index=False)


