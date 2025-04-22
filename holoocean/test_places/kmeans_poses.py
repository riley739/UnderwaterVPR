import json
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

#TODO THIS ONLY LOOKS AT X AND Y  AND NOTHING ELSE change
# translations.append(matrix_array[:2, 3])
# TO 
# translations.append(matrix_array[:3, 3])
# TO also care about the z axis 
# Camera Rotation is not accounted for 


# Load SE(3) transformation matrices from JSON file
json_file = "poses.json"  # Change this to your actual JSON file
with open(json_file, "r") as file:
    image_data = json.load(file)

se3_data = image_data["frames"]
# Convert JSON data to NumPy arrays and extract translation vectors
translations = []
image_ids = []  # To keep track of pose IDs
for image in se3_data:
    matrix_array = np.array(image["pose"], dtype=np.float64)
    if matrix_array.shape == (4, 4):  # Ensure it's a valid SE(3) matrix
        translations.append(matrix_array[:2, 3])  # Extract (TX, TY)
        image_ids.append(image["image_name"])

translations = np.array(translations)  # Convert to NumPy array

# Set clustering parameters
max_distance = 2 # Maximum allowed Euclidean distance within a cluster
num_clusters = min(10, len(translations) // 2)  # Initial number of clusters

output_json = f"pose_{max_distance}.json"
# Perform constrained K-Means clustering
while True:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(translations)
    
    # Compute distances of each point to its cluster center
    cluster_centers = kmeans.cluster_centers_
    distances = cdist(translations, cluster_centers, metric="euclidean")
    
    # Find max distance per cluster
    max_distances = np.zeros(num_clusters)
    for i in range(len(translations)):
        cluster_idx = labels[i]
        max_distances[cluster_idx] = max(max_distances[cluster_idx], distances[i, cluster_idx])
    
    # Check if any cluster exceeds the max distance
    if np.all(max_distances <= max_distance):
        break  # Stop if all clusters satisfy the constraint
    else:
        num_clusters += 1  # Increase the number of clusters and retry

# Assign each image ID to its cluster
clustered_poses = {str(i): [] for i in range(num_clusters)}
for i, image_id in enumerate(image_ids):
    se3_data[i]["place_id"] = str(labels[i])
    cluster_id = labels[i]
    clustered_poses[str(cluster_id)].append(image_id)

image_data["frames"] = se3_data
# Save clustered results
with open(output_json, "w") as f:
    json.dump(output_json, f, indent=4)


# Save clustered results
output_json = f"clustered_se3_poses_{max_distance}.json"
with open(output_json, "w") as f:
    json.dump(clustered_poses, f, indent=4)

print(f"Clustering completed with {num_clusters} clusters. Results saved to {json_file}.")
