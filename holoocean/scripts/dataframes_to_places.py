
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os
import pandas as pd  

def get_poses(dfs):
    poses = [] 
    for df in dfs:
        for i,row in df.iterrows():
            poses.append(np.array([row["x"], row["y"]]))

    return np.array(poses)

def assign_place_ids(positions, distance_threshold=10):
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

dataframes = [
                '/home/rbeh9716/Desktop/UnderwaterVPR/data/val/holoocean/database/images.csv',
                '/home/rbeh9716/Desktop/UnderwaterVPR/data/val/holoocean/query/images.csv'
]

dfs = [] 
for df in dataframes:
    dfs.append(pd.read_csv(df))



poses = get_poses(dfs) 


place_ids = assign_place_ids(poses)
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
plt.savefig("places.png")



first_db_size = len(dfs[0])

db_places = set()
query_places = set()
dfs[0]["place_id"] = 0
dfs[1]["place_id"] = 0

for i,place in enumerate(place_ids):
    if i < first_db_size: 
        db_places.add(place)
        dfs[0].at[i, "place_id"] = place
    else:
        if place not in db_places:
            print(f"Warning Query place id {place} is not represented in database")
        dfs[1].at[ i - first_db_size, "place_id"] = place


dfs[0].to_csv('database_updated.csv', index=False)
dfs[1].to_csv('query_updated.csv', index=False)


# x_positions = poses[:, 0]  # X positions
# y_positions = poses[:, 1]  # Y positions
# z_positions = poses[:, 2]  # Z positions


# df = pd.DataFrame({
#     'image_name': names,
#     'x': x_positions,
#     'y': y_positions,
#     'z': z_positions,
#     "place_id" : place_ids
# })


# df.to_csv('scripts/holoocean.csv', index=False)


