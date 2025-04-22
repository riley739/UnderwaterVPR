
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Generate sample pose data (x, y positions only for visualization)
np.random.seed(42)
poses = np.vstack([
    np.random.normal(loc=(0, 0), scale=1.0, size=(20, 2)),
    np.random.normal(loc=(10, 10), scale=1.0, size=(20, 2)),
    np.random.normal(loc=(20, 0), scale=1.0, size=(20, 2))
])

print(poses)

# Method 3: Cluster using DBSCAN
db = DBSCAN(eps=2.5, min_samples=1)  # eps is the distance threshold
place_ids = db.fit_predict(poses)

# Plot the results, color-coded by place_id
plt.figure(figsize=(8, 6))
scatter = plt.scatter(poses[:, 0], poses[:, 1], c=place_ids, cmap='tab20', s=60)
plt.title("Clustered Places (Color-coded by Place ID)")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.colorbar(scatter, label="Place ID")
plt.tight_layout()
plt.savefig("image.png")
