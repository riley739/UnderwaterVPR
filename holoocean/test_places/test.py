import matplotlib.pyplot as plt
import numpy as np

# Define grid size
num_rows, num_cols = 5, 10

# Generate dummy images (random noise) - replace with your actual images
images = [np.random.rand(64, 64) for _ in range(num_rows * num_cols)]  # 64x64 grayscale images

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))  # Adjust figure size

# Display images
for ax, img in zip(axes.flat, images):
    ax.imshow(img, cmap='gray')  # Display each image
    ax.axis('off')  # Hide axes

plt.tight_layout()
plt.show()
