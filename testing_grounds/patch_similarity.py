import torch
import torchvision.transforms as transforms
from PIL import Image

# Load DINOv2 model (ViT-S/14)
model = torch.hub.load('facebookresearch/dinov2', "dinov2_vits14")
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((518, 518)),  # Resize to ensure full patch division
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load and preprocess an image
image = Image.open("your_image.jpg").convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

# Extract patchwise features
with torch.no_grad():
    features = model.forward_features(image_tensor)

patch_tokens = features["x_norm_patchtokens"]  # (1, num_patches, feature_dim)
patch_tokens = patch_tokens.squeeze(0).cpu().numpy()  # Remove batch dim


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Reduce feature dimensions using t-SNE
tsne = TSNE(n_components=2, perplexity=10, random_state=42)
patch_embeddings_2d = tsne.fit_transform(patch_tokens)

# Reshape to grid (assuming 14x14 patches for ViT-S/14)
num_patches = int(np.sqrt(patch_tokens.shape[0]))  # E.g., 14 for 14x14 patches
patch_grid_x = patch_embeddings_2d[:, 0].reshape(num_patches, num_patches)
patch_grid_y = patch_embeddings_2d[:, 1].reshape(num_patches, num_patches)

# Plot the image
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(image)

# Overlay patch embeddings
for i in range(num_patches):
    for j in range(num_patches):
        ax.text(j * (image.size[0] // num_patches), i * (image.size[1] // num_patches),
                "•", fontsize=12, color="red", ha="center", va="center")

plt.title("DINOv2 Patchwise Features (t-SNE Projection)")
plt.show()


import seaborn as sns

# Compute cosine similarity between patches
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(patch_tokens)

# Plot similarity heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(similarity_matrix, cmap="coolwarm", square=True)
plt.title("Patchwise Feature Similarity (Cosine Distance)")
plt.show()
