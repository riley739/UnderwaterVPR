import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Load DINOv2 model
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large')

# Load and preprocess image
image_path = 'img.png'
image = Image.open(image_path)
inputs = processor(image, return_tensors="pt")

# Extract features
with torch.no_grad():
    outputs = model(**inputs)

features = outputs.last_hidden_state  # (batch_size, num_patches + 1, feature_dim)
print(features.shape)  # Example: torch.Size([1, 257, 1024])


# Remove CLS token (position 0) and reshape to patch grid
patch_tokens = features[:, 1:, :].squeeze(0)  # (num_patches, feature_dim)

# Assuming 14x14 grid for DINOv2 large (224x224 input)
num_patches = int(patch_tokens.shape[0] ** 0.5)
patch_grid = patch_tokens.reshape(num_patches, num_patches, -1)

# Select a feature channel to visualize (e.g., 100th feature)
feature_map = patch_grid[:, :, 100].cpu().numpy()

# Plot the feature map
plt.imshow(feature_map, cmap='viridis')
plt.colorbar()
plt.title("Feature Map (Channel 100)")
plt.savefig('feature_map.png')


# # Extract attention weights (from the model's internal layers)
# attn_weights = outputs.attentions  # Tuple of attention maps from all layers

# # Visualize the attention of the last layer
# last_layer_attention = attn_weights[-1]  # Shape: (batch, heads, tokens, tokens)
# attn_map = last_layer_attention[0, 0, 1:, 1:].reshape(num_patches, num_patches).cpu().numpy()

# plt.imshow(attn_map, cmap='inferno')
# plt.title('Attention Map (Last Layer)')
# plt.colorbar()
# plt.savefig('Attention_map.png')


from sklearn.decomposition import PCA

# Flatten patches and apply PCA to reduce dimensionality
patch_features = patch_tokens.cpu().numpy()
pca = PCA(n_components=2)
pca_features = pca.fit_transform(patch_features.reshape(-1, patch_features.shape[-1]))

# Visualize PCA-transformed features
plt.scatter(pca_features[:, 0], pca_features[:, 1], c=np.arange(len(pca_features)), cmap='plasma')
plt.title("PCA of DINOv2 Features")
plt.colorbar()
plt.savefig('pca.png')