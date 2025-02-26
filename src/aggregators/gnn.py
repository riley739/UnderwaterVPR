import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

class FeatureGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeatureGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

class PlaceRecognitionGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PlaceRecognitionGCN, self).__init__()
        self.gcn = FeatureGCN(input_dim, hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim * 2, 1)  # Binary classifier for place similarity

    def forward(self, x, edge_index, batch, pairs):
        feature_embeddings = self.gcn(x, edge_index)
        image_embeddings = global_mean_pool(feature_embeddings, batch)  # Aggregate features per image

        # Extract embeddings for the image pairs
        pair_embeddings = torch.cat([image_embeddings[pairs[:, 0]], image_embeddings[pairs[:, 1]]], dim=1)
        return torch.sigmoid(self.fc(pair_embeddings))
    

class Graph():
    def __init__(self):
        pass 



    #EDGE PREDICTION TASK USING CLS Tokens? 

# Simulated Data Example
num_images = 50
features_per_image = 128  # Assume each image has 128 features (keypoints)
feature_dim = 256
hidden_dim = 128
output_dim = 64

num_nodes = num_images * features_per_image

# Random feature descriptors (simulating CNN outputs)
x = torch.randn((num_nodes, feature_dim))

# Random edges: should be formed based on feature similarity (e.g., KNN matching)
edge_index = torch.randint(0, num_nodes, (2, num_nodes * 3))

# Assign each node to an image
batch = torch.arange(num_images).repeat_interleave(features_per_image)

# Pairs of images to classify as same/different place
pairs = torch.randint(0, num_images, (20, 2))  # 20 pairs of images

# Initialize and test the model
model = PlaceRecognitionGCN(feature_dim, hidden_dim, output_dim)
output = model(x, edge_index, batch, pairs)

print(output.shape)  # Should be (20, 1), probabilities for each pair

print(output)