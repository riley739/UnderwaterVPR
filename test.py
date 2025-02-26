import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torchvision import models, transforms
from torch_geometric.data import Data
import networkx as nx
import numpy as np

# Step 1: Feature Extractor (ResNet-18)
class ImageFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove last FC layer
    
    def forward(self, x):
        x = self.feature_extractor(x)  # Shape: (batch, 512, 1, 1)
        return x.view(x.size(0), -1)  # Flatten to (batch, 512)

# Step 2: Define GNN for Edge Prediction
class EdgePredictorGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.edge_predictor = torch.nn.Linear(2 * hidden_channels, 1)  # Predict edge existence

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x
    
    def predict_edges(self, x, edge_candidates):
        """ Predicts edge probabilities for candidate edges """
        node_i = x[edge_candidates[0]]
        node_j = x[edge_candidates[1]]
        edge_features = torch.cat([node_i, node_j], dim=1)
        return torch.sigmoid(self.edge_predictor(edge_features)).squeeze()

# Step 3: Generate a Synthetic Graph (Example Data)
def create_synthetic_graph(num_nodes=10, num_edges=20):
    np.random.seed(42)
    features = torch.rand((num_nodes, 512))  # Fake CNN embeddings
    edges = torch.randint(0, num_nodes, (2, num_edges))  # Random edges
    return Data(x=features, edge_index=edges)

# Initialize everything
dataset = create_synthetic_graph()
model = EdgePredictorGNN(in_channels=512, hidden_channels=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCELoss()  # Binary classification loss

def train():
    model.train()
    optimizer.zero_grad()
    node_embeddings = model(dataset.x, dataset.edge_index)
    print(node_embeddings)
    edge_preds = model.predict_edges(node_embeddings, dataset.edge_index)
    print(edge_preds)
    edge_labels = torch.ones(edge_preds.size(0))  # Assume all edges are valid

    loss = loss_fn(edge_preds, edge_labels)
    loss.backward()
    optimizer.step()
    print(f'Training Loss: {loss.item():.4f}')

def test_new_image():
    model.eval()
    new_image_feat = torch.rand(1, 128)  # Simulated new image feature
    node_embeddings = model(dataset.x, dataset.edge_index)
    similarities = torch.mm(new_image_feat, node_embeddings.T)  # Cosine similarity test
    match_index = torch.argmax(similarities).item()
    print(f'New image matches node {match_index}')

# Run Training
for epoch in range(5):
    train()

# Test a new image
test_new_image()
