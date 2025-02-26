import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torchvision import models, transforms
from torch_geometric.data import Data
import networkx as nx
import numpy as np

# Step 2: Define GNN for Edge Prediction
class EdgePredictorGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.edge_predictor = torch.nn.Linear(2 * hidden_channels, 1)  # Predict edge existence

    def forward(self, x, edge_index):
        _, x, = x 
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x
    
    def predict_edges(self, x, edge_candidates):
        """ Predicts edge probabilities for candidate edges """
        node_i = x[edge_candidates[0]]
        node_j = x[edge_candidates[1]]
        edge_features = torch.cat([node_i, node_j], dim=1)
        return torch.sigmoid(self.edge_predictor(edge_features)).squeeze()
    

