import joblib 
import pickle
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import torch

def load_everything():
    dataset = torch.load("logs/output.pt")    
    return dataset

dataset = load_everything()

print(len(dataset))
print(dataset.num_features)
print(dataset)
print(50*'=')

print(dataset)
print(dataset.num_nodes)
print(dataset.num_edges)
print(max(dataset.y))
print(dataset.train_mask.sum())
print(dataset.is_undirected())
