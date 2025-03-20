import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv #GATConv
import numpy as np 

def load_dataset(dataset_path):
    dataset = torch.load("logs/output.pt")    
    return dataset

class GCN(torch.nn.Module):
    def __init__(self, dataset, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        # Initialize the layers
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.out = Linear(hidden_channels, 10)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second Message Passing Layer
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer 
        # x = F.softmax(self.out(x), dim=1)
        return x


    


data = load_dataset("logs/output.pt")

#Initialize model
model = GCN(data, hidden_channels=16)

# Use GPUasdfasdfasdd
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_mask = np.ones(data.num_nodes, dtype=bool)  # Start with all unmasked (0s)
train_mask[np.random.choice(data.num_nodes, 500, replace=False)] = False # Randomly 

test_mask = ~train_mask  # Inverse of train_mask

model = model.to(device)
data = data.to(device)

# Initialize Optimizer
learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=learning_rate, 
                             weight_decay=decay)
# Define loss function (CrossEntropyLoss for Classification Problems with 
# probability distributions)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad() 
      # Use all data as input, because all nodes have node features
      out = model(data.x, data.edge_index)  
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[train_mask], data.y[train_mask])  
      loss.backward() 
      optimizer.step()
      return loss

def test():
      model.eval()
      out = model(data.x, data.edge_index)
      # Use the class with highest probability.
      pred = out.argmax(dim=1)  
      # Check against ground-truth labels.
      test_correct = pred[test_mask] == data.y[test_mask]  
      # Derive ratio of correct predictions.

      test_acc = int(test_correct.sum()) / int(test_mask.sum())  
      return test_acc, pred

losses = []
for epoch in range(0, 1000):
    loss = train()
    losses.append(loss)
    if epoch % 100 == 0:
      print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')



test_acc, pred = test()
print(test_acc)



# print(np.where(test_mask == True))
# print(pred)
