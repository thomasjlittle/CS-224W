# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.datasets import MoleculeNet
from torch_geometric.nn import GCNConv, global_mean_pool

# Load dataset (Tox21)
dataset = MoleculeNet(root='data/Tox21', name='Tox21')
train_dataset = dataset[:6000]
val_dataset = dataset[6000:7000]
test_dataset = dataset[7000:]

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define GNN model class
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        # Graph convolutional layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Readout layer (global mean pooling)
        self.readout = global_mean_pool
        # Linear classifier layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        # Unpack data attributes (node features and edge indices)
        x, edge_index = data.x.float(), data.edge_index.long()
        # Apply first graph convolutional layer with ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        # Apply second graph convolutional layer with ReLU activation
        x = F.relu(self.conv2(x, edge_index))
        # Apply readout layer to get graph embedding from node embeddings
        x = self.readout(x, data.batch)
        # Apply linear classifier layer with sigmoid activation 
        x = torch.sigmoid(self.fc(x))
        return x

# Create GNN model instance with input dimension 9 (number of node features), hidden dimension 64 and output dimension 12 (number of classes) 
model = GNN(9, 64 ,12)

# Define loss function (binary cross entropy) and optimizer (Adam)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Define number of epochs for training loop 
num_epochs = 10

# Training loop 
for epoch in range(num_epochs):
    # Set model to training mode 
    model.train()
    # Initialize running loss and accuracy 
    train_loss = 0.0 
    train_acc = 0.0 
    # Loop over batches of training data 
    for i,data in enumerate(train_loader):
      # Zero the parameter gradients 
      optimizer.zero_grad()
      # Forward pass: compute predictions from input data 
      outputs= model(data)  
      # Get true labels from data attributes  
      labels= data.y.float()  

      # Create a boolean mask that indicates which samples have nan labels
      mask = ~torch.any(data.y.isnan(), dim=1)
      # Slice the outputs and labels tensors using the mask
      outputs = outputs[mask]
      labels = labels[mask]


      # Compute loss using predictions and true labels  
      loss= criterion(outputs ,labels)  
      # Backward pass: compute gradients from loss  
      loss.backward()  
      # Update parameters using optimizer step  
      optimizer.step()  
      # Add loss to running loss   
      train_loss += loss.item()

# Convert the outputs to binary labels
predictions = torch.round(torch.sigmoid(outputs))
# Create a confusion vector by dividing predictions by labels
confusion_vector = predictions / labels
# Count false positives, false negatives and true positives
false_positives = torch.sum(confusion_vector == float('inf')).item()
false_negatives = torch.sum(confusion_vector == 0).item()
true_positives = torch.sum(confusion_vector == 1).item()

# Calculate accuracy using PyTorch's accuracy function
accuracy = torch.sum(predictions == labels).item() / len(labels)
# Print accuracy
print(f"Accuracy: {accuracy}")

print("egg")