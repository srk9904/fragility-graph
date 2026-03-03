import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FragilityGNN(torch.nn.Module):
    """
    Graph Neural Network for Code Fragility Prediction.
    Uses GraphSAGE (Sample and Aggregate) to generate node embeddings
    and predict a fragility score [0, 1].
    """
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(FragilityGNN, self).__init__()
        
        # Layer 1: Structural Context Aggregation
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        
        # Layer 2: Deep Dependency Interaction
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Layer 3: Fragility Prediction Header
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        # 1. Neighbor aggregation + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # 2. Refine embeddings with multi-hop context
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # 3. Final Fragility Regression (Sigmoid for 0-1 range)
        x = self.conv3(x, edge_index)
        return torch.sigmoid(x)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
