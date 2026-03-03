import torch
from .models.gnn_model import FragilityGNN, load_model
from torch_geometric.data import Data
import os

class GNNInference:
    """
    Real-time Fragility Inference for the Backend.
    """
    def __init__(self, model_path=None):
        self.model = FragilityGNN(in_channels=7, hidden_channels=32)
        if model_path and os.path.exists(model_path):
            load_model(self.model, model_path)
        self.model.eval()

    def predict_fragility(self, x, edge_index):
        """
        Runs a forward pass to get [0, 1] fragility scores.
        """
        with torch.no_grad():
            scores = self.model(x, edge_index)
            return scores.numpy().flatten()

    def create_dynamic_data(self, node_features, edge_indices):
        """
        Helper to convert incoming API deltas to PyG Data formats.
        """
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        return x, edge_index
