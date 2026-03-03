import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from neo4j import GraphDatabase
from .gnn_model import FragilityGNN
import os

class GNNTrainer:
    """
    Handles data extraction from Neo4j and model optimization.
    """
    def __init__(self, neo4j_uri, neo4j_user, neo4j_pw):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pw))
        self.model = FragilityGNN(in_channels=7, hidden_channels=32)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def close(self):
        self.driver.close()

    def get_training_data(self):
        """
        Extract code graph from Neo4j and convert to PyTorch Geometric Data.
        """
        with self.driver.session() as session:
            # 1. Fetch Nodes and Features
            node_query = """
            MATCH (n) 
            WHERE n:Function OR n:Class 
            RETURN id(n) AS id, 
                   n.complexity AS complexity,
                   n.lines_of_code AS loc,
                   n.change_freq AS freq,
                   size((n)-[:CALLS]->()) AS out_degree,
                   size((n)<-[:CALLS]-()) AS in_degree
            """
            nodes = session.run(node_query).data()
            
            # 2. Fetch Edges
            edge_query = """
            MATCH (a)-[r:CALLS]->(b)
            RETURN id(a) AS src, id(b) AS dst
            """
            edges = session.run(edge_query).data()

            # Mapping Neo4j IDs to 0-indexed tensors
            id_map = {node['id']: i for i, node in enumerate(nodes)}
            
            # Features: [complexity, loc, freq, out_degree, in_degree, is_function, is_class]
            x = torch.tensor([
                [n.get('complexity', 1), n.get('loc', 10), n.get('freq', 1), 
                 n.get('out_degree', 0), n.get('in_degree', 0), 1, 0] 
                for n in nodes
            ], dtype=torch.float)

            edge_index = torch.tensor([
                [id_map[e['src']], id_map[e['dst']]] for e in edges if e['src'] in id_map and e['dst'] in id_map
            ], dtype=torch.long).t().contiguous()

            # Mock Labels (In production, these come from historical bug data)
            y = torch.rand((len(nodes), 1)) 

            return Data(x=x, edge_index=edge_index, y=y)

    def train(self, epochs=100):
        data = self.get_training_data()
        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = F.binary_cross_entropy(out, data.y)
            loss.backward()
            self.optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss {loss.item():.4f}")
        
        return self.model
