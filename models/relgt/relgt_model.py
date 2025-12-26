"""
RelGT Model Wrapper

"""
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class RelGTWrapper(nn.Module):
    def __init__(self, config, num_node_types):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        
        # [Strategy] Initialize with a very large buffer by default.
        # Even if 'num_node_types' passed in is small, we force a large buffer (e.g., 5000)
        # This reduces the chance of collision (clamping valid IDs to 'unknown').
        # Using max() ensures we respect the input if it's already huge.
        self.real_num_node_types = max(num_node_types + 1000, 5000)
        self.real_num_edge_types = 500
        
        # Node Type Embedding
        self.node_type_embedding = nn.Embedding(self.real_num_node_types, self.hidden_dim)
        
        # Edge Type Embedding
        self.edge_type_embedding = nn.Embedding(self.real_num_edge_types, self.hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            conv = TransformerConv(
                self.hidden_dim,
                self.hidden_dim // self.num_heads,
                heads=self.num_heads,
                dropout=config.dropout_rate,
                edge_dim=self.hidden_dim 
            )
            self.layers.append(conv)
            
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x, edge_index, node_types, edge_types):
        """
        x: [N, D]
        edge_index: [2, E]
        node_types: [N]
        edge_types: [E]
        """
        device = x.device
        
        # --- DEFENSIVE CLAMPING (The "Best Practice" Part) ---
        # 1. Clamp Node Types
        if node_types.numel() > 0:
            # Check max efficiently (only triggers sync if strictly needed, but clamp is safe anyway)
            # We simply unconditionally clamp to be safe. It's very fast.
            # Using (self.real_num_node_types - 1) as the 'unknown/overflow' bucket.
            node_types = node_types.clamp(max=self.real_num_node_types - 1)
            
        # 2. Clamp Edge Types
        if edge_types.numel() > 0:
            edge_types = edge_types.clamp(max=self.real_num_edge_types - 1)
        # -----------------------------------------------------

        # Edge Embedding
        if edge_index.numel() > 0:
            edge_emb = self.edge_type_embedding(edge_types)
        else:
            edge_emb = torch.zeros(0, self.hidden_dim, device=device)

        # GNN Layers
        for layer in self.layers:
            if edge_index.numel() > 0:
                out = layer(x, edge_index, edge_attr=edge_emb)
            else:
                # Handle disconnected graph case
                out = torch.zeros_like(x)
            
            x = self.norm(x + out)
            
        return x