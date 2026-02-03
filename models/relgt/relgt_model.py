"""
RelGT Model Wrapper - Enhanced with Feed-Forward Networks (FFN)
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
        
        # [Safety Check] Ensure hidden_dim is divisible by num_heads
        # This is required for the TransformerConv to work correctly with concatenation
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"

        # Initialize with a large buffer for types to handle potential ID shifts
        self.real_num_node_types = max(num_node_types + 1000, 5000)
        self.real_num_edge_types = 500
        
        # Node Type Embedding
        self.node_type_embedding = nn.Embedding(self.real_num_node_types, self.hidden_dim)
        
        # Edge Type Embedding
        self.edge_type_embedding = nn.Embedding(self.real_num_edge_types, self.hidden_dim)
        
        # Transformer Components
        self.layers = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        
        for _ in range(self.num_layers):
            # 1. Attention Layer (Graph Transformer Convolution)
            # We explicitly set edge_dim to allow edge features injection
            conv = TransformerConv(
                self.hidden_dim,
                self.hidden_dim // self.num_heads,
                heads=self.num_heads,
                dropout=config.dropout_rate,
                edge_dim=self.hidden_dim 
            )
            self.layers.append(conv)
            self.norms1.append(nn.LayerNorm(self.hidden_dim))
            
            # 2. Feed-Forward Network (FFN)
            # Standard Transformer FFN: Linear -> Activation -> Dropout -> Linear
            # Expansion factor is typically 4
            ffn = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            )
            self.ffns.append(ffn)
            self.norms2.append(nn.LayerNorm(self.hidden_dim))

    def forward(self, x, edge_index, node_types, edge_types):
        """
        Args:
            x: Node features [N, D]
            edge_index: Graph connectivity [2, E]
            node_types: Node type IDs [N]
            edge_types: Edge type IDs [E]
        """
        device = x.device
        
        # --- DEFENSIVE CLAMPING ---
        # Ensure IDs are within embedding range to prevent index out of bounds
        if node_types.numel() > 0:
            node_types = node_types.clamp(max=self.real_num_node_types - 1)
            
        if edge_types.numel() > 0:
            edge_types = edge_types.clamp(max=self.real_num_edge_types - 1)
        # --------------------------

        # Edge Embedding
        if edge_index.numel() > 0:
            edge_emb = self.edge_type_embedding(edge_types)
        else:
            edge_emb = torch.zeros(0, self.hidden_dim, device=device)

        # Transformer Layers Loop (Attention -> Add&Norm -> FFN -> Add&Norm)
        for i, layer in enumerate(self.layers):
            # Save input for residual connection
            x_in = x

            # --- Sub-layer 1: Multi-Head Attention ---
            if edge_index.numel() > 0:
                attn_out = layer(x, edge_index, edge_attr=edge_emb)
            else:
                attn_out = torch.zeros_like(x)
            
            # Residual Connection + LayerNorm 1
            x = self.norms1[i](x_in + attn_out)
            
            # --- Sub-layer 2: Feed Forward Network ---
            # Save input for second residual connection
            x_in2 = x
            ffn_out = self.ffns[i](x)
            
            # Residual Connection + LayerNorm 2
            x = self.norms2[i](x_in2 + ffn_out)
            
        return x