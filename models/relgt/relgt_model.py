import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from utils.debug_probe import DebugProbe # [NEW]

# --- HELPER: LayerScale ---
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-2):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    def forward(self, x): return x * self.gamma

# --- IMPLEMENTATION 1: Sparse PyG ---
class PyGGraphTransformer(nn.Module):
    def __init__(self, config, edge_embedding):
        super().__init__()
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.dropout = nn.Dropout(config.dropout_rate)
        self.edge_embedding = edge_embedding 

        self.layers = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.gamma1 = nn.ModuleList()
        self.gamma2 = nn.ModuleList()

        for _ in range(self.num_layers):
            conv = TransformerConv(
                self.hidden_dim, self.hidden_dim // self.num_heads,
                heads=self.num_heads, dropout=config.dropout_rate,
                edge_dim=self.hidden_dim, root_weight=True
            )
            self.layers.append(conv)
            self.norms1.append(nn.LayerNorm(self.hidden_dim))
            self.gamma1.append(LayerScale(self.hidden_dim))
            
            ffn = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 4), nn.GELU(), nn.Dropout(config.dropout_rate),
                nn.Linear(self.hidden_dim * 4, self.hidden_dim), nn.Dropout(config.dropout_rate)
            )
            self.ffns.append(ffn)
            self.norms2.append(nn.LayerNorm(self.hidden_dim))
            self.gamma2.append(LayerScale(self.hidden_dim))

    def forward(self, x, edge_index, edge_types, batch=None):
        if edge_index.numel() > 0: edge_emb = self.edge_embedding(edge_types)
        else: edge_emb = torch.zeros(0, self.hidden_dim, device=x.device)

        for i in range(self.num_layers):
            x_norm = self.norms1[i](x)
            if edge_index.numel() > 0: attn_out = self.layers[i](x_norm, edge_index, edge_attr=edge_emb)
            else: attn_out = x_norm
            x = x + self.gamma1[i](self.dropout(attn_out))
            
            # [PROBE] Post-Attn
            DebugProbe.check_tensor(f"RelGT_L{i}_PostAttn", x)

            x_norm2 = self.norms2[i](x)
            ffn_out = self.ffns[i](x_norm2)
            x = x + self.gamma2[i](ffn_out)
            
            # [PROBE] Post-FFN
            DebugProbe.check_tensor(f"RelGT_L{i}_PostFFN", x)
        return x

# --- IMPLEMENTATION 2: Custom Dense ---
class DenseGraphTransformer(nn.Module):
    def __init__(self, config, edge_embedding):
        super().__init__()
        self.num_layers = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.dropout = nn.Dropout(config.dropout_rate)
        self.edge_embedding = edge_embedding

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.ModuleDict({
                'attn': nn.MultiheadAttention(self.hidden_dim, self.num_heads, dropout=config.dropout_rate, batch_first=True),
                'norm1': nn.LayerNorm(self.hidden_dim),
                'gamma1': LayerScale(self.hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim * 4), nn.GELU(), nn.Dropout(config.dropout_rate),
                    nn.Linear(self.hidden_dim * 4, self.hidden_dim), nn.Dropout(config.dropout_rate)
                ),
                'norm2': nn.LayerNorm(self.hidden_dim),
                'gamma2': LayerScale(self.hidden_dim)
            }))

    def forward(self, x, edge_index, edge_types, batch):
        x_dense, mask = to_dense_batch(x, batch)
        B, N_max, D = x_dense.shape
        key_padding_mask = ~mask 
        
        # [PROBE] Check Dense Input
        DebugProbe.check_tensor("Dense_In", x_dense)

        for i, layer in enumerate(self.layers):
            x_norm = layer['norm1'](x_dense)
            attn_out, _ = layer['attn'](x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
            x_dense = x_dense + layer['gamma1'](self.dropout(attn_out))
            
            # [PROBE] Post-Attn
            DebugProbe.check_tensor(f"Dense_RelGT_L{i}_PostAttn", x_dense)

            x_norm2 = layer['norm2'](x_dense)
            ffn_out = layer['ffn'](x_norm2)
            x_dense = x_dense + layer['gamma2'](ffn_out)
            
            # [PROBE] Post-FFN
            DebugProbe.check_tensor(f"Dense_RelGT_L{i}_PostFFN", x_dense)

        out = x_dense[mask] 
        return out

class RelGTWrapper(nn.Module):
    def __init__(self, config, num_node_types=None):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        assert self.hidden_dim % self.config.num_heads == 0, "Hidden dim must be divisible by heads"

        self.real_num_edge_types = 500
        self.edge_type_embedding = nn.Embedding(self.real_num_edge_types, self.hidden_dim)
        
        self.impl_type = getattr(config, 'relgt_impl', 'custom') 
        if self.impl_type == 'pyg': self.model = PyGGraphTransformer(config, self.edge_type_embedding)
        else: self.model = DenseGraphTransformer(config, self.edge_type_embedding)

    def forward(self, x, edge_index, node_types, edge_types, batch_index=None):
        if edge_index.numel() > 0: edge_types = edge_types.clamp(max=self.real_num_edge_types - 1)
        
        if self.impl_type == 'pyg': return self.model(x, edge_index, edge_types)
        else:
            if batch_index is None: batch_index = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            return self.model(x, edge_index, edge_types, batch_index)