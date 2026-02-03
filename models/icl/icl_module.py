
"""
上下文学习(ICL)模块 - Mask Support

"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from config.model_config import KumoRFMConfig

class ICLModule(nn.Module):
    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self._label_vocab = {}
        self.context_encoder = ContextEncoder(config)
        self.label_encoder = LabelEncoder(config)
        self.icl_layers = nn.ModuleList([ICLTransformerLayer(config) for _ in range(config.icl_num_layers)])
        self.position_encoding = nn.Parameter(torch.randn(1, 1000, self.hidden_dim))
        self.task_heads = nn.ModuleDict()
        self.query_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

    def register_task_head(self, task_name: str, task_head: nn.Module) -> None:
        self.task_heads[task_name] = task_head

    def forward(self, context_graphs: Any, context_labels: Any,
                test_graph: torch.Tensor, task_type: str,
                metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        # NOTE: This forward is for legacy call compatibility. 
        # The main vectorized call is in KumoRFM._run_icl_vectorized
        return torch.zeros(1)

class ContextEncoder(nn.Module):
    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.graph_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        if graph_embedding.dim() == 1: graph_embedding = graph_embedding.unsqueeze(0)
        return self.norm(self.graph_projection(graph_embedding))

class LabelEncoder(nn.Module):
    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.class_embedding = nn.Embedding(5000, self.hidden_dim)
        self.regression_encoder = nn.Linear(1, self.hidden_dim)

    def forward(self, labels: Any, task_type: str, metadata: Optional[Dict] = None, device: torch.device = None) -> torch.Tensor:
        if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if isinstance(labels, torch.Tensor) and labels.dim() == 2:
            labels = labels.to(device)
            if task_type == "classification":
                return self.class_embedding(labels.long()) 
            elif task_type == "regression":
                return self.regression_encoder(labels.float().unsqueeze(-1)) 
        return torch.zeros(1, 1, self.hidden_dim, device=device)

class ICLTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.hidden_dim, config.icl_num_heads, dropout=config.dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim*4), nn.ReLU(), nn.Dropout(config.dropout_rate), nn.Linear(config.hidden_dim*4, config.hidden_dim))
        self.norm1, self.norm2 = nn.LayerNorm(config.hidden_dim), nn.LayerNorm(config.hidden_dim)
    
    def forward(self, x, src_key_padding_mask=None):
        # MultiheadAttention uses key_padding_mask=True for ignored positions
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), key_padding_mask=src_key_padding_mask)
        x = x + attn_out
        return x + self.ffn(self.norm2(x))

class ClassificationHead(nn.Module):
    def __init__(self, config: KumoRFMConfig, num_classes: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.projection(x)

class RegressionHead(nn.Module):
    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.projection(x).squeeze(-1)

class LinkPredictionHead(nn.Module):
    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 1) # Output logit for BCE
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.projection(x)
