import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from config.model_config import KumoRFMConfig
from utils.debug_probe import DebugProbe # [NEW]

class ICLModule(nn.Module):
    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        self.context_encoder = ContextEncoder(config)
        self.label_encoder = LabelEncoder(config)
        
        self.icl_layers = nn.ModuleList([
            ICLTransformerLayer(config) for _ in range(config.icl_num_layers)
        ])
        
        self.position_encoding = nn.Parameter(torch.randn(1, 1000, self.hidden_dim))
        self.task_heads = nn.ModuleDict()
        self.query_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

    def register_task_head(self, task_name: str, task_head: nn.Module) -> None:
        self.task_heads[task_name] = task_head

    def forward(self, context_embs: List[torch.Tensor], context_labels: List[Any],
                test_emb: torch.Tensor, task_type: str,
                metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        device = test_emb.device
        
        seq_tensors = []
        for e in context_embs:
            if e.dim() == 1: e = e.unsqueeze(0)
            seq_tensors.append(e)
            
        if test_emb.dim() == 1: test_emb = test_emb.unsqueeze(0)
        seq_tensors.append(test_emb)
        
        if len(seq_tensors) > 0:
            x = torch.cat(seq_tensors, dim=0).unsqueeze(0).to(device)
        else:
            x = torch.zeros(1, 1, self.hidden_dim, device=device)

        # [PROBE] Check ICL Input Sequence Diversity
        DebugProbe.check_tensor("ICL_Seq_Input", x.squeeze(0))

        seq_len = x.size(1)
        if seq_len <= self.position_encoding.size(1): pe = self.position_encoding[:, :seq_len, :].to(device)
        else: pe = self.position_encoding[:, :self.position_encoding.size(1), :].to(device)

        x = x + pe

        for i, layer in enumerate(self.icl_layers):
            x = layer(x)
            # [PROBE] Check ICL Layer Output
            DebugProbe.check_tensor(f"ICL_L{i}_Out", x.squeeze(0))

        test_out = x[:, -1, :] 
        
        # [PROBE] Check Final Test Token (Before Head)
        DebugProbe.check_tensor("ICL_Test_Token", test_out)

        if task_type in self.task_heads: return self.task_heads[task_type](test_out)
        else:
            if self.task_heads: return self.task_heads[list(self.task_heads.keys())[0]](test_out)
            return torch.zeros(1, 1, device=device, requires_grad=True)

class ContextEncoder(nn.Module):
    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.graph_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(), nn.Dropout(config.dropout_rate),
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
        self.class_embedding = nn.Embedding(5002, self.hidden_dim)
        self.regression_encoder = nn.Linear(1, self.hidden_dim)

    def forward(self, labels: Any, task_type: str, metadata: Optional[Dict] = None, device: torch.device = None) -> torch.Tensor:
        if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)
            if labels.dim() == 1: labels = labels.unsqueeze(-1)
            
            if task_type == "classification":
                l_idx = labels.long().clamp(0, self.class_embedding.num_embeddings - 1)
                return self.class_embedding(l_idx)
            elif task_type == "regression" or task_type == "link_prediction":
                return self.regression_encoder(labels.float())
        return torch.zeros(1, 1, self.hidden_dim, device=device)

class ICLTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.hidden_dim, config.icl_num_heads, dropout=config.dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim*4), 
            nn.ReLU(), nn.Dropout(config.dropout_rate), 
            nn.Linear(config.hidden_dim*4, config.hidden_dim)
        )
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, x, src_key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

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
            nn.Linear(config.hidden_dim, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.projection(x)