"""
上下文学习(ICL)模块

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
        # [Fix] Robust device handling
        device = test_graph.device
        
        # Determine Batch Size
        if isinstance(context_graphs, list):
             # List of tensors (Old Loop Mode or Single Sample)
             batch_size = 1 # Simplified, typically handled externally
        else:
             # Tensor input [Batch, Num_Ctx, Dim] (New Vectorized Mode)
             batch_size = context_graphs.size(0)

        # 1. Encode Context
        if isinstance(context_graphs, torch.Tensor) and context_graphs.dim() == 3:
            # Fast Path: Already Encoded [Batch, N, Dim]
            ctx_emb = context_graphs
        else:
            # Slow Path: Encode list of graphs
            context_embeddings_list = []
            # Note: This path is mainly for single-sample inference compatibility
            for graph in context_graphs:
                context_embeddings_list.append(self.context_encoder(graph))
            ctx_emb = torch.stack(context_embeddings_list, dim=1) if context_embeddings_list else torch.tensor([], device=device)

        # 2. Encode Labels
        label_emb = self.label_encoder(context_labels, task_type, metadata, device=device)
        
        # 3. Combine
        # Ensure dimensions match for broadcast
        if ctx_emb.dim() == 3 and label_emb.dim() == 3:
            context_embeddings = ctx_emb + label_emb
        else:
            # Fallback for old list-based behavior if dimensions mismatch randomly
            # (Should not happen with new LabelEncoder logic)
            context_embeddings = ctx_emb + label_emb

        # 4. Prepare Sequence
        if context_embeddings.shape[0] == 0:
             # Empty context case
             context_embeddings = torch.zeros(batch_size, 0, self.hidden_dim, device=device)

        # Encode Test [Batch, 1, Dim]
        if test_graph.dim() == 2:
            test_embedding = self.context_encoder(test_graph).unsqueeze(1)
        elif test_graph.dim() == 1: # Already encoded vector
            test_embedding = test_graph.view(batch_size, 1, -1)
        else: # [Batch, 1, Dim] already
            test_embedding = test_graph

        # Query Token
        query_token = self.query_token.expand(batch_size, -1, -1)
        
        # Concat
        sequence = torch.cat([context_embeddings, test_embedding, query_token], dim=1)

        # Positional Encoding
        seq_len = sequence.size(1)
        if seq_len > self.position_encoding.size(1):
             sequence = sequence[:, -self.position_encoding.size(1):, :]
             seq_len = sequence.size(1)
        sequence = sequence + self.position_encoding[:, :seq_len, :]

        # Transformer
        for layer in self.icl_layers:
            sequence = layer(sequence)

        query_output = sequence[:, -1, :]
        
        if task_type in self.task_heads:
            return self.task_heads[task_type](query_output)
        return query_output

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
        self.class_embedding = nn.Embedding(1000, self.hidden_dim)
        self.regression_encoder = nn.Linear(1, self.hidden_dim)
        self.multilabel_encoder = nn.Linear(100, self.hidden_dim)

    def forward(self, labels: Any, task_type: str, metadata: Optional[Dict] = None, device: torch.device = None) -> torch.Tensor:
        if device is None: 
            try: device = next(self.parameters()).device
            except: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not hasattr(self, "_label_vocab"): self._label_vocab = {}

        # --- [CRITICAL FIX START] Handle Batched Tensor Input Correctly ---
        if isinstance(labels, torch.Tensor) and labels.dim() == 2:
            # Input is [Batch, Seq_Len]
            labels = labels.to(device)
            if task_type == "classification":
                # Ensure long for embedding lookup
                return self.class_embedding(labels.long()) # Returns [Batch, Seq, Hidden]
            elif task_type == "regression":
                # Ensure float and add last dim for linear
                return self.regression_encoder(labels.float().unsqueeze(-1)) # [Batch, Seq, 1] -> [Batch, Seq, Hidden]
            elif task_type == "multilabel":
                 # Not fully supported in batch mode yet for simplicity
                 pass
        # --- [CRITICAL FIX END] ---

        # Fallback for List input or Single Sample (1D Tensor)
        batch_size = 1
        if isinstance(labels, torch.Tensor):
            labels = labels.to(device)
            if labels.dim() > 0: batch_size = labels.size(0)
        
        if task_type == "classification":
            numeric = self._ensure_numeric_list(labels, as_int=True)
            labels = torch.tensor(numeric, device=device, dtype=torch.long).unsqueeze(0)
            if (labels < 0).any(): labels = labels.clamp_min(0)
            
            # Dynamic Growth logic (simplified)
            if labels.numel() > 0 and labels.max() >= self.class_embedding.num_embeddings:
                 # In training, we usually fix size or handle this outside. 
                 # For safety, clamp or resize (resize is hard in DDP). Clamp for now.
                 labels = labels.clamp(max=self.class_embedding.num_embeddings-1)

            return self.class_embedding(labels).squeeze(1) # [1, N, D]

        elif task_type == "regression":
            # For list input, we flatten and embed
            numeric = self._ensure_numeric_list(labels, as_int=False)
            t = torch.tensor(numeric, device=device, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
            return self.regression_encoder(t).squeeze(1) # [1, N, D]

        return torch.zeros(1, 1, self.hidden_dim, device=device)

    def _ensure_numeric_list(self, labels: Any, as_int: bool = False) -> List[float]:
        result = []
        if isinstance(labels, torch.Tensor):
            for val in labels.detach().cpu().view(-1).tolist(): result.append(int(val) if as_int else float(val))
        elif isinstance(labels, (list, tuple)):
            for item in labels: result.extend(self._ensure_numeric_list(item, as_int))
        else:
            try:
                val = float(labels)
                result.append(int(val) if as_int else val)
            except:
                key = str(labels)
                if key not in self._label_vocab: self._label_vocab[key] = len(self._label_vocab)
                result.append(self._label_vocab[key] if as_int else float(self._label_vocab[key]))
        return result

class ICLTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.hidden_dim, config.icl_num_heads, dropout=config.dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim*4), nn.ReLU(), nn.Dropout(config.dropout_rate), nn.Linear(config.hidden_dim*4, config.hidden_dim))
        self.norm1, self.norm2 = nn.LayerNorm(config.hidden_dim), nn.LayerNorm(config.hidden_dim)
    def forward(self, x):
        x = x + self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        return x + self.ffn(self.norm2(x))

class ClassificationHead(nn.Module):
    def __init__(self, config: KumoRFMConfig, num_classes: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, num_classes)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

class RegressionHead(nn.Module):
    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, 1)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x).squeeze(-1)

class LinkPredictionHead(nn.Module):
    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)