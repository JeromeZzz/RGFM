"""
上下文学习(ICL)模块
实现KumoRFM的上下文学习机制
(Fix: Robust device handling for DataParallel)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import math

from config.model_config import KumoRFMConfig


class ICLModule(nn.Module):
    """
    上下文学习模块
    处理上下文示例和测试实例的模式匹配
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self._label_vocab = {}
        self._class_vocab = {}
        self.num_heads = config.icl_num_heads
        self.num_layers = config.icl_num_layers

        # 上下文编码器
        self.context_encoder = ContextEncoder(config)

        # 标签编码器
        self.label_encoder = LabelEncoder(config)

        # ICL Transformer层
        self.icl_layers = nn.ModuleList([
            ICLTransformerLayer(config)
            for _ in range(self.num_layers)
        ])

        # 位置编码
        self.position_encoding = nn.Parameter(
            torch.randn(1, 1000, self.hidden_dim)  # 支持最多1000个位置
        )

        # 任务特定的预测头
        self.task_heads = nn.ModuleDict()

        # 查询标记
        self.query_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))

    def register_task_head(self, task_name: str, task_head: nn.Module) -> None:
        """注册任务特定的预测头"""
        self.task_heads[task_name] = task_head

    def forward(self,
                context_graphs: List[torch.Tensor],
                context_labels: List[Any],
                test_graph: torch.Tensor,
                task_type: str,
                metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        前向传播
        """
        # [FIX] Robustly get device from input tensor
        device = test_graph.device

        batch_size = test_graph.size(0) if test_graph.dim() > 1 else 1
        num_context = len(context_graphs)

        # 编码上下文
        context_embeddings = []
        for i, (graph_emb, label) in enumerate(zip(context_graphs, context_labels)):
            # 编码图表示
            ctx_emb = self.context_encoder(graph_emb)

            # 编码标签 [FIX] Pass device explicitly
            label_emb = self.label_encoder(label, task_type, metadata, device=device)

            # 合并图和标签表示
            combined = ctx_emb + label_emb
            context_embeddings.append(combined)

        # 堆叠上下文
        if context_embeddings:
            context_embeddings = torch.stack(context_embeddings, dim=1)
        else:
            # 没有上下文时使用零向量
            context_embeddings = torch.zeros(batch_size, 1, self.hidden_dim, device=device)

        # 编码测试实例
        test_embedding = self.context_encoder(test_graph)
        test_embedding = test_embedding.unsqueeze(1)

        # 添加查询标记
        query_token = self.query_token.expand(batch_size, -1, -1)

        # 组合序列：[context_1, ..., context_n, test, query]
        sequence = torch.cat([context_embeddings, test_embedding, query_token], dim=1)

        # 添加位置编码
        seq_len = sequence.size(1)
        if seq_len > self.position_encoding.size(1):
            sequence = sequence[:, -self.position_encoding.size(1):, :]
            seq_len = sequence.size(1)

        position_encoding = self.position_encoding[:, :seq_len, :]
        sequence = sequence + position_encoding

        # 通过ICL Transformer层
        for layer in self.icl_layers:
            sequence = layer(sequence)

        # 提取查询表示
        query_output = sequence[:, -1, :]

        # 应用任务头
        if task_type in self.task_heads:
            predictions = self.task_heads[task_type](query_output)
        else:
            predictions = query_output

        return predictions


class ContextEncoder(nn.Module):
    """上下文编码器"""

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.graph_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        if graph_embedding.dim() == 1:
            graph_embedding = graph_embedding.unsqueeze(0)
        encoded = self.graph_projection(graph_embedding)
        encoded = self.norm(encoded)
        return encoded


class LabelEncoder(nn.Module):
    """标签编码器"""

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.class_embedding = nn.Embedding(1000, self.hidden_dim)
        self.regression_encoder = nn.Linear(1, self.hidden_dim)
        self.multilabel_encoder = nn.Linear(100, self.hidden_dim)

    def forward(self,
                labels: Any,
                task_type: str,
                metadata: Optional[Dict[str, Any]] = None,
                device: Optional[torch.device] = None) -> torch.Tensor:
        """
        编码标签
        [FIX] Added 'device' argument to avoid StopIteration
        """
        if not hasattr(self, "_label_vocab"):
            self._label_vocab = {}

        # Fallback
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device('cpu')

        if isinstance(labels, torch.Tensor):
            batch_size = labels.size(0) if labels.dim() > 0 else 1
            labels = labels.to(device)
        else:
            numeric_list = self._ensure_numeric_list(labels, as_int=False)
            labels = torch.tensor(numeric_list, device=device)
            batch_size = labels.size(0) if labels.dim() > 0 else 1

        if task_type == "classification":
            numeric = self._ensure_numeric_list(labels, as_int=True)
            labels = torch.tensor(numeric, device=device, dtype=torch.long).unsqueeze(0)
            if (labels < 0).any(): labels = labels.clamp_min(0)

            # grow embedding if needed
            if labels.numel() > 0:
                max_label = int(labels.max().item())
                num_emb = int(self.class_embedding.num_embeddings)
                if max_label >= num_emb:
                    new_size = max(max_label + 1, num_emb * 2)
                    new_emb = nn.Embedding(new_size, self.hidden_dim).to(device)
                    with torch.no_grad():
                        new_emb.weight[:num_emb].copy_(self.class_embedding.weight.data)
                    self.class_embedding = new_emb

            encoded = self.class_embedding(labels)

        elif task_type == "regression":
            if labels.dim() == 0:
                labels = labels.unsqueeze(0).unsqueeze(-1)
            elif labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            encoded = self.regression_encoder(labels.float())

        elif task_type == "multilabel":
            if labels.dim() == 1: labels = labels.unsqueeze(0)
            if labels.size(-1) < 100:
                padding = torch.zeros(batch_size, 100 - labels.size(-1), device=device)
                labels = torch.cat([labels, padding], dim=-1)
            else:
                labels = labels[:, :100]
            encoded = self.multilabel_encoder(labels.float())

        elif task_type == "link_prediction":
            numeric = self._ensure_numeric_list(labels, as_int=True)
            labels = torch.tensor(numeric, device=device, dtype=torch.long).unsqueeze(0)
            if (labels < 0).any(): labels = labels.clamp_min(0)
            encoded = self.class_embedding(labels)

        else:
            encoded = torch.zeros(batch_size, self.hidden_dim, device=device)

        if encoded.dim() > 2:
            encoded = encoded.squeeze(1)

        return encoded

    def _ensure_numeric_list(self, labels: Any, as_int: bool = False) -> List[float]:
        result: List[float] = []
        if isinstance(labels, torch.Tensor):
            flat = labels.detach().cpu().view(-1).tolist()
            for val in flat: result.append(int(val) if as_int else float(val))
            return result
        if isinstance(labels, (list, tuple)):
            for item in labels: result.extend(self._ensure_numeric_list(item, as_int=as_int))
            return result
        if isinstance(labels, (int, float, bool)):
            val = int(labels) if as_int else float(labels)
            result.append(val)
            return result
        key = str(labels)
        if key not in self._label_vocab: self._label_vocab[key] = len(self._label_vocab)
        mapped = self._label_vocab[key]
        result.append(int(mapped) if as_int else float(mapped))
        return result


class ICLTransformerLayer(nn.Module):
    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.icl_num_heads
        self.dropout_rate = config.dropout_rate
        self.self_attention = nn.MultiheadAttention(self.hidden_dim, self.num_heads, dropout=self.dropout_rate,
                                                    batch_first=True)
        self.feed_forward = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim * 4), nn.ReLU(),
                                          nn.Dropout(self.dropout_rate),
                                          nn.Linear(self.hidden_dim * 4, self.hidden_dim))
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attention(x, x, x)
        x = residual + self.dropout(attn_output)
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + self.dropout(ff_output)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, config: KumoRFMConfig, num_classes: int):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(),
                                        nn.Dropout(config.dropout_rate), nn.Linear(config.hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.projection(x)


class RegressionHead(nn.Module):
    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(),
                                        nn.Dropout(config.dropout_rate), nn.Linear(config.hidden_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.projection(x).squeeze(-1)


class LinkPredictionHead(nn.Module):
    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(),
                                        nn.Dropout(config.dropout_rate),
                                        nn.Linear(config.hidden_dim, config.hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.projection(x)