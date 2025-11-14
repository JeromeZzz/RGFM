"""
上下文学习(ICL)模块
实现KumoRFM的上下文学习机制
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

        Args:
            context_graphs: 上下文子图的表示列表
            context_labels: 上下文标签列表
            test_graph: 测试子图的表示
            task_type: 任务类型
            metadata: 额外的元数据

        Returns:
            predictions: 预测结果
        """
        batch_size = test_graph.size(0) if test_graph.dim() > 1 else 1
        num_context = len(context_graphs)

        # 编码上下文
        context_embeddings = []
        for i, (graph_emb, label) in enumerate(zip(context_graphs, context_labels)):
            # 编码图表示
            ctx_emb = self.context_encoder(graph_emb)

            # 编码标签
            label_emb = self.label_encoder(label, task_type, metadata)

            # 合并图和标签表示
            combined = ctx_emb + label_emb
            context_embeddings.append(combined)

        # 堆叠上下文
        if context_embeddings:
            context_embeddings = torch.stack(context_embeddings, dim=1)  # (batch_size, num_context, hidden_dim)
        else:
            # 没有上下文时使用零向量
            context_embeddings = torch.zeros(batch_size, 1, self.hidden_dim, device=test_graph.device)

        # 编码测试实例
        test_embedding = self.context_encoder(test_graph)  # (batch_size, hidden_dim)
        test_embedding = test_embedding.unsqueeze(1)  # (batch_size, 1, hidden_dim)

        # 添加查询标记
        query_token = self.query_token.expand(batch_size, -1, -1)

        # 组合序列：[context_1, ..., context_n, test, query]
        sequence = torch.cat([context_embeddings, test_embedding, query_token], dim=1)

        # 添加位置编码
        seq_len = sequence.size(1)
        position_encoding = self.position_encoding[:, :seq_len, :]
        sequence = sequence + position_encoding

        # 通过ICL Transformer层
        for layer in self.icl_layers:
            sequence = layer(sequence)

        # 提取查询表示 / Extract query representation
        query_output = sequence[:, -1, :]  # (batch_size, hidden_dim)
        # Ensure on same device as module parameters
        query_output = query_output.to(next(self.parameters()).device)

        # 应用任务头
        if task_type in self.task_heads:
            predictions = self.task_heads[task_type](query_output)
        else:
            # 默认线性投影
            predictions = query_output

        return predictions


class ContextEncoder(nn.Module):
    """上下文编码器"""

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config

        # 图表示投影
        self.graph_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

        # 归一化
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, graph_embedding: torch.Tensor) -> torch.Tensor:
        """
        编码图表示
        graph_embedding: (batch_size, hidden_dim) 或 (hidden_dim,)
        """
        if graph_embedding.dim() == 1:
            graph_embedding = graph_embedding.unsqueeze(0)

        # 投影和归一化
        encoded = self.graph_projection(graph_embedding)
        encoded = self.norm(encoded)

        return encoded


class LabelEncoder(nn.Module):
    """标签编码器"""

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # 不同任务类型的编码器
        self.class_embedding = nn.Embedding(1000, self.hidden_dim)  # 支持最多1000个类
        self.regression_encoder = nn.Linear(1, self.hidden_dim)
        self.multilabel_encoder = nn.Linear(100, self.hidden_dim)  # 支持最多100个标签

    def forward(self,
                labels: Any,
                task_type: str,
                metadata: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        编码标签

        Args:
            labels: 标签数据
            task_type: classification, regression, multilabel, link_prediction
            metadata: 元数据

        Returns:
            (batch_size, hidden_dim) 或 (hidden_dim,)
        """
        # Ensure labels tensor lives on the same device as module params
        device = next(self.parameters()).device
        if not hasattr(self, "_label_vocab"):
            self._label_vocab = {}

        if isinstance(labels, torch.Tensor):
            batch_size = labels.size(0) if labels.dim() > 0 else 1
            labels = labels.to(device)
        else:
            numeric_list = self._ensure_numeric_list(labels, as_int=False)
            labels = torch.tensor(numeric_list, device=device)
            batch_size = labels.size(0) if labels.dim() > 0 else 1

        if task_type == "classification":
            # 分类任务：确保标签为非负整数索引，并根据需要扩展嵌入表
            numeric = self._ensure_numeric_list(labels, as_int=True)
            labels = torch.tensor(numeric, device=device, dtype=torch.long).unsqueeze(0)
            # clamp negatives (should not happen after dataset normalization)
            if (labels < 0).any():
                labels = labels.clamp_min(0)
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
            # 回归任务
            if labels.dim() == 0:
                labels = labels.unsqueeze(0).unsqueeze(-1)
            elif labels.dim() == 1:
                labels = labels.unsqueeze(-1)
            encoded = self.regression_encoder(labels.float())

        elif task_type == "multilabel":
            # 多标签任务
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            # 填充或截断到固定长度
            if labels.size(-1) < 100:
                padding = torch.zeros(batch_size, 100 - labels.size(-1), device=device)
                labels = torch.cat([labels, padding], dim=-1)
            else:
                labels = labels[:, :100]
            encoded = self.multilabel_encoder(labels.float())

        elif task_type == "link_prediction":
            # 链接预测：将目标节点ID编码（与分类同样的安全处理）
            numeric = self._ensure_numeric_list(labels, as_int=True)
            labels = torch.tensor(numeric, device=device, dtype=torch.long).unsqueeze(0)
            if (labels < 0).any():
                labels = labels.clamp_min(0)
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

        else:
            # 默认：零向量
            encoded = torch.zeros(batch_size, self.hidden_dim, device=device)

        if encoded.dim() > 2:
            encoded = encoded.squeeze(1)

        return encoded

    def _ensure_numeric_list(self, labels: Any, as_int: bool = False) -> List[float]:
        """
        Convert labels (possibly nested lists, tensors, strings) into numeric lists.
        """
        result: List[float] = []

        if isinstance(labels, torch.Tensor):
            flat = labels.detach().cpu().view(-1).tolist()
            for val in flat:
                result.append(int(val) if as_int else float(val))
            return result

        if isinstance(labels, (list, tuple)):
            for item in labels:
                result.extend(self._ensure_numeric_list(item, as_int=as_int))
            return result

        if isinstance(labels, (int, float, bool)):
            val = int(labels) if as_int else float(labels)
            result.append(val)
            return result

        key = str(labels)
        if key not in self._label_vocab:
            self._label_vocab[key] = len(self._label_vocab)
        mapped = self._label_vocab[key]
        result.append(int(mapped) if as_int else float(mapped))
        return result

    def _to_class_indices(self, labels: Any, device: torch.device) -> torch.Tensor:
        """Normalize arbitrary labels into contiguous integer indices."""
        if isinstance(labels, torch.Tensor):
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            return labels.long().to(device)

        if not isinstance(labels, (list, tuple)):
            labels = [labels]

        indices = []
        for label in labels:
            if isinstance(label, torch.Tensor):
                if label.numel() == 1:
                    label = label.item()
                else:
                    raise ValueError("Classification label tensor has more than one element")

            if isinstance(label, (int, float)):
                indices.append(int(label))
            else:
                key = str(label)
                if key not in self._class_vocab:
                    self._class_vocab[key] = len(self._class_vocab)
                indices.append(self._class_vocab[key])

        if not indices:
            indices = [0]
        return torch.tensor(indices, device=device, dtype=torch.long)


class ICLTransformerLayer(nn.Module):
    """ICL Transformer层"""

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.icl_num_heads
        self.dropout_rate = config.dropout_rate

        # 自注意力
        self.self_attention = nn.MultiheadAttention(
            self.hidden_dim,
            self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )

        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, hidden_dim)
        """
        # 自注意力
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attention(x, x, x)
        x = residual + self.dropout(attn_output)

        # 前馈网络
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + self.dropout(ff_output)

        return x


class ClassificationHead(nn.Module):
    """分类任务头"""

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
    """回归任务头"""

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
    """链接预测任务头"""

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 返回节点嵌入，用于相似度计算
        return self.projection(x)
