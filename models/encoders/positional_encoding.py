"""
位置编码模块
包括节点类型编码、跳数编码、时间编码和子图结构编码
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import math
from datetime import datetime

from config.model_config import KumoRFMConfig


class PositionalEncoding(nn.Module):
    """标准正弦位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:x.size(1)]


class NodeTypeEncoder(nn.Module):
    """
    节点类型编码器
    为不同表的节点添加类型标识
    """

    def __init__(self, config: KumoRFMConfig, num_node_types: int):
        super().__init__()
        self.config = config

        # 节点类型嵌入
        self.node_type_embedding = nn.Embedding(
            num_node_types,
            config.hidden_dim
        )

    def forward(self, node_types: torch.Tensor) -> torch.Tensor:
        """
        node_types: (batch_size, num_nodes) 节点类型ID
        返回: (batch_size, num_nodes, hidden_dim)
        """
        return self.node_type_embedding(node_types)


class HopEncoder(nn.Module):
    """
    跳数编码器
    编码节点到中心节点的距离
    """

    def __init__(self, config: KumoRFMConfig, max_hops: int = 3):
        super().__init__()
        self.config = config
        self.max_hops = max_hops

        # 跳数嵌入
        self.hop_embedding = nn.Embedding(
            max_hops + 1,  # 0跳表示中心节点
            config.hidden_dim
        )

        # 可学习的权重矩阵
        self.hop_weights = nn.Parameter(
            torch.ones(max_hops + 1, config.hidden_dim)
        )

    def forward(self, hop_distances: torch.Tensor) -> torch.Tensor:
        """
        hop_distances: (batch_size, num_nodes) 跳数距离
        返回: (batch_size, num_nodes, hidden_dim)
        """
        # 限制最大跳数
        hop_distances = torch.clamp(hop_distances, 0, self.max_hops)

        # 获取嵌入
        embeddings = self.hop_embedding(hop_distances)

        # 应用可学习权重
        weights = self.hop_weights[hop_distances]
        weighted_embeddings = embeddings * weights

        return weighted_embeddings


class TimeEncoder(nn.Module):
    """
    时间编码器
    编码节点相对于预测时间的时间位置
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # 时间变换网络
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        )

        # 可学习的时间尺度参数
        self.time_scales = nn.Parameter(torch.ones(4))  # 天、周、月、年

    def forward(self, time_diffs: torch.Tensor) -> torch.Tensor:
        """
        time_diffs: (batch_size, num_nodes) 时间差（秒）
        返回: (batch_size, num_nodes, hidden_dim)
        """
        batch_size, num_nodes = time_diffs.shape

        # 转换为不同时间尺度
        time_features = []

        # 天
        days = time_diffs / (24 * 3600)
        time_features.append(days * self.time_scales[0])

        # 周
        weeks = time_diffs / (7 * 24 * 3600)
        time_features.append(weeks * self.time_scales[1])

        # 月（近似30天）
        months = time_diffs / (30 * 24 * 3600)
        time_features.append(months * self.time_scales[2])

        # 年（近似365天）
        years = time_diffs / (365 * 24 * 3600)
        time_features.append(years * self.time_scales[3])

        # 组合特征
        combined_time = sum(time_features) / len(time_features)
        combined_time = combined_time.unsqueeze(-1)  # (batch_size, num_nodes, 1)

        # 通过MLP
        time_encoding = self.time_mlp(combined_time)

        return time_encoding


class SubgraphStructureEncoder(nn.Module):
    """
    子图结构编码器
    使用轻量级GNN编码局部图结构
    """

    def __init__(self, config: KumoRFMConfig, num_layers: int = 2):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_layers = num_layers

        # 轻量级GNN层
        self.gnn_layers = nn.ModuleList([
            LightweightGNNLayer(self.hidden_dim)
            for _ in range(num_layers)
        ])

        # 最终投影
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self,
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        node_features: (batch_size, num_nodes, hidden_dim)
        edge_index: (2, num_edges)
        edge_types: (num_edges,) 边类型

        返回: (batch_size, num_nodes, hidden_dim)
        """
        x = node_features

        # 应用GNN层
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_types)

        # 最终投影
        structure_encoding = self.output_projection(x)

        return structure_encoding


class LightweightGNNLayer(nn.Module):
    """轻量级GNN层"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 消息传递
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 更新
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 残差连接
        self.residual = nn.Linear(hidden_dim, hidden_dim)

        # 归一化
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        简单的消息传递
        """
        # 假设batch_size=1，简化处理
        if x.dim() == 3:
            batch_size, num_nodes, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)

        src, dst = edge_index

        # 消息传递
        src_features = x[src]
        dst_features = x[dst]
        messages = self.message_mlp(torch.cat([src_features, dst_features], dim=-1))

        # 聚合消息
        # 使用scatter_add进行聚合
        aggregated = torch.zeros_like(x)
        aggregated.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # 更新节点
        updated = self.update_mlp(torch.cat([x, aggregated], dim=-1))

        # 残差连接和归一化
        x = self.norm(updated + self.residual(x))

        # 恢复batch维度
        if 'batch_size' in locals():
            x = x.view(batch_size, num_nodes, hidden_dim)

        return x


class MultiElementTokenizer(nn.Module):
    """
    多元素Token化器
    将节点分解为5个元素：特征、类型、跳数、时间、结构
    """

    def __init__(self,
                 config: KumoRFMConfig,
                 num_node_types: int,
                 max_hops: int = 3):
        super().__init__()
        self.config = config

        # 各种编码器
        self.node_type_encoder = NodeTypeEncoder(config, num_node_types)
        self.hop_encoder = HopEncoder(config, max_hops)
        self.time_encoder = TimeEncoder(config)
        self.structure_encoder = SubgraphStructureEncoder(config)

        # 融合权重
        self.fusion_weights = nn.Parameter(torch.ones(5))

    def forward(self,
                node_features: torch.Tensor,
                node_types: torch.Tensor,
                hop_distances: torch.Tensor,
                time_diffs: torch.Tensor,
                edge_index: torch.Tensor,
                edge_types: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        生成多元素token表示

        返回: (batch_size, num_nodes, hidden_dim)
        """
        # 1. 节点特征（已经编码过）
        feat_encoding = node_features

        # 2. 节点类型编码
        type_encoding = self.node_type_encoder(node_types)

        # 3. 跳数编码
        hop_encoding = self.hop_encoder(hop_distances)

        # 4. 时间编码
        time_encoding = self.time_encoder(time_diffs)

        # 5. 结构编码
        structure_encoding = self.structure_encoder(
            node_features, edge_index, edge_types
        )

        # 加权融合
        weights = torch.softmax(self.fusion_weights, dim=0)

        fused_encoding = (
                weights[0] * feat_encoding +
                weights[1] * type_encoding +
                weights[2] * hop_encoding +
                weights[3] * time_encoding +
                weights[4] * structure_encoding
        )

        return fused_encoding