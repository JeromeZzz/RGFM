"""
时序异构图数据结构
"""

import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
import numpy as np
from collections import defaultdict


class TemporalHeterogeneousGraph:
    """
    时序异构图
    支持多种节点类型、边类型和时间信息
    """

    def __init__(self):
        # 节点信息
        self.node_types: List[str] = []
        self.node_counts: Dict[str, int] = {}
        self.node_features: Dict[str, torch.Tensor] = {}
        self.node_timestamps: Dict[str, torch.Tensor] = {}

        # 边信息
        self.edge_types: List[Tuple[str, str, str]] = []  # (source_type, edge_name, target_type)
        self.edge_indices: Dict[Tuple[str, str, str], torch.Tensor] = {}
        self.edge_timestamps: Dict[Tuple[str, str, str], torch.Tensor] = {}
        self.edge_features: Dict[Tuple[str, str, str], torch.Tensor] = {}

        # 全局信息
        self.global_node_id_offset: Dict[str, int] = {}
        self.total_nodes = 0

        # 时间信息
        self.min_timestamp: Optional[datetime] = None
        self.max_timestamp: Optional[datetime] = None

    def add_node_type(self, node_type: str, num_nodes: int,
                      features: Optional[torch.Tensor] = None,
                      timestamps: Optional[torch.Tensor] = None) -> None:
        """
        添加节点类型

        Args:
            node_type: 节点类型名称
            num_nodes: 节点数量
            features: 节点特征 (num_nodes, feature_dim)
            timestamps: 节点时间戳 (num_nodes,)
        """
        if node_type in self.node_types:
            raise ValueError(f"Node type {node_type} already exists")

        self.node_types.append(node_type)
        self.node_counts[node_type] = num_nodes
        self.global_node_id_offset[node_type] = self.total_nodes
        self.total_nodes += num_nodes

        if features is not None:
            self.node_features[node_type] = features

        if timestamps is not None:
            self.node_timestamps[node_type] = timestamps

    def add_edge_type(self, edge_name: str,
                      node_types: Tuple[str, str],
                      edge_index: Optional[torch.Tensor] = None,
                      timestamps: Optional[torch.Tensor] = None,
                      features: Optional[torch.Tensor] = None) -> None:
        """
        添加边类型

        Args:
            edge_name: 边类型名称
            node_types: (源节点类型, 目标节点类型)
            edge_index: 边索引 (2, num_edges)
            timestamps: 边时间戳 (num_edges,)
            features: 边特征 (num_edges, feature_dim)
        """
        source_type, target_type = node_types
        edge_type = (source_type, edge_name, target_type)

        if edge_type in self.edge_types:
            raise ValueError(f"Edge type {edge_type} already exists")

        self.edge_types.append(edge_type)

        if edge_index is not None:
            self.edge_indices[edge_type] = edge_index

        if timestamps is not None:
            self.edge_timestamps[edge_type] = timestamps

        if features is not None:
            self.edge_features[edge_type] = features

    def get_subgraph(self, center_nodes: Dict[str, List[int]],
                     num_hops: int,
                     max_nodes_per_hop: Optional[List[int]] = None,
                     time_window: Optional[Tuple[datetime, datetime]] = None) -> 'TemporalHeterogeneousGraph':
        """
        获取子图

        Args:
            center_nodes: {node_type: [node_ids]}
            num_hops: 采样跳数
            max_nodes_per_hop: 每跳最大节点数
            time_window: 时间窗口 (start_time, end_time)

        Returns:
            子图
        """
        subgraph = TemporalHeterogeneousGraph()

        # 实现子图采样逻辑
        # 这里只是一个简化的框架
        sampled_nodes = defaultdict(set)

        # 添加中心节点
        for node_type, node_ids in center_nodes.items():
            sampled_nodes[node_type].update(node_ids)

        # 多跳采样
        for hop in range(num_hops):
            # 获取当前层的邻居
            # 这里需要实现具体的邻居采样逻辑
            pass

        # 构建子图
        # 添加采样到的节点和边

        return subgraph

    def to_pyg_data(self) -> Any:
        """
        转换为PyTorch Geometric的HeteroData格式
        """
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            raise ImportError("Please install torch_geometric")

        data = HeteroData()

        # 添加节点
        for node_type in self.node_types:
            if node_type in self.node_features:
                data[node_type].x = self.node_features[node_type]

            if node_type in self.node_timestamps:
                data[node_type].timestamps = self.node_timestamps[node_type]

            data[node_type].num_nodes = self.node_counts[node_type]

        # 添加边
        for edge_type in self.edge_types:
            if edge_type in self.edge_indices:
                data[edge_type].edge_index = self.edge_indices[edge_type]

            if edge_type in self.edge_timestamps:
                data[edge_type].timestamps = self.edge_timestamps[edge_type]

            if edge_type in self.edge_features:
                data[edge_type].edge_attr = self.edge_features[edge_type]

        return data

    def get_temporal_neighbors(self, node_type: str, node_id: int,
                               timestamp: datetime,
                               time_delta: float = 1.0) -> Dict[str, List[int]]:
        """
        获取时间邻居

        Args:
            node_type: 节点类型
            node_id: 节点ID
            timestamp: 查询时间
            time_delta: 时间窗口大小（天）

        Returns:
            {邻居类型: [邻居ID列表]}
        """
        neighbors = defaultdict(list)

        # 遍历所有相关的边类型
        for edge_type in self.edge_types:
            source_type, edge_name, target_type = edge_type

            # 检查是否是相关的边类型
            if source_type == node_type:
                # 出边
                if edge_type in self.edge_indices and edge_type in self.edge_timestamps:
                    edge_index = self.edge_indices[edge_type]
                    edge_times = self.edge_timestamps[edge_type]

                    # 找到源节点是当前节点的边
                    mask = edge_index[0] == node_id

                    # 时间过滤
                    # 这里需要实现时间戳比较逻辑

                    # 收集邻居
                    valid_neighbors = edge_index[1][mask].tolist()
                    neighbors[target_type].extend(valid_neighbors)

            elif target_type == node_type:
                # 入边
                if edge_type in self.edge_indices and edge_type in self.edge_timestamps:
                    edge_index = self.edge_indices[edge_type]
                    edge_times = self.edge_timestamps[edge_type]

                    # 找到目标节点是当前节点的边
                    mask = edge_index[1] == node_id

                    # 时间过滤
                    # 这里需要实现时间戳比较逻辑

                    # 收集邻居
                    valid_neighbors = edge_index[0][mask].tolist()
                    neighbors[source_type].extend(valid_neighbors)

        return dict(neighbors)

    def __repr__(self) -> str:
        return (f"TemporalHeterogeneousGraph(\n"
                f"  node_types={self.node_types},\n"
                f"  node_counts={self.node_counts},\n"
                f"  edge_types={[f'{s}-{e}->{t}' for s, e, t in self.edge_types]},\n"
                f"  total_nodes={self.total_nodes}\n"
                f")")