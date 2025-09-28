"""
后向子图采样器
用于构建预测时的输入特征子图
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import random

from data.temporal_graph import TemporalHeterogeneousGraph
from config.model_config import SamplingConfig


class BackwardSubgraphSampler:
    """
    后向子图采样器
    从目标实体出发，采样历史时间内的k跳邻居
    """

    def __init__(self, config: SamplingConfig):
        self.config = config
        self.strategy = config.strategy
        self.max_neighbors_per_hop = config.max_neighbors_per_hop
        self.time_decay_factor = config.time_decay_factor

    def sample(self,
               graph: TemporalHeterogeneousGraph,
               target_entity: Tuple[str, int],  # (node_type, node_id)
               prediction_time: datetime,
               num_hops: int = 2,
               max_nodes: int = 300) -> TemporalHeterogeneousGraph:
        """
        采样后向子图

        Args:
            graph: 完整的时序异构图
            target_entity: 目标实体 (节点类型, 节点ID)
            prediction_time: 预测时间点
            num_hops: 采样跳数
            max_nodes: 最大节点数

        Returns:
            采样得到的子图 G≤t[e]
        """
        node_type, node_id = target_entity

        # 初始化采样节点集合
        sampled_nodes = defaultdict(set)
        sampled_nodes[node_type].add(node_id)

        # 记录节点的跳数信息
        hop_distances = defaultdict(dict)
        hop_distances[node_type][node_id] = 0

        # 记录采样的边
        sampled_edges = defaultdict(list)

        # 多跳采样
        for hop in range(num_hops):
            # 获取当前层的节点
            current_layer_nodes = []
            for nt, nodes in sampled_nodes.items():
                for n in nodes:
                    if hop_distances[nt].get(n, float('inf')) == hop:
                        current_layer_nodes.append((nt, n))

            if not current_layer_nodes:
                break

            # 为当前层的每个节点采样邻居
            max_neighbors_this_hop = self.max_neighbors_per_hop[min(hop, len(self.max_neighbors_per_hop) - 1)]

            for curr_node_type, curr_node_id in current_layer_nodes:
                neighbors = self._sample_neighbors(
                    graph,
                    curr_node_type,
                    curr_node_id,
                    prediction_time,
                    max_neighbors_this_hop
                )

                # 添加邻居节点和边
                for edge_type, neighbor_list in neighbors.items():
                    source_type, edge_name, target_type = edge_type

                    for neighbor_id, timestamp in neighbor_list:
                        # 检查时间约束
                        if timestamp > prediction_time:
                            continue

                        # 添加节点
                        if source_type == curr_node_type:
                            # 出边
                            sampled_nodes[target_type].add(neighbor_id)
                            if neighbor_id not in hop_distances[target_type]:
                                hop_distances[target_type][neighbor_id] = hop + 1

                            # 记录边
                            sampled_edges[edge_type].append((curr_node_id, neighbor_id, timestamp))
                        else:
                            # 入边
                            sampled_nodes[source_type].add(neighbor_id)
                            if neighbor_id not in hop_distances[source_type]:
                                hop_distances[source_type][neighbor_id] = hop + 1

                            # 记录边
                            sampled_edges[edge_type].append((neighbor_id, curr_node_id, timestamp))

                # 检查是否达到最大节点数
                total_nodes = sum(len(nodes) for nodes in sampled_nodes.values())
                if total_nodes >= max_nodes:
                    break

        # 构建子图
        subgraph = self._build_subgraph(graph, sampled_nodes, sampled_edges, hop_distances)

        return subgraph

    def _sample_neighbors(self,
                          graph: TemporalHeterogeneousGraph,
                          node_type: str,
                          node_id: int,
                          prediction_time: datetime,
                          max_neighbors: int) -> Dict[Tuple[str, str, str], List[Tuple[int, datetime]]]:
        """
        采样单个节点的邻居

        Returns:
            {edge_type: [(neighbor_id, timestamp)]}
        """
        neighbors = defaultdict(list)

        # 遍历所有相关的边类型
        for edge_type in graph.edge_types:
            source_type, edge_name, target_type = edge_type

            if edge_type not in graph.edge_indices:
                continue

            edge_index = graph.edge_indices[edge_type]

            # 获取时间戳
            if edge_type in graph.edge_timestamps:
                edge_timestamps = graph.edge_timestamps[edge_type]
            else:
                # 如果没有时间戳，使用默认时间
                edge_timestamps = torch.full((edge_index.shape[1],), 0.0)

            # 根据边的方向采样
            if source_type == node_type:
                # 出边
                mask = edge_index[0] == node_id
                valid_neighbors = edge_index[1][mask]
                valid_timestamps = edge_timestamps[mask]

                # 时间过滤
                time_mask = valid_timestamps <= self._datetime_to_timestamp(prediction_time)
                valid_neighbors = valid_neighbors[time_mask]
                valid_timestamps = valid_timestamps[time_mask]

                # 采样策略
                sampled_indices = self._apply_sampling_strategy(
                    valid_neighbors,
                    valid_timestamps,
                    prediction_time,
                    max_neighbors
                )

                for idx in sampled_indices:
                    neighbors[edge_type].append((
                        valid_neighbors[idx].item(),
                        self._timestamp_to_datetime(valid_timestamps[idx].item())
                    ))

            elif target_type == node_type:
                # 入边
                mask = edge_index[1] == node_id
                valid_neighbors = edge_index[0][mask]
                valid_timestamps = edge_timestamps[mask]

                # 时间过滤
                time_mask = valid_timestamps <= self._datetime_to_timestamp(prediction_time)
                valid_neighbors = valid_neighbors[time_mask]
                valid_timestamps = valid_timestamps[time_mask]

                # 采样策略
                sampled_indices = self._apply_sampling_strategy(
                    valid_neighbors,
                    valid_timestamps,
                    prediction_time,
                    max_neighbors
                )

                for idx in sampled_indices:
                    neighbors[edge_type].append((
                        valid_neighbors[idx].item(),
                        self._timestamp_to_datetime(valid_timestamps[idx].item())
                    ))

        return dict(neighbors)

    def _apply_sampling_strategy(self,
                                 neighbors: torch.Tensor,
                                 timestamps: torch.Tensor,
                                 prediction_time: datetime,
                                 max_neighbors: int) -> List[int]:
        """
        应用采样策略
        """
        if len(neighbors) <= max_neighbors:
            return list(range(len(neighbors)))

        if self.strategy == 'random':
            # 随机采样
            indices = torch.randperm(len(neighbors))[:max_neighbors]
            return indices.tolist()

        elif self.strategy == 'temporal_importance':
            # 基于时间重要性采样
            pred_timestamp = self._datetime_to_timestamp(prediction_time)
            time_diffs = pred_timestamp - timestamps

            # 计算时间衰减权重
            weights = torch.exp(-time_diffs * self.time_decay_factor)

            # 加权采样
            if torch.sum(weights) > 0:
                probs = weights / torch.sum(weights)
                indices = torch.multinomial(probs, max_neighbors, replacement=False)
                return indices.tolist()
            else:
                return torch.randperm(len(neighbors))[:max_neighbors].tolist()

        elif self.strategy == 'structure_aware':
            # 基于结构重要性采样（这里简化为度数）
            # 实际实现应该考虑节点的结构重要性
            return torch.randperm(len(neighbors))[:max_neighbors].tolist()

        else:
            # 默认随机采样
            return torch.randperm(len(neighbors))[:max_neighbors].tolist()

    def _build_subgraph(self,
                        original_graph: TemporalHeterogeneousGraph,
                        sampled_nodes: Dict[str, Set[int]],
                        sampled_edges: Dict[Tuple[str, str, str], List[Tuple[int, int, datetime]]],
                        hop_distances: Dict[str, Dict[int, int]]) -> TemporalHeterogeneousGraph:
        """
        构建采样后的子图
        """
        subgraph = TemporalHeterogeneousGraph()

        # 创建节点ID映射
        node_id_mapping = {}

        # 添加节点
        for node_type, node_ids in sampled_nodes.items():
            node_ids = sorted(list(node_ids))
            num_nodes = len(node_ids)

            # 创建ID映射
            node_id_mapping[node_type] = {
                old_id: new_id for new_id, old_id in enumerate(node_ids)
            }

            # 获取节点特征
            if node_type in original_graph.node_features:
                original_features = original_graph.node_features[node_type]
                features = original_features[node_ids]
            else:
                features = None

            # 获取节点时间戳
            if node_type in original_graph.node_timestamps:
                original_timestamps = original_graph.node_timestamps[node_type]
                timestamps = original_timestamps[node_ids]
            else:
                timestamps = None

            subgraph.add_node_type(node_type, num_nodes, features, timestamps)

            # 存储跳数信息作为额外属性
            hop_tensor = torch.zeros(num_nodes, dtype=torch.long)
            for old_id, new_id in node_id_mapping[node_type].items():
                hop_tensor[new_id] = hop_distances[node_type].get(old_id, -1)

            setattr(subgraph, f'{node_type}_hop_distances', hop_tensor)

        # 添加边
        for edge_type, edge_list in sampled_edges.items():
            if not edge_list:
                continue

            source_type, edge_name, target_type = edge_type

            # 转换边索引
            new_edge_index = []
            new_timestamps = []

            for src_id, tgt_id, timestamp in edge_list:
                if src_id in node_id_mapping[source_type] and tgt_id in node_id_mapping[target_type]:
                    new_src = node_id_mapping[source_type][src_id]
                    new_tgt = node_id_mapping[target_type][tgt_id]
                    new_edge_index.append([new_src, new_tgt])
                    new_timestamps.append(self._datetime_to_timestamp(timestamp))

            if new_edge_index:
                edge_index = torch.tensor(new_edge_index, dtype=torch.long).t()
                timestamps = torch.tensor(new_timestamps, dtype=torch.float)

                subgraph.add_edge_type(edge_name, (source_type, target_type),
                                       edge_index, timestamps)

        return subgraph

    def _datetime_to_timestamp(self, dt: datetime) -> float:
        """将datetime转换为时间戳"""
        return dt.timestamp()

    def _timestamp_to_datetime(self, ts: float) -> datetime:
        """将时间戳转换为datetime"""
        return datetime.fromtimestamp(ts)