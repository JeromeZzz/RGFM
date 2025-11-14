"""
上下文采样器
用于选择和构建上下文示例
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import random
from dataclasses import dataclass
import math

from data.temporal_graph import TemporalHeterogeneousGraph
from .backward_sampler import BackwardSubgraphSampler
from config.model_config import SamplingConfig, TaskConfig
from .context_label_table import ContextLabelRecord, InContextLabelTable


@dataclass
class ContextExample:
    """上下文示例"""
    subgraph: TemporalHeterogeneousGraph
    entity: Tuple[str, int]  # (node_type, node_id)
    timestamp: datetime
    label: Any  # 标签值
    metadata: Dict[str, Any] = None


class ContextSampler:
    """
    上下文采样器
    负责选择合适的历史示例作为上下文
    """

    def __init__(self,
                 config: SamplingConfig,
                 backward_sampler: BackwardSubgraphSampler):
        self.config = config
        self.backward_sampler = backward_sampler
        self.context_cache = {}  # 缓存已采样的上下文
        self.label_table: Optional[InContextLabelTable] = None

    def attach_label_table(self, table: InContextLabelTable) -> None:
        """Attach the online in-context label table."""
        self.label_table = table

    def sample_context(self,
                       graph: TemporalHeterogeneousGraph,
                       target_entity: Tuple[str, int],
                       prediction_time: datetime,
                       task_config: TaskConfig,
                       num_examples: int = 10,
                       strategy: str = 'mixed') -> List[ContextExample]:
        """
        采样上下文示例

        Args:
            graph: 完整图
            target_entity: 目标实体
            prediction_time: 预测时间
            task_config: 任务配置
            num_examples: 上下文示例数量
            strategy: 采样策略 ('temporal', 'structural', 'mixed', 'self' 等)

        Returns:
            上下文示例列表
        """
        if self.label_table is not None:
            return self._sample_from_label_table(
                graph,
                target_entity,
                prediction_time,
                task_config,
                num_examples,
                strategy
            )

        # 当未挂载 label table 时，退回到旧的启发式策略
        context_examples: List[ContextExample] = []

        if strategy == 'temporal':
            context_examples = self._sample_temporal_context(
                graph, target_entity, prediction_time, task_config, num_examples
            )
        elif strategy == 'structural':
            context_examples = self._sample_structural_context(
                graph, target_entity, prediction_time, task_config, num_examples
            )
        elif strategy == 'self':
            context_examples = self._sample_self_context(
                graph, target_entity, prediction_time, task_config, num_examples
            )
        else:
            num_temporal = num_examples // 3
            num_structural = num_examples // 3
            num_self = num_examples - num_temporal - num_structural

            context_examples = (
                self._sample_temporal_context(
                    graph, target_entity, prediction_time, task_config, num_temporal
                )
                + self._sample_structural_context(
                    graph, target_entity, prediction_time, task_config, num_structural
                )
                + self._sample_self_context(
                    graph, target_entity, prediction_time, task_config, num_self
                )
            )

        return context_examples

    def _sample_from_label_table(self,
                                 graph: TemporalHeterogeneousGraph,
                                 target_entity: Tuple[str, int],
                                 prediction_time: datetime,
                                 task_config: TaskConfig,
                                 num_examples: int,
                                 strategy: str) -> List[ContextExample]:
        """使用上下文标签表构建上下文示例。"""
        node_type, _ = target_entity
        strategies = self._expand_strategy(strategy)
        budget = max(1, math.ceil(num_examples / len(strategies)))
        cutoff = float(prediction_time.timestamp())
        contexts: List[ContextExample] = []

        for strat in strategies:
            records = self.label_table.sample(
                node_type=node_type,
                before_time=cutoff,
                k=budget,
                strategy=strat,
                fixed_interval_seconds=getattr(
                    self.config, "context_sampling_interval_seconds", 86400.0
                ),
            )
            for record in records:
                ctx = self._build_context_example(graph, record, task_config, strat)
                if ctx:
                    contexts.append(ctx)
                if len(contexts) >= num_examples:
                    break
            if len(contexts) >= num_examples:
                break

        if not contexts:
            # fallback to legacy temporal strategy to avoid empty contexts
            return self._sample_temporal_context(
                graph, target_entity, prediction_time, task_config, num_examples
            )

        return contexts

    def _build_context_example(self,
                               graph: TemporalHeterogeneousGraph,
                               record: ContextLabelRecord,
                               task_config: TaskConfig,
                               strategy: str) -> Optional[ContextExample]:
        """将标签记录转换为 ContextExample。"""
        timestamp = datetime.fromtimestamp(record.timestamp)
        num_hops = getattr(self.config, 'num_hops', len(self.config.max_neighbors_per_hop))
        max_nodes = getattr(self.config, 'max_neighbors', 300)
        try:
            subgraph = self.backward_sampler.sample(
                graph,
                record.entity,
                timestamp,
                num_hops=num_hops,
                max_nodes=max_nodes,
            )
        except Exception:
            return None

        metadata = dict(record.metadata or {})
        metadata.setdefault('source', 'label_table')
        metadata['strategy'] = strategy

        return ContextExample(
            subgraph=subgraph,
            entity=record.entity,
            timestamp=timestamp,
            label=record.label,
            metadata=metadata,
        )

    def _expand_strategy(self, strategy: str) -> List[str]:
        """兼容不同策略名称并映射到新的策略集合。"""
        if strategy in ('uniform', 'most_recent', 'fixed_interval'):
            return [strategy]
        if strategy == 'mixed':
            return ['most_recent', 'uniform', 'fixed_interval']
        if strategy in ('temporal', 'structural', 'self'):
            return ['most_recent']
        return ['uniform']

    def _sample_temporal_context(self,
                                 graph: TemporalHeterogeneousGraph,
                                 target_entity: Tuple[str, int],
                                 prediction_time: datetime,
                                 task_config: TaskConfig,
                                 num_examples: int) -> List[ContextExample]:
        """
        基于时间邻近度采样上下文
        选择时间上接近预测时间的历史实体
        """
        examples = []
        node_type, node_id = target_entity

        # 获取同类型的所有节点
        if node_type not in graph.node_timestamps:
            return examples

        timestamps = graph.node_timestamps[node_type]
        pred_timestamp = self._datetime_to_timestamp(prediction_time)

        # 计算时间差
        time_diffs = torch.abs(timestamps - pred_timestamp)

        # 排除目标节点和未来的节点
        valid_mask = torch.ones(len(timestamps), dtype=torch.bool)
        valid_mask[node_id] = False
        valid_mask[timestamps > pred_timestamp] = False

        if not valid_mask.any():
            return examples

        # 选择时间最近的节点
        valid_indices = torch.where(valid_mask)[0]
        valid_time_diffs = time_diffs[valid_mask]

        # 按时间差排序
        sorted_indices = torch.argsort(valid_time_diffs)
        selected_indices = sorted_indices[:num_examples]

        # 创建上下文示例
        for idx in selected_indices:
            actual_idx = valid_indices[idx].item()
            context_time = self._timestamp_to_datetime(timestamps[actual_idx].item())

            # 采样子图
            context_subgraph = self.backward_sampler.sample(
                graph,
                (node_type, actual_idx),
                context_time,
                num_hops=self.config.max_neighbors_per_hop.__len__() - 1
            )

            # 生成标签
            label = self._generate_label(
                graph,
                (node_type, actual_idx),
                context_time,
                task_config
            )

            examples.append(ContextExample(
                subgraph=context_subgraph,
                entity=(node_type, actual_idx),
                timestamp=context_time,
                label=label
            ))

        return examples

    def _sample_structural_context(self,
                                   graph: TemporalHeterogeneousGraph,
                                   target_entity: Tuple[str, int],
                                   prediction_time: datetime,
                                   task_config: TaskConfig,
                                   num_examples: int) -> List[ContextExample]:
        """
        基于结构相似性采样上下文
        选择图结构相似的历史实体
        """
        examples = []
        node_type, node_id = target_entity

        # 获取目标节点的结构特征
        target_features = self._extract_structural_features(
            graph, target_entity, prediction_time
        )

        # 获取所有候选节点
        num_nodes = graph.node_counts.get(node_type, 0)
        if num_nodes == 0:
            return examples

        # 计算所有节点的结构特征
        candidate_features = []
        candidate_indices = []

        for idx in range(num_nodes):
            if idx == node_id:
                continue

            # 检查时间约束
            if node_type in graph.node_timestamps:
                node_time = graph.node_timestamps[node_type][idx]
                if node_time > self._datetime_to_timestamp(prediction_time):
                    continue

            features = self._extract_structural_features(
                graph, (node_type, idx), prediction_time
            )
            candidate_features.append(features)
            candidate_indices.append(idx)

        if not candidate_features:
            return examples

        # 计算相似度
        candidate_features = torch.stack(candidate_features)
        target_features = target_features.unsqueeze(0)

        # 使用余弦相似度
        similarities = torch.nn.functional.cosine_similarity(
            candidate_features, target_features, dim=1
        )

        # 选择最相似的节点
        sorted_indices = torch.argsort(similarities, descending=True)
        selected_indices = sorted_indices[:num_examples]

        # 创建上下文示例
        for idx in selected_indices:
            actual_idx = candidate_indices[idx]

            # 确定上下文时间
            if node_type in graph.node_timestamps:
                context_time = self._timestamp_to_datetime(
                    graph.node_timestamps[node_type][actual_idx].item()
                )
            else:
                # 使用预测时间之前的随机时间
                days_before = random.randint(1, 30)
                context_time = prediction_time - timedelta(days=days_before)

            # 采样子图
            context_subgraph = self.backward_sampler.sample(
                graph,
                (node_type, actual_idx),
                context_time,
                num_hops=self.config.max_neighbors_per_hop.__len__() - 1
            )

            # 生成标签
            label = self._generate_label(
                graph,
                (node_type, actual_idx),
                context_time,
                task_config
            )

            examples.append(ContextExample(
                subgraph=context_subgraph,
                entity=(node_type, actual_idx),
                timestamp=context_time,
                label=label
            ))

        return examples

    def _sample_self_context(self,
                             graph: TemporalHeterogeneousGraph,
                             target_entity: Tuple[str, int],
                             prediction_time: datetime,
                             task_config: TaskConfig,
                             num_examples: int) -> List[ContextExample]:
        """
        自回归上下文
        使用目标实体自身的历史状态
        """
        examples = []
        node_type, node_id = target_entity

        # 生成历史时间点
        time_points = []
        for i in range(1, num_examples + 1):
            # 每周一个历史点
            hist_time = prediction_time - timedelta(weeks=i)
            time_points.append(hist_time)

        # 为每个历史时间点创建上下文
        for hist_time in time_points:
            # 采样历史子图
            context_subgraph = self.backward_sampler.sample(
                graph,
                target_entity,
                hist_time,
                num_hops=self.config.max_neighbors_per_hop.__len__() - 1
            )

            # 生成历史标签
            label = self._generate_label(
                graph,
                target_entity,
                hist_time,
                task_config
            )

            examples.append(ContextExample(
                subgraph=context_subgraph,
                entity=target_entity,
                timestamp=hist_time,
                label=label,
                metadata={'is_self_context': True}
            ))

        return examples

    def _extract_structural_features(self,
                                     graph: TemporalHeterogeneousGraph,
                                     entity: Tuple[str, int],
                                     time_point: datetime) -> torch.Tensor:
        """
        提取节点的结构特征
        包括度数、邻居类型分布等
        """
        node_type, node_id = entity
        features = []

        # 度数特征
        in_degree = 0
        out_degree = 0

        for edge_type in graph.edge_types:
            source_type, _, target_type = edge_type

            if edge_type not in graph.edge_indices:
                continue

            edge_index = graph.edge_indices[edge_type]

            if source_type == node_type:
                # 出度
                out_degree += (edge_index[0] == node_id).sum().item()

            if target_type == node_type:
                # 入度
                in_degree += (edge_index[1] == node_id).sum().item()

        features.extend([in_degree, out_degree, in_degree + out_degree])

        # 邻居类型分布
        neighbor_type_counts = defaultdict(int)

        neighbors = graph.get_temporal_neighbors(
            node_type, node_id, time_point, time_delta=30.0
        )

        for n_type, n_list in neighbors.items():
            neighbor_type_counts[n_type] = len(n_list)

        # 转换为固定长度的特征向量
        for nt in graph.node_types:
            features.append(neighbor_type_counts.get(nt, 0))

        return torch.tensor(features, dtype=torch.float)

    def _generate_label(self,
                        graph: TemporalHeterogeneousGraph,
                        entity: Tuple[str, int],
                        context_time: datetime,
                        task_config: TaskConfig) -> Any:
        """
        生成上下文的标签
        这是一个简化的实现，实际应该基于任务配置和前向采样器
        """
        # 根据任务类型生成模拟标签
        if task_config.task_type == 'classification':
            # 分类任务：随机类别
            num_classes = task_config.num_classes or 2
            return torch.randint(0, num_classes, (1,)).item()

        elif task_config.task_type == 'regression':
            # 回归任务：随机数值
            return torch.randn(1).item()

        elif task_config.task_type == 'multilabel':
            # 多标签任务：随机多热编码
            num_labels = task_config.num_labels or 5
            return torch.randint(0, 2, (num_labels,)).float()

        elif task_config.task_type == 'link_prediction':
            # 链接预测：随机目标节点
            node_type, _ = entity
            num_nodes = graph.node_counts.get(node_type, 100)
            return torch.randint(0, num_nodes, (1,)).item()

        else:
            return 0.0

    def _datetime_to_timestamp(self, dt: datetime) -> float:
        """将datetime转换为时间戳"""
        return dt.timestamp()

    def _timestamp_to_datetime(self, ts: float) -> datetime:
        """将时间戳转换为datetime"""
        return datetime.fromtimestamp(ts)
