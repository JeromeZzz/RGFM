"""
KumoRFM主模型
整合所有组件的端到端模型
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from config.model_config import KumoRFMConfig, TaskConfig
from data.temporal_graph import TemporalHeterogeneousGraph
from sampling.backward_sampler import BackwardSubgraphSampler
from sampling.context_sampler import ContextSampler, ContextExample
from .encoders.multimodal_encoder import MultiModalEncoder
from .encoders.table_encoder import MultiTableEncoder
from .encoders.positional_encoding import MultiElementTokenizer
from .relgt.relgt_model import RelGTWrapper
from .icl.icl_module import ICLModule, ClassificationHead, RegressionHead, LinkPredictionHead
from .icl.dual_context import DualContextMechanism


class KumoRFM(nn.Module):
    """
    KumoRFM (Kumo Relational Foundation Model)
    关系数据基础模型
    """

    def __init__(self,
                 config: KumoRFMConfig,
                 database_schema: Dict[str, Dict[str, str]]):
        """
        Args:
            config: 模型配置
            database_schema: 数据库模式 {table_name: {column_name: column_type}}
        """
        super().__init__()
        self.config = config
        self.database_schema = database_schema

        # 初始化采样器
        sampling_config = config.__dict__.get('sampling_config', None)
        if sampling_config is None:
            from config.model_config import SamplingConfig
            sampling_config = SamplingConfig()

        self.backward_sampler = BackwardSubgraphSampler(sampling_config)
        self.context_sampler = ContextSampler(sampling_config, self.backward_sampler)

        # 初始化编码器
        self._init_encoders()

        # 初始化RelGT
        num_node_types = len(database_schema)
        self.relgt = RelGTWrapper(config, num_node_types)

        # 初始化ICL模块
        self.icl_module = ICLModule(config)
        self.dual_context = DualContextMechanism(config)

        # 注册任务头
        self._register_task_heads()

        # 缓存
        self.encoding_cache = {}

    def _init_encoders(self):
        """初始化各种编码器"""
        # 多模态编码器
        from config.model_config import ColumnEncoderConfig
        column_config = ColumnEncoderConfig()
        self.multimodal_encoder = MultiModalEncoder(column_config, self.config.hidden_dim)

        # 注册所有列
        for table_name, columns in self.database_schema.items():
            for column_name, column_type in columns.items():
                self.multimodal_encoder.register_column(
                    f"{table_name}.{column_name}",
                    column_type
                )

        # 表编码器
        table_names = list(self.database_schema.keys())
        self.table_encoder = MultiTableEncoder(self.config, table_names)

        # Token化器
        num_node_types = len(self.database_schema)
        self.tokenizer = MultiElementTokenizer(self.config, num_node_types)

    def _register_task_heads(self):
        """注册任务特定的预测头"""
        # 分类任务头（默认二分类）
        self.icl_module.register_task_head(
            'classification',
            ClassificationHead(self.config, num_classes=2)
        )

        # 回归任务头
        self.icl_module.register_task_head(
            'regression',
            RegressionHead(self.config)
        )

        # 链接预测任务头
        self.icl_module.register_task_head(
            'link_prediction',
            LinkPredictionHead(self.config)
        )

    def forward(self,
                graph: TemporalHeterogeneousGraph,
                target_entity: Tuple[str, int],
                prediction_time: datetime,
                task_config: TaskConfig,
                context_strategy: str = 'mixed',
                num_context: int = 10) -> Dict[str, Any]:
        """
        前向传播

        Args:
            graph: 时序异构图
            target_entity: 目标实体 (节点类型, 节点ID)
            prediction_time: 预测时间
            task_config: 任务配置
            context_strategy: 上下文采样策略
            num_context: 上下文示例数量

        Returns:
            预测结果字典
        """
        # 1. 动态子图采样
        test_subgraph = self.backward_sampler.sample(
            graph,
            target_entity,
            prediction_time,
            num_hops=self.config.num_hops,
            max_nodes=self.config.max_neighbors
        )

        # 2. 上下文采样
        context_examples = self.context_sampler.sample_context(
            graph,
            target_entity,
            prediction_time,
            task_config,
            num_examples=num_context,
            strategy=context_strategy
        )

        # 3. 编码测试子图
        test_embedding = self._encode_subgraph(test_subgraph, target_entity, prediction_time)

        # 4. 编码上下文
        context_embeddings = []
        context_labels = []
        context_entities = []
        context_timestamps = []

        for ctx in context_examples:
            ctx_embedding = self._encode_subgraph(
                ctx.subgraph,
                ctx.entity,
                ctx.timestamp
            )
            context_embeddings.append(ctx_embedding)
            context_labels.append(ctx.label)
            context_entities.append(ctx.entity)
            context_timestamps.append(ctx.timestamp.timestamp())

        # 5. 应用双重上下文机制
        if context_embeddings:
            context_tensor = torch.stack(context_embeddings).unsqueeze(0)
            context_labels_tensor = self._process_labels(context_labels, task_config)
            context_timestamps_tensor = torch.tensor(context_timestamps)

            enhanced_test_embedding, attention_weights = self.dual_context(
                context_tensor,
                context_labels_tensor,
                context_entities,
                test_embedding.unsqueeze(0),
                target_entity,
                context_timestamps_tensor,
                prediction_time.timestamp()
            )
            test_embedding = enhanced_test_embedding.squeeze(0)
        else:
            attention_weights = {}

        # 6. ICL推理
        predictions = self.icl_module(
            context_embeddings,
            context_labels,
            test_embedding,
            task_config.task_type,
            metadata={'task_config': task_config}
        )

        # 7. 后处理
        results = self._postprocess_predictions(predictions, task_config)

        # 添加注意力权重和其他信息
        results['attention_weights'] = attention_weights
        results['num_context_used'] = len(context_examples)

        return results

    def _encode_subgraph(self,
                         subgraph: TemporalHeterogeneousGraph,
                         target_entity: Tuple[str, int],
                         timestamp: datetime) -> torch.Tensor:
        """
        编码子图

        Returns:
            目标实体的最终表示
        """
        # 1. 多模态特征编码
        node_features_dict = {}

        for node_type in subgraph.node_types:
            if node_type in subgraph.node_features:
                # 已有特征
                features = subgraph.node_features[node_type]
            else:
                # 需要编码
                # 这里简化处理，实际应该从原始数据编码
                num_nodes = subgraph.node_counts[node_type]
                features = torch.randn(num_nodes, self.config.hidden_dim)

            node_features_dict[node_type] = features

        # 2. 表内编码
        table_embeddings = self.table_encoder(node_features_dict)

        # 3. 准备RelGT输入
        # 合并所有节点
        all_node_features = []
        all_node_types = []
        node_id_mapping = {}
        offset = 0

        for i, node_type in enumerate(subgraph.node_types):
            num_nodes = subgraph.node_counts[node_type]
            node_features = table_embeddings.get(node_type, node_features_dict[node_type])

            all_node_features.append(node_features)
            all_node_types.extend([i] * num_nodes)

            # 记录ID映射
            node_id_mapping[node_type] = (offset, offset + num_nodes)
            offset += num_nodes

        all_node_features = torch.cat(all_node_features, dim=0)
        all_node_types = torch.tensor(all_node_types)

        # 合并所有边
        all_edges = []
        all_edge_types = []

        for edge_idx, edge_type in enumerate(subgraph.edge_types):
            if edge_type in subgraph.edge_indices:
                source_type, _, target_type = edge_type
                edge_index = subgraph.edge_indices[edge_type]

                # 转换节点ID
                source_offset = node_id_mapping[source_type][0]
                target_offset = node_id_mapping[target_type][0]

                converted_edges = edge_index.clone()
                converted_edges[0] += source_offset
                converted_edges[1] += target_offset

                all_edges.append(converted_edges)
                all_edge_types.extend([edge_idx] * edge_index.shape[1])

        if all_edges:
            all_edge_index = torch.cat(all_edges, dim=1)
            all_edge_types = torch.tensor(all_edge_types)
        else:
            all_edge_index = torch.zeros((2, 0), dtype=torch.long)
            all_edge_types = torch.zeros(0, dtype=torch.long)

        # 4. 多元素Token化
        # 获取跳数信息
        hop_distances = self._get_hop_distances(subgraph, target_entity, node_id_mapping)

        # 获取时间差
        time_diffs = self._get_time_differences(subgraph, timestamp, node_id_mapping)

        # Token化
        tokenized_features = self.tokenizer(
            all_node_features,
            all_node_types,
            hop_distances,
            time_diffs,
            all_edge_index,
            all_edge_types
        )

        # 5. RelGT处理
        node_embeddings = self.relgt(
            tokenized_features,
            all_edge_index,
            all_node_types,
            all_edge_types
        )

        # 6. 提取目标实体的嵌入
        target_node_type, target_node_id = target_entity
        if target_node_type in node_id_mapping:
            start, end = node_id_mapping[target_node_type]
            global_target_id = start + target_node_id

            if 0 <= global_target_id < node_embeddings.shape[0]:
                target_embedding = node_embeddings[global_target_id]
            else:
                # 目标节点不在子图中，使用零向量
                target_embedding = torch.zeros(self.config.hidden_dim)
        else:
            target_embedding = torch.zeros(self.config.hidden_dim)

        return target_embedding

    def _get_hop_distances(self,
                           subgraph: TemporalHeterogeneousGraph,
                           target_entity: Tuple[str, int],
                           node_id_mapping: Dict[str, Tuple[int, int]]) -> torch.Tensor:
        """获取所有节点的跳数距离"""
        total_nodes = sum(subgraph.node_counts.values())
        hop_distances = torch.ones(total_nodes) * 3  # 默认最大跳数

        # 从子图属性中获取跳数信息
        for node_type in subgraph.node_types:
            if hasattr(subgraph, f'{node_type}_hop_distances'):
                hop_tensor = getattr(subgraph, f'{node_type}_hop_distances')
                start, end = node_id_mapping[node_type]
                hop_distances[start:end] = hop_tensor

        return hop_distances

    def _get_time_differences(self,
                              subgraph: TemporalHeterogeneousGraph,
                              prediction_time: datetime,
                              node_id_mapping: Dict[str, Tuple[int, int]]) -> torch.Tensor:
        """计算时间差"""
        total_nodes = sum(subgraph.node_counts.values())
        time_diffs = torch.zeros(total_nodes)

        pred_timestamp = prediction_time.timestamp()

        for node_type in subgraph.node_types:
            if node_type in subgraph.node_timestamps:
                timestamps = subgraph.node_timestamps[node_type]
                start, end = node_id_mapping[node_type]
                time_diffs[start:end] = pred_timestamp - timestamps

        return time_diffs

    def _process_labels(self,
                        labels: List[Any],
                        task_config: TaskConfig) -> torch.Tensor:
        """处理标签数据"""
        if not labels:
            return torch.zeros(0)

        if task_config.task_type == 'classification':
            return torch.tensor(labels, dtype=torch.long).unsqueeze(0)
        elif task_config.task_type == 'regression':
            return torch.tensor(labels, dtype=torch.float).unsqueeze(0)
        elif task_config.task_type == 'multilabel':
            return torch.stack([torch.tensor(l, dtype=torch.float) for l in labels]).unsqueeze(0)
        else:
            return torch.tensor(labels).unsqueeze(0)

    def _postprocess_predictions(self,
                                 predictions: torch.Tensor,
                                 task_config: TaskConfig) -> Dict[str, Any]:
        """后处理预测结果"""
        results = {}

        if task_config.task_type == 'classification':
            # 分类：softmax得到概率
            probs = torch.softmax(predictions, dim=-1)
            results['probabilities'] = probs.detach().cpu().numpy()
            results['predicted_class'] = torch.argmax(probs, dim=-1).item()

        elif task_config.task_type == 'regression':
            # 回归：直接输出
            results['predicted_value'] = predictions.detach().cpu().item()

        elif task_config.task_type == 'multilabel':
            # 多标签：sigmoid得到概率
            probs = torch.sigmoid(predictions)
            results['probabilities'] = probs.detach().cpu().numpy()
            results['predicted_labels'] = (probs > 0.5).int().detach().cpu().numpy()

        elif task_config.task_type == 'link_prediction':
            # 链接预测：返回嵌入
            results['node_embedding'] = predictions.detach().cpu().numpy()

        return results

    def set_task_config(self, task_config: TaskConfig):
        """更新任务配置"""
        # 如果需要更新任务头（如类别数变化）
        if task_config.task_type == 'classification' and task_config.num_classes:
            self.icl_module.register_task_head(
                'classification',
                ClassificationHead(self.config, task_config.num_classes)
            )

        elif task_config.task_type == 'multilabel' and task_config.num_labels:
            # 可以添加多标签任务头
            pass