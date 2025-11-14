"""
双重上下文机制
实现实体内和子图间的上下文学习
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import math
from collections import defaultdict

from config.model_config import KumoRFMConfig


class DualContextMechanism(nn.Module):
    """
    双重上下文机制
    结合实体内上下文和子图间上下文
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # 实体内上下文注意力
        self.intra_entity_attention = IntraEntityAttention(config)

        # 子图间上下文注意力
        self.inter_subgraph_attention = InterSubgraphAttention(config)

        # 上下文融合
        self.context_fusion = ContextFusion(config)

        # 相关性评分器
        self.relevance_scorer = RelevanceScorer(config)

    def forward(self,
                context_embeddings: torch.Tensor,
                context_labels: torch.Tensor,
                context_entities: List[Tuple[str, int]],
                test_embedding: torch.Tensor,
                test_entity: Tuple[str, int],
                context_timestamps: Optional[torch.Tensor] = None,
                test_timestamp: Optional[float] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        应用双重上下文机制

        Args:
            context_embeddings: (batch_size, num_context, hidden_dim)
            context_labels: (batch_size, num_context, ...)
            context_entities: 上下文实体列表
            test_embedding: (batch_size, hidden_dim)
            test_entity: 测试实体
            context_timestamps: (batch_size, num_context) 时间戳
            test_timestamp: 测试时间戳

        Returns:
            enhanced_embedding: (batch_size, hidden_dim) 增强的测试嵌入
            attention_weights: 注意力权重字典
        """
        batch_size = test_embedding.size(0)

        # 计算相关性分数
        relevance_scores = self.relevance_scorer(
            context_embeddings,
            context_entities,
            test_embedding,
            test_entity,
            context_timestamps,
            test_timestamp
        )

        # 实体内上下文
        intra_output, intra_weights = self.intra_entity_attention(
            context_embeddings,
            context_labels,
            context_entities,
            test_embedding,
            test_entity,
            relevance_scores
        )

        # 子图间上下文
        inter_output, inter_weights = self.inter_subgraph_attention(
            context_embeddings,
            context_labels,
            test_embedding,
            relevance_scores
        )

        # 融合两种上下文
        enhanced_embedding = self.context_fusion(
            test_embedding,
            intra_output,
            inter_output
        )

        # 收集注意力权重
        attention_weights = {
            'intra_entity': intra_weights,
            'inter_subgraph': inter_weights,
            'relevance_scores': relevance_scores
        }

        return enhanced_embedding, attention_weights


class IntraEntityAttention(nn.Module):
    """
    实体内上下文注意力
    关注目标实体自身和邻域的历史
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # 查询、键、值投影
        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.key_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # 邻域关系编码器
        self.neighborhood_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # 输出投影
        self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.hidden_dim)

    def forward(self,
                context_embeddings: torch.Tensor,
                context_labels: torch.Tensor,
                context_entities: List[Tuple[str, int]],
                test_embedding: torch.Tensor,
                test_entity: Tuple[str, int],
                relevance_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算实体内注意力
        """
        batch_size, num_context, hidden_dim = context_embeddings.shape

        # 识别相关的上下文（同一实体或邻域实体）
        entity_mask = self._get_entity_mask(context_entities, test_entity)

        # 查询、键、值
        query = self.query_proj(test_embedding).unsqueeze(1)  # (batch, 1, hidden)
        keys = self.key_proj(context_embeddings)  # (batch, num_context, hidden)
        values = self.value_proj(context_embeddings)  # (batch, num_context, hidden)

        # 计算注意力分数
        scores = torch.matmul(query, keys.transpose(-2, -1)) * self.scale

        # 应用实体掩码和相关性分数
        scores = scores + relevance_scores.unsqueeze(1)
        if entity_mask is not None:
            # Ensure mask on the same device as scores
            entity_mask = entity_mask.to(scores.device)
            scores = scores.masked_fill(~entity_mask.unsqueeze(0).unsqueeze(1), float('-inf'))

        # 注意力权重
        attention_weights = torch.softmax(scores, dim=-1)

        # 加权聚合
        attended_values = torch.matmul(attention_weights, values)

        # 输出投影
        output = self.output_proj(attended_values.squeeze(1))

        return output, attention_weights.squeeze(1)

    def _get_entity_mask(self,
                         context_entities: List[Tuple[str, int]],
                         test_entity: Tuple[str, int]) -> Optional[torch.Tensor]:
        """
        创建实体掩码
        标识同一实体或相关实体的上下文
        """
        if not context_entities:
            return None

        test_type, test_id = test_entity
        mask = torch.zeros(len(context_entities), dtype=torch.bool)

        for i, (ctx_type, ctx_id) in enumerate(context_entities):
            # 同一实体
            if ctx_type == test_type and ctx_id == test_id:
                mask[i] = True
            # 这里可以添加邻域实体的判断逻辑

        return mask


class InterSubgraphAttention(nn.Module):
    """
    子图间上下文注意力
    捕获不同历史快照间的模式
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads

        # 多头注意力
        self.multihead_attention = nn.MultiheadAttention(
            self.hidden_dim,
            self.num_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )

        # 图级别的模式提取器
        self.pattern_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # 归一化
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self,
                context_embeddings: torch.Tensor,
                context_labels: torch.Tensor,
                test_embedding: torch.Tensor,
                relevance_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算子图间注意力
        """
        batch_size = test_embedding.size(0)

        # 提取子图模式
        context_patterns = self.pattern_extractor(context_embeddings)
        test_pattern = self.pattern_extractor(test_embedding.unsqueeze(1))

        # 归一化
        context_patterns = self.norm(context_patterns)
        test_pattern = self.norm(test_pattern)

        # 多头注意力
        attended_output, attention_weights = self.multihead_attention(
            test_pattern,
            context_patterns,
            context_patterns
        )

        # 提取输出
        output = attended_output.squeeze(1)

        # 平均注意力权重（多头）
        avg_attention_weights = attention_weights.mean(dim=1) if attention_weights.dim() > 2 else attention_weights

        return output, avg_attention_weights


class ContextFusion(nn.Module):
    """
    上下文融合模块
    融合实体内和子图间的上下文信息
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.Sigmoid()
        )

        # 融合投影
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )

        # 残差连接
        self.residual_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self,
                test_embedding: torch.Tensor,
                intra_context: torch.Tensor,
                inter_context: torch.Tensor) -> torch.Tensor:
        """
        融合不同类型的上下文
        """
        # 拼接所有输入
        combined = torch.cat([test_embedding, intra_context, inter_context], dim=-1)

        # 计算门控权重
        gate_weights = self.gate(combined)

        # 门控融合
        gated_intra = gate_weights * intra_context
        gated_inter = (1 - gate_weights) * inter_context

        # 最终融合
        fused = torch.cat([test_embedding, gated_intra, gated_inter], dim=-1)
        output = self.fusion_proj(fused)

        # 残差连接
        output = output + self.residual_weight * test_embedding

        return output


class RelevanceScorer(nn.Module):
    """
    相关性评分器
    计算上下文示例与测试实例的相关性
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # 相似度计算器
        self.similarity_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )

        # 时间衰减
        self.time_decay = nn.Parameter(torch.tensor(0.1))

    def forward(self,
                context_embeddings: torch.Tensor,
                context_entities: List[Tuple[str, int]],
                test_embedding: torch.Tensor,
                test_entity: Tuple[str, int],
                context_timestamps: Optional[torch.Tensor] = None,
                test_timestamp: Optional[float] = None) -> torch.Tensor:
        """
        计算相关性分数

        Returns:
            relevance_scores: (batch_size, num_context)
        """
        batch_size, num_context, _ = context_embeddings.shape

        # 扩展测试嵌入
        test_expanded = test_embedding.unsqueeze(1).expand(-1, num_context, -1)

        # 拼接特征
        combined = torch.cat([context_embeddings, test_expanded], dim=-1)
        combined = combined.view(-1, self.hidden_dim * 2)

        # 计算相似度分数
        similarity_scores = self.similarity_net(combined)
        similarity_scores = similarity_scores.view(batch_size, num_context)

        # 应用时间衰减（如果有时间信息）
        if context_timestamps is not None and test_timestamp is not None:
            time_diffs = torch.abs(context_timestamps - test_timestamp)
            time_weights = torch.exp(-self.time_decay * time_diffs)
            relevance_scores = similarity_scores * time_weights
        else:
            relevance_scores = similarity_scores

        return relevance_scores
