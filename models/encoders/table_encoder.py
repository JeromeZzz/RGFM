"""
表内Transformer编码器
处理表内的行级聚合
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math

from config.model_config import KumoRFMConfig


class TableTransformer(nn.Module):
    """
    表内Transformer
    对每个表独立应用，学习列间依赖关系
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.num_layers = 2  # 表内处理使用较少的层数

        # 位置编码（用于列位置）
        self.column_position_encoding = nn.Parameter(
            torch.randn(1, 100, self.hidden_dim)  # 支持最多100列
        )

        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TableTransformerLayer(config)
            for _ in range(self.num_layers)
        ])

        # 行聚合
        self.row_aggregation = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        # 可学习的聚合权重
        self.aggregation_weights = nn.Parameter(torch.ones(1))

    def forward(self,
                cell_embeddings: torch.Tensor,
                column_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        处理表的单元格嵌入

        Args:
            cell_embeddings: (batch_size, num_rows, num_cols, hidden_dim)
            column_mask: (batch_size, num_cols) 标识有效列

        Returns:
            row_embeddings: (batch_size, num_rows, hidden_dim)
        """
        batch_size, num_rows, num_cols, hidden_dim = cell_embeddings.shape

        # 添加列位置编码
        pos_encoding = self.column_position_encoding[:, :num_cols, :]
        cell_embeddings = cell_embeddings + pos_encoding.unsqueeze(1)

        # 重塑为 (batch_size * num_rows, num_cols, hidden_dim)
        cell_embeddings = cell_embeddings.view(-1, num_cols, hidden_dim)

        # 应用Transformer层
        x = cell_embeddings
        for layer in self.transformer_layers:
            x = layer(x, column_mask)

        # 聚合列到行表示
        if column_mask is not None:
            # 扩展掩码
            column_mask_expanded = column_mask.unsqueeze(1).expand(
                batch_size, num_rows, num_cols
            ).reshape(-1, num_cols)

            # 加权聚合
            weights = torch.softmax(
                self.aggregation_weights * column_mask_expanded.float(),
                dim=1
            )
            row_embeddings = (x * weights.unsqueeze(-1)).sum(dim=1)
        else:
            # 简单平均
            row_embeddings = x.mean(dim=1)

        # 应用行聚合网络
        row_embeddings = self.row_aggregation(row_embeddings)

        # 重塑回原始形状
        row_embeddings = row_embeddings.view(batch_size, num_rows, hidden_dim)

        return row_embeddings


class TableTransformerLayer(nn.Module):
    """
    单个表Transformer层
    """

    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.dropout_rate = config.dropout_rate

        # 多头注意力
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

    def forward(self,
                x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch_size, seq_len, hidden_dim)
        mask: (batch_size, seq_len)
        """
        # 自注意力
        if mask is not None:
            # 转换为注意力掩码
            attn_mask = mask.float()
            attn_mask = attn_mask.masked_fill(mask == 0, float('-inf'))
            attn_mask = attn_mask.masked_fill(mask == 1, float(0.0))
        else:
            attn_mask = None

        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attention(x, x, x, key_padding_mask=attn_mask)
        x = residual + self.dropout(attn_output)

        # 前馈网络
        residual = x
        x = self.norm2(x)
        ff_output = self.feed_forward(x)
        x = residual + self.dropout(ff_output)

        return x


class MultiTableEncoder(nn.Module):
    """
    多表编码器
    管理多个表的独立编码
    """

    def __init__(self, config: KumoRFMConfig, table_names: List[str]):
        super().__init__()
        self.config = config
        self.table_names = table_names

        # 为每个表创建独立的编码器
        self.table_encoders = nn.ModuleDict({
            table_name: TableTransformer(config)
            for table_name in table_names
        })

        # 表类型嵌入
        self.table_type_embeddings = nn.Embedding(
            len(table_names),
            config.hidden_dim
        )

        # 表名到索引的映射
        self.table_to_idx = {name: idx for idx, name in enumerate(table_names)}

    def forward(self,
                table_data: Dict[str, torch.Tensor],
                table_masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        编码多个表的数据

        Args:
            table_data: {table_name: cell_embeddings}
            table_masks: {table_name: column_mask}

        Returns:
            {table_name: row_embeddings}
        """
        encoded_tables = {}

        for table_name, cell_embeddings in table_data.items():
            if table_name not in self.table_encoders:
                continue

            # 获取表编码器
            encoder = self.table_encoders[table_name]

            # 获取列掩码
            column_mask = table_masks.get(table_name) if table_masks else None

            # 编码表
            row_embeddings = encoder(cell_embeddings, column_mask)

            # 添加表类型嵌入
            table_idx = self.table_to_idx[table_name]
            table_type_emb = self.table_type_embeddings(
                torch.tensor(table_idx, device=cell_embeddings.device)
            )
            row_embeddings = row_embeddings + table_type_emb.unsqueeze(0).unsqueeze(0)

            encoded_tables[table_name] = row_embeddings

        return encoded_tables