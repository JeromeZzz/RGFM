"""
多模态编码器
支持多种数据类型的统一编码
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

from config.model_config import ColumnEncoderConfig
from .column_encoder import (
    NumericalEncoder,
    CategoricalEncoder,
    TextEncoder,
    TimeEncoder,
    EmbeddingEncoder
)


class MultiModalEncoder(nn.Module):
    """
    多模态编码器
    将不同类型的列数据编码为统一的向量表示
    """

    def __init__(self, config: ColumnEncoderConfig, output_dim: int):
        super().__init__()
        self.config = config
        self.output_dim = output_dim

        # 各类型编码器
        self.encoders = nn.ModuleDict()

        # 列类型到编码器的映射
        self.column_encoders = {}

        # 输出投影层
        self.projection_layers = nn.ModuleDict()

    def register_column(self,
                        column_name: str,
                        column_type: str,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        注册列及其编码器

        Args:
            column_name: 列名
            column_type: 列类型 ('numerical', 'categorical', 'text', 'time', 'embedding')
            metadata: 列的元数据（如词汇表大小、嵌入维度等）
        """
        encoder_key = f"{column_type}_{column_name}"
        # Torch module names cannot contain '.'; sanitize
        encoder_key = encoder_key.replace('.', '_')

        if column_type == 'numerical':
            if encoder_key not in self.encoders:
                encoder = NumericalEncoder(self.config)
                self.encoders[encoder_key] = encoder

                # 投影到输出维度
                proj_dim = self.config.numerical_embedding_dim
                self.projection_layers[encoder_key] = nn.Linear(proj_dim, self.output_dim)

        elif column_type == 'categorical':
            if encoder_key not in self.encoders:
                vocab_size = metadata.get('vocab_size', 100) if metadata else 100
                encoder = CategoricalEncoder(self.config, vocab_size)
                self.encoders[encoder_key] = encoder

                # 投影到输出维度
                proj_dim = self.config.categorical_embedding_dim
                self.projection_layers[encoder_key] = nn.Linear(proj_dim, self.output_dim)

        elif column_type == 'text':
            if encoder_key not in self.encoders:
                encoder = TextEncoder(self.config)
                self.encoders[encoder_key] = encoder

                # 投影到输出维度
                proj_dim = self.config.text_embedding_dim
                self.projection_layers[encoder_key] = nn.Linear(proj_dim, self.output_dim)

        elif column_type == 'time':
            if encoder_key not in self.encoders:
                encoder = TimeEncoder(self.config)
                self.encoders[encoder_key] = encoder

                # 投影到输出维度
                proj_dim = self.config.time_embedding_dim
                self.projection_layers[encoder_key] = nn.Linear(proj_dim, self.output_dim)

        elif column_type == 'embedding':
            if encoder_key not in self.encoders:
                input_dim = metadata.get('embedding_dim', 128) if metadata else 128
                encoder = EmbeddingEncoder(self.config, input_dim)
                self.encoders[encoder_key] = encoder

                # 投影到输出维度
                proj_dim = self.config.embedding_projection_dim or input_dim
                self.projection_layers[encoder_key] = nn.Linear(proj_dim, self.output_dim)

        else:
            raise ValueError(f"Unsupported column type: {column_type}")

        self.column_encoders[column_name] = (column_type, encoder_key)

    def forward(self,
                column_data: Dict[str, torch.Tensor],
                column_masks: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        编码多列数据

        Args:
            column_data: {column_name: data_tensor}
            column_masks: {column_name: mask_tensor} 标识有效值

        Returns:
            {column_name: encoded_tensor} 每列的编码结果
        """
        encoded = {}

        for column_name, data in column_data.items():
            if column_name not in self.column_encoders:
                continue

            column_type, encoder_key = self.column_encoders[column_name]
            encoder = self.encoders[encoder_key]
            projection = self.projection_layers[encoder_key]

            # 获取掩码
            mask = column_masks.get(column_name) if column_masks else None

            # 编码
            if column_type in ['numerical', 'categorical']:
                column_encoding = encoder(data, mask)
            elif column_type == 'text':
                # 文本编码器需要字符串输入
                # 这里假设data是token_ids
                column_encoding = encoder(data)
            elif column_type == 'time':
                column_encoding = encoder(data)
            elif column_type == 'embedding':
                column_encoding = encoder(data)
            else:
                column_encoding = data

            # 投影到输出维度
            encoded[column_name] = projection(column_encoding)

        return encoded

    def encode_table_row(self,
                         row_data: Dict[str, Any],
                         table_schema: Dict[str, str]) -> torch.Tensor:
        """
        编码表的一行数据

        Args:
            row_data: {column_name: value}
            table_schema: {column_name: column_type}

        Returns:
            行的编码向量 (num_columns, output_dim)
        """
        column_encodings = []

        for column_name, column_type in table_schema.items():
            if column_name not in row_data:
                # 缺失值处理
                encoding = torch.zeros(self.output_dim)
            else:
                value = row_data[column_name]

                # 转换为张量
                if column_type == 'numerical':
                    tensor_value = torch.tensor([value], dtype=torch.float)
                elif column_type == 'categorical':
                    tensor_value = torch.tensor([value], dtype=torch.long)
                elif column_type == 'time':
                    # 假设时间值是timestamp
                    tensor_value = torch.tensor([value], dtype=torch.float)
                elif column_type == 'embedding':
                    tensor_value = torch.tensor(value, dtype=torch.float)
                else:
                    tensor_value = value

                # 编码
                column_data = {column_name: tensor_value.unsqueeze(0)}
                encoded = self.forward(column_data)

                if column_name in encoded:
                    encoding = encoded[column_name].squeeze(0)
                else:
                    encoding = torch.zeros(self.output_dim)

            column_encodings.append(encoding)

        return torch.stack(column_encodings)

    def get_column_dim(self, column_name: str) -> int:
        """获取列编码后的维度"""
        return self.output_dim

    def reset_parameters(self) -> None:
        """重置参数"""
        for encoder in self.encoders.values():
            if hasattr(encoder, 'reset_parameters'):
                encoder.reset_parameters()

        for proj in self.projection_layers.values():
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
