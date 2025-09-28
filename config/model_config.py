"""
KumoRFM 模型配置
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class KumoRFMConfig:
    """KumoRFM主配置"""
    # 模型维度
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8

    # 采样参数
    max_neighbors: int = 300  # K: 局部邻居数量
    num_hops: int = 2  # k跳邻居
    context_window_size: int = 10  # 上下文示例数量

    # RelGT参数
    num_global_tokens: int = 4096  # B: 全局质心数量
    use_global_attention: bool = True

    # 编码器参数
    numerical_embedding_dim: int = 64
    categorical_embedding_dim: int = 64
    text_embedding_dim: int = 768
    time_embedding_dim: int = 64

    # 训练参数
    batch_size: int = 256
    learning_rate: float = 1e-4
    dropout_rate: float = 0.3
    gradient_clip: float = 1.0

    # ICL参数
    icl_num_layers: int = 2
    icl_num_heads: int = 8

    # 任务相关
    task_types: List[str] = None  # ['classification', 'regression', 'link_prediction']

    def __post_init__(self):
        if self.task_types is None:
            self.task_types = ['classification', 'regression', 'link_prediction']


@dataclass
class SamplingConfig:
    """采样配置"""
    strategy: str = 'temporal_importance'  # 'random', 'temporal_importance', 'structure_aware'
    max_neighbors_per_hop: List[int] = None  # 每跳的最大邻居数
    time_decay_factor: float = 0.9  # 时间衰减因子

    def __post_init__(self):
        if self.max_neighbors_per_hop is None:
            self.max_neighbors_per_hop = [150, 100, 50]  # 递减的邻居数


@dataclass
class ColumnEncoderConfig:
    """列编码器配置"""
    # 数值型
    numerical_embedding_dim: int = 64
    use_numerical_normalization: bool = True

    # 类别型
    categorical_embedding_dim: int = 64
    max_categorical_size: int = 10000
    unknown_token_id: int = 0

    # 文本型
    text_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    text_max_length: int = 512
    text_pooling: str = 'mean'  # 'mean', 'max', 'cls'

    # 时间型
    time_embedding_dim: int = 64
    time_encoding_type: str = 'sinusoidal'  # 'sinusoidal', 'learned'

    # 嵌入型
    embedding_projection_dim: Optional[int] = None  # None表示不投影


@dataclass
class TaskConfig:
    """任务配置"""
    task_type: str  # 'classification', 'regression', 'link_prediction', 'multilabel'
    num_classes: Optional[int] = None  # 分类任务的类别数
    num_labels: Optional[int] = None  # 多标签任务的标签数
    target_column: str = None  # 目标列名
    aggregation: str = 'mean'  # 聚合方式

    # 时间窗口
    time_window_start: Optional[int] = None  # 相对于预测时间的起始偏移（天）
    time_window_end: Optional[int] = None  # 相对于预测时间的结束偏移（天）

    # 链接预测特定参数
    negative_sampling_ratio: int = 5  # 负样本比例
    link_prediction_mode: str = 'transductive'  # 'transductive', 'inductive'


@dataclass
class ExperimentConfig:
    """实验配置"""
    seed: int = 42
    num_epochs: int = 100
    early_stopping_patience: int = 10
    save_dir: str = './checkpoints'
    log_dir: str = './logs'

    # 数据划分
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 评估指标
    metrics: List[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['accuracy', 'f1', 'auc', 'mae', 'rmse']