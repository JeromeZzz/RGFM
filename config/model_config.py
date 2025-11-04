"""
KumoRFM 模型配置
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class KumoRFMConfig:
    """KumoRFM Main Configuration"""
    # Model dimensions
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8

    # Sampling parameters
    max_neighbors: int = 300  # K: local neighbor count
    num_hops: int = 2  # k-hop neighbors
    context_window_size: int = 10  # context example count

    # RelGT parameters
    num_global_tokens: int = 4096  # B: global centroid count
    use_global_attention: bool = True

    # Encoder parameters
    numerical_embedding_dim: int = 64
    categorical_embedding_dim: int = 64
    text_embedding_dim: int = 768
    time_embedding_dim: int = 64

    # Training parameters
    batch_size: int = 256
    learning_rate: float = 1e-4
    dropout_rate: float = 0.3
    gradient_clip: float = 1.0

    # ICL parameters
    icl_num_layers: int = 2
    icl_num_heads: int = 8

    # Task related
    task_types: List[str] = None  # ['classification', 'regression', 'link_prediction']

    def __post_init__(self):
        if self.task_types is None:
            self.task_types = ['classification', 'regression', 'link_prediction']


@dataclass
class SamplingConfig:
    """Sampling Configuration"""
    strategy: str = 'temporal_importance'  # 'random', 'temporal_importance', 'structure_aware'
    max_neighbors_per_hop: List[int] = None  # max neighbors per hop
    time_decay_factor: float = 0.9  # time decay factor

    def __post_init__(self):
        if self.max_neighbors_per_hop is None:
            self.max_neighbors_per_hop = [150, 100, 50]  # decreasing neighbor count


@dataclass
class ColumnEncoderConfig:
    """Column Encoder Configuration"""
    # Numerical
    numerical_embedding_dim: int = 64
    use_numerical_normalization: bool = True

    # Categorical
    categorical_embedding_dim: int = 64
    max_categorical_size: int = 10000
    unknown_token_id: int = 0

    # Text
    text_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    text_max_length: int = 512
    text_pooling: str = 'mean'  # 'mean', 'max', 'cls'

    # Time
    time_embedding_dim: int = 64
    time_encoding_type: str = 'sinusoidal'  # 'sinusoidal', 'learned'

    # Embedding
    embedding_projection_dim: Optional[int] = None  # None means no projection


@dataclass
class TaskConfig:
    """Task Configuration"""
    task_type: str  # 'classification', 'regression', 'link_prediction', 'multilabel'
    num_classes: Optional[int] = None  # number of classes for classification
    num_labels: Optional[int] = None  # number of labels for multilabel task
    target_column: str = None  # target column name
    aggregation: str = 'mean'  # aggregation method

    # Time window
    time_window_start: Optional[int] = None  # start offset relative to prediction time (days)
    time_window_end: Optional[int] = None  # end offset relative to prediction time (days)

    # Link prediction specific parameters
    negative_sampling_ratio: int = 5  # negative sampling ratio
    link_prediction_mode: str = 'transductive'  # 'transductive', 'inductive'


@dataclass
class ExperimentConfig:
    """Experiment Configuration"""
    seed: int = 42
    num_epochs: int = 100
    early_stopping_patience: int = 10
    save_dir: str = './checkpoints'
    log_dir: str = './logs'

    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Evaluation metrics
    metrics: List[str] = None

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['accuracy', 'f1', 'auc', 'mae', 'rmse']