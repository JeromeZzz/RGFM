"""
KumoRFM Configuration Definitions

"""
from dataclasses import dataclass, field
from typing import Optional, List, Any

@dataclass
class SamplingConfig:
    # Subgraph Sampling
    num_hops: int = 2
    max_neighbors: int = 10
    
    # [Fix] Must be a List, not int. Defaulting to [10, 10] for 2-hop sampling.
    max_neighbors_per_hop: List[int] = field(default_factory=lambda: [10, 10])
    
    strategy: str = 'recent'
    
    # BackwardSampler specifics
    time_decay_factor: float = 0.1  
    sample_recent_first: bool = True
    
    # Context Sampling
    num_context: int = 5
    context_strategy: str = 'mixed'

@dataclass
class KumoRFMConfig:
    # Model Architecture
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout_rate: float = 0.3
    
    # ICL Specifics
    icl_num_layers: int = 2
    icl_num_heads: int = 4
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    
    # Sampling (Nested Config)
    sampling_config: SamplingConfig = field(default_factory=SamplingConfig)

@dataclass
class ExperimentConfig:
    num_epochs: int = 50
    early_stopping_patience: int = 10
    save_dir: str = './relbench_outputs'
    log_dir: str = './relbench_outputs/logs'
    
    # Training fields
    metrics: List[str] = field(default_factory=lambda: ['auc', 'rmse', 'mae'])
    optimizer: str = 'adamw'
    weight_decay: float = 1e-5
    wandb_project: Optional[str] = None
    seed: int = 42
    
    # [NEW] DDP / GPU Settings
    nprocs: int = 1                  # Default number of DDP processes (GPUs)
    
    # [NEW] DataLoader Performance Parameters
    num_workers: int = 0             # 0 means main process only (auto-detect logic handled in script if 0)
    prefetch_factor: Optional[int] = 2 # Number of batches to prefetch per worker
    pin_memory: bool = False          # Use page-locked memory for faster GPU transfer
    persistent_workers: bool = False # Keep workers alive between epochs

@dataclass
class TaskConfig:
    task_type: str  # 'classification', 'regression', 'link_prediction'
    num_classes: int = 2
    metric: str = 'auc' # or 'mae', 'rmse'
    target_column: Optional[str] = None
    
    # Time window fields
    time_window_start: Optional[float] = None
    time_window_end: Optional[float] = None

@dataclass
class ColumnEncoderConfig:
    text_model: str = 'prajjwal1/bert-tiny'
    use_text: bool = True
    use_image: bool = False
    
    # Embedding Dimensions
    numerical_embedding_dim: int = 16
    categorical_embedding_dim: int = 32
    time_embedding_dim: int = 16
    text_embedding_dim: int = 64
    image_embedding_dim: int = 64