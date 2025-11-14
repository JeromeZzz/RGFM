"""
训练模块
说明：训练阶段仅需要 KumoRFMTrainer。微调与 PQL 相关组件在推理/工具阶段再按需导入，
避免在训练路径中引入不必要的依赖（如 PQL dataclass 等）。
"""

from .trainer import KumoRFMTrainer

# 可选导入：仅在需要微调工具时才使用，避免训练阶段引入 PQL 依赖
try:
    from .finetune import KumoRFMFinetuner, finetune_kumorfm, FinetuneDataset  # type: ignore
except Exception:
    KumoRFMFinetuner = None  # type: ignore
    finetune_kumorfm = None  # type: ignore
    FinetuneDataset = None  # type: ignore

__all__ = [
    'KumoRFMTrainer',
    'KumoRFMFinetuner',
    'finetune_kumorfm',
    'FinetuneDataset'
]
