"""
ICL Module Initialization
"""

from .icl_module import (
    ICLModule,
    ContextEncoder,
    LabelEncoder,
    ICLTransformerLayer,
    ClassificationHead,
    RegressionHead,
    LinkPredictionHead
)

from .dual_context import (
    DualContextMechanism,
    IntraEntityAttention,
    RelevanceScorer
)

__all__ = [
    'ICLModule',
    'ContextEncoder',
    'LabelEncoder',
    'ICLTransformerLayer',
    'ClassificationHead',
    'RegressionHead',
    'LinkPredictionHead',
    'DualContextMechanism',
    'IntraEntityAttention',
    'RelevanceScorer'
]