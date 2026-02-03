"""
Column-level encoders used by MultiModalEncoder.
Minimal implementations to keep the model importable and runnable.
"""

from typing import Optional
import torch
import torch.nn as nn


class NumericalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = getattr(config, 'numerical_embedding_dim', 64)
        self.proj = nn.Linear(1, dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Expect x shape [B, 1] or [B]
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.proj(x)


class CategoricalEncoder(nn.Module):
    def __init__(self, config, vocab_size: int):
        super().__init__()
        dim = getattr(config, 'categorical_embedding_dim', 64)
        self.emb = nn.Embedding(max(vocab_size, 1), dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Expect x shape [B] or [B, 1]
        if x.dim() == 2 and x.size(-1) == 1:
            x = x.squeeze(-1)
        return self.emb(x.long())


class MultiCategoricalEncoder(nn.Module):
    """Simple encoder for multiple categorical values per example.
    Encodes each token and averages.
    """
    def __init__(self, config, vocab_size: int):
        super().__init__()
        dim = getattr(config, 'categorical_embedding_dim', 64)
        self.emb = nn.Embedding(max(vocab_size, 1), dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        emb = self.emb(x.long())
        return emb.mean(dim=1)


class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = getattr(config, 'text_embedding_dim', 128)
        # Minimal text encoder: a small projection; assumes input already numeric (e.g., pooled tokens)
        self.proj = nn.Linear(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept [B] or [B,1]; in real use this should be a transformer or sentence model.
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return self.proj(x.float())


class TimeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = getattr(config, 'time_embedding_dim', 64)
        self.proj = nn.Linear(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        # Simple linear projection of timestamps
        return self.proj(x.float())


class EmbeddingEncoder(nn.Module):
    def __init__(self, config, input_dim: int):
        super().__init__()
        proj_dim = getattr(config, 'embedding_projection_dim', None) or input_dim
        self.proj = nn.Linear(input_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.proj(x.float())
