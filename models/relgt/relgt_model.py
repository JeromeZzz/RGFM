"""
RelGT wrapper integrated with the official local module.

This wrapper wires KumoRFM to the RelGT LocalModule by treating the
tokenized multi-element representation as a sequence of tokens where
the first token corresponds to the target node and the remaining tokens
are its neighborhood. The LocalModule aggregates neighbors and returns
the target embedding.
"""

from typing import Optional
import torch
import torch.nn as nn


class RelGTWrapper(nn.Module):
    """Wrapper that uses RelGT's LocalModule for subgraph encoding.

    Expects tokenized_features for a single subgraph (sequence of nodes)
    where row-0 is the target node token and the rest are neighbors.
    Returns a single embedding vector for the target node.
    """

    def __init__(self, config, num_node_types: int):
        super().__init__()
        from .relgt.local_module import LocalModule  # import real module

        self.hidden = getattr(config, 'hidden_dim', 256)
        self.num_heads = getattr(config, 'num_heads', 8)
        self.num_layers = getattr(config, 'num_layers', 2)
        self.dropout = getattr(config, 'dropout_rate', 0.0)
        self.LocalModule = LocalModule
        # LocalModule requires seq_len at construction; build per-call.

    def forward(
        self,
        tokenized_features: torch.Tensor,  # [L, D] or [1, L, D]
        edge_index: Optional[torch.Tensor] = None,
        node_types: Optional[torch.Tensor] = None,
        edge_types: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokenized_features.dim() == 2:
            tokenized_features = tokenized_features.unsqueeze(0)
        elif tokenized_features.dim() != 3:
            raise ValueError("tokenized_features must be [L, D] or [B, L, D]")

        B, L, D = tokenized_features.shape
        if B != 1:
            # Current KumoRFM encodes single subgraph per call
            raise ValueError("RelGTWrapper expects batch size 1")

        device = tokenized_features.device
        # Instantiate LocalModule for this sequence length
        local = self.LocalModule(
            seq_len=L,
            input_dim=D,
            n_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=D,
            dropout_rate=self.dropout,
            attention_dropout_rate=0.0,
        ).to(device)

        # If batch/sequence is too small, BatchNorm1d in LocalModule fails in train mode.
        # For tiny subgraphs (B < 2 or L < 2), run LocalModule in eval mode.
        if (tokenized_features.size(0) < 2) or (L < 2):
            local.eval()
        else:
            local.train(self.training)

        # Forward through local transformer; returns [D] for batch=1
        out = local(tokenized_features, pretrain_token=False)
        if out.dim() > 1:
            out = out.squeeze(0)
        return out
