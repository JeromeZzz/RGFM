"""
Dual Context Mechanism

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math

from config.model_config import KumoRFMConfig

class DualContextMechanism(nn.Module):
    def __init__(self, config: KumoRFMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.intra_entity_attention = IntraEntityAttention(config)
        self.relevance_scorer = RelevanceScorer(config)
        self.fusion_gate = nn.Sequential(nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.Sigmoid())
        self.output_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, 
                ctx_emb: torch.Tensor,       # [Batch, Num_Ctx, Dim]
                ctx_labels: torch.Tensor,    # [Batch, Num_Ctx, 1]
                test_emb: torch.Tensor,      # [Batch, 1, Dim]
                ctx_ts: torch.Tensor,        # [Batch, Num_Ctx]
                test_ts: torch.Tensor,       # [Batch, 1]
                intra_mask: torch.Tensor     # [Batch, Num_Ctx] (Pre-computed mask)
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. Inter-Subgraph Relevance [Batch, Num_Ctx]
        rel_scores = self.relevance_scorer(ctx_emb, test_emb)
        
        # 2. Intra-Entity Attention [Batch, Dim]
        # intra_mask is passed in (Computed efficiently outside)
        intra_context = self.intra_entity_attention(ctx_emb, ctx_ts, test_ts, mask=intra_mask)
        
        # 3. Inter-Subgraph Aggregation [Batch, Dim]
        inter_weights = F.softmax(rel_scores, dim=1).unsqueeze(-1)
        inter_context = torch.sum(ctx_emb * inter_weights, dim=1)
        
        # 4. Fusion
        combined = torch.cat([intra_context, inter_context], dim=-1)
        gate = self.fusion_gate(combined)
        enhanced_context = gate * intra_context + (1 - gate) * inter_context
        
        # 5. Residual
        output = self.norm(test_emb.squeeze(1) + self.output_proj(enhanced_context))
        
        return output, rel_scores

class IntraEntityAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.time_decay = nn.Parameter(torch.tensor([0.1]))
        
    def forward(self, ctx_emb, ctx_ts, test_ts, mask=None):
        # Fully Vectorized: [Batch, 1] - [Batch, N] -> [Batch, N]
        diff = test_ts - ctx_ts
        diff = torch.clamp(diff, min=0)
        decay = torch.exp(-torch.abs(self.time_decay) * diff)
        
        if mask is not None:
            decay = decay * mask
            
        weight_sum = decay.sum(dim=1, keepdim=True) + 1e-9
        norm_weights = (decay / weight_sum).unsqueeze(-1) # [Batch, N, 1]
        return torch.sum(ctx_emb * norm_weights, dim=1)

class RelevanceScorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.scale = math.sqrt(config.hidden_dim)
        
    def forward(self, ctx_emb, test_emb):
        # [Batch, N, D]
        keys = self.key_proj(ctx_emb)
        # [Batch, 1, D]
        query = self.query_proj(test_emb)
        # BMM: [B, 1, D] @ [B, D, N] -> [B, 1, N]
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        return scores.squeeze(1)