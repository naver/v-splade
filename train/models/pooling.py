# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
Pooling layer: (B, L, D) → (B, D).

V_SPLADE uses MAX pooling (SPLADE-style position-wise max over the
sequence). Other strategies are provided for completeness.
"""

import torch
import torch.nn as nn
from enum import Enum


class PoolingType(Enum):
    MAX = "max"
    MEAN = "mean"
    EOS = "eos"
    CLS = "cls"


class Pooling(nn.Module):
    """Pool a sequence of hidden states to a single vector."""

    def __init__(self, pooling_type: str):
        super().__init__()
        self.pooling_type = PoolingType(pooling_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.pooling_type == PoolingType.MAX:
            return self._max_pool(hidden_states, attention_mask)
        if self.pooling_type == PoolingType.MEAN:
            return self._mean_pool(hidden_states, attention_mask)
        if self.pooling_type == PoolingType.EOS:
            return self._eos_pool(hidden_states, attention_mask)
        if self.pooling_type == PoolingType.CLS:
            return hidden_states[:, 0]
        raise ValueError(f"Unknown pooling type: {self.pooling_type}")

    @staticmethod
    def _max_pool(hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        h_masked = hidden_states * mask + (~mask.bool()) * (-1e9)
        return h_masked.max(dim=1).values

    @staticmethod
    def _mean_pool(hidden_states, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        seq_len = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (hidden_states * mask).sum(dim=1) / seq_len.squeeze(1)

    @staticmethod
    def _eos_pool(hidden_states, attention_mask):
        seq_lengths = (attention_mask.sum(dim=1) - 1).clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, seq_lengths]
