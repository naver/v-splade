# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
Output head — projects encoder representations into the retrieval space.

V_SPLADE uses SparseHead: the encoder's LM head is reused to project
hidden states to vocab-dim sparse weights via log1p(relu(.)).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from typing import Tuple


class HeadType(Enum):
    DENSE = "dense"
    SPARSE = "sparse"


class DenseHead(nn.Module):
    """L2-normalized dense embedding."""

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return F.normalize(pooled, dim=-1)


class SparseHead(nn.Module):
    """Hidden states → LM head → log1p(relu) → sparse vocab-dim weights.

    Returns (h_raw, w_sparse).  The LM head is held as a *reference* (not a
    registered sub-module) to avoid duplicated state-dict keys, since the
    encoder already owns the same module.
    """

    def __init__(self, lm_head: nn.Module, hidden_size: int):
        super().__init__()
        object.__setattr__(self, "_lm_head_ref", lm_head)
        self.scale = hidden_size ** -0.25

    @property
    def lm_head(self):
        return self._lm_head_ref

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self._lm_head_ref(hidden_states) * self.scale
        w = torch.log1p(torch.relu(h))
        return h, w


def build_head(head_type: str, lm_head=None, hidden_size: int = 768, **kwargs) -> nn.Module:
    if head_type == "dense":
        return DenseHead()
    if head_type == "sparse":
        if lm_head is None:
            raise ValueError("sparse head requires lm_head")
        return SparseHead(lm_head, hidden_size)
    raise ValueError(f"Unknown head_type: {head_type}. Choose: dense | sparse")
