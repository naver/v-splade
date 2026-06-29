# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
Query encoder strategies.

V_SPLADE quality variant uses an inference-free *Learned Sparse Retriever*
query encoder (``li_lsr``): a frozen copy of the backbone word embeddings,
optionally adapted with a low-rank LoRA, followed by a 1-dim projection
and an activation. At inference time the projection becomes a
``vocab_size``-entry lookup table, so encoding a query is just an
``Embedding(input_ids)`` — no transformer forward.

The simpler ``bow`` encoder turns the query into a binary token-presence
indicator over the vocab. Useful for smoke tests and the cheapest
inference-time setting.
"""

import torch
import torch.nn as nn
from enum import Enum
from typing import Optional


class QueryEncoderType(Enum):
    BOW = "bow"
    LI_LSR = "li_lsr"


class InferenceFreeQueryEncoder(nn.Module):
    """Li-LSR style inference-free sparse query encoder.

    Training:  token_id → embedding (frozen + LoRA) → projection(D→1) →
               activation → scatter into a vocab-dim vector.
    Inference: token_id → precomputed lookup table (no forward pass).
    """

    def __init__(
        self,
        embed_weight: torch.Tensor,
        vocab_size: int,
        hidden_size: int,
        lora_r: int = 0,
        activation: str = "relu",
    ):
        super().__init__()
        assert activation in ("relu", "softplus"), f"Unknown activation: {activation}"
        self.activation = activation

        # Frozen copy of backbone word embeddings.
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.embeddings.weight.data.copy_(embed_weight.detach())
        self.embeddings.weight.requires_grad = False

        # Optional LoRA on the embedding lookup.
        self.use_lora = lora_r > 0
        if self.use_lora:
            self.lora_A = nn.Parameter(torch.randn(vocab_size, lora_r) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(lora_r, hidden_size))

        # Projection: hidden_size → 1 (fully tuned).
        self.projection = nn.Linear(hidden_size, 1, bias=True)

        self.vocab_size = vocab_size
        self._lookup_table: Optional[torch.Tensor] = None

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return torch.log1p(torch.relu(x))
        return torch.nn.functional.softplus(x)

    def _valid_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids < self.vocab_size

    def _get_embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        safe_ids = input_ids.clamp(max=self.vocab_size - 1)
        e = self.embeddings(safe_ids)
        if self.use_lora:
            e = e + self.lora_A[safe_ids] @ self.lora_B
        return e

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward. Returns (B, vocab_size) sparse vector."""
        valid = self._valid_mask(input_ids)
        mask = attention_mask.float() * valid.float()
        scores = self.projection(self._get_embed(input_ids)).squeeze(-1)
        scores = self._activate(scores) * mask
        B = input_ids.size(0)
        safe_ids = input_ids.clamp(max=self.vocab_size - 1)
        out = torch.zeros(B, self.vocab_size, device=input_ids.device, dtype=scores.dtype)
        out.scatter_add_(1, safe_ids, scores)
        return out

    @torch.no_grad()
    def build_lookup_table(self) -> torch.Tensor:
        device = self.embeddings.weight.device
        all_ids = torch.arange(self.vocab_size, device=device)
        e = self._get_embed(all_ids.unsqueeze(0))
        s = self.projection(e).squeeze(-1).squeeze(0)
        self._lookup_table = self._activate(s)
        return self._lookup_table

    def encode_with_lookup(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inference: pure lookup, no embedding/projection forward."""
        if self._lookup_table is None:
            self.build_lookup_table()
        valid = self._valid_mask(input_ids)
        mask = attention_mask.float() * valid.float()
        safe_ids = input_ids.clamp(max=self.vocab_size - 1)
        scores = self._lookup_table.to(input_ids.device)[safe_ids] * mask
        B = input_ids.size(0)
        out = torch.zeros(B, self.vocab_size, device=input_ids.device, dtype=scores.dtype)
        out.scatter_add_(1, safe_ids, scores)
        return out


class BOWQueryEncoder(nn.Module):
    """Bag-of-Words: binary token indicator over the vocab."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, _ = input_ids.shape
        device = input_ids.device
        bow = torch.zeros(B, self.vocab_size, device=device, dtype=torch.bfloat16)
        safe_ids = input_ids.clamp(0, self.vocab_size - 1) * attention_mask.long()
        bow.scatter_(1, safe_ids, 1.0)
        bow[:, 0] = 0.0  # clear padding token (id=0) accumulation
        return bow


def build_query_encoder(
    query_encoder_type: str,
    vocab_size: int,
    hidden_size: int = 768,
    embed_weight: Optional[torch.Tensor] = None,
    lora_r: int = 0,
    activation: str = "relu",
    **kwargs,
) -> nn.Module:
    if query_encoder_type == "bow":
        return BOWQueryEncoder(vocab_size)
    if query_encoder_type == "li_lsr":
        if embed_weight is None:
            raise ValueError("li_lsr requires embed_weight from the encoder")
        return InferenceFreeQueryEncoder(
            embed_weight, vocab_size, hidden_size,
            lora_r=lora_r, activation=activation,
        )
    raise ValueError(
        f"Unknown query_encoder_type: {query_encoder_type}. Choose: bow | li_lsr"
    )
