# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
UnifiedRetriever -- V_SPLADE core model.

Combines:
    VBert encoder + Sparse head (SPLADE) + BOW query encoder + Losses

Training objective (per forward call):
    L_total = NCE(q, p, hn) + lambda_p * FLOPS(p) + lambda_cap * CaptionPushUp

The model expects a passage (image) and an optional caption; the query is
encoded as a Bag-of-Words vocabulary indicator. The sparse passage
representation is supervised by:
  - NCE in-batch ranking against the query (and hard negatives, if any)
  - FLOPS regularization to encourage sparsity
  - Caption-gated token supervision: pull passage activations toward overlapping
    caption-vocabulary tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from models.encoder import build_encoder
from models.pooling import Pooling
from models.head import SparseHead
from models.query_encoder import (
    build_query_encoder,
    BOWQueryEncoder,
    InferenceFreeQueryEncoder,
)
from models.losses import FLOPSLoss, NCELoss, CaptionPushUpLoss, ZipfianPushUpLoss  # noqa: F401


def compute_logits(q, p, hn, temperature):
    """Compute in-batch negative logits and labels.

    Args:
        q:  (B, V) query sparse reps
        p:  (B, V) passage sparse reps
        hn: (B*num_hn, V) hard-negative passage reps, or None
        temperature: scaling factor for the inner product

    Returns:
        logits: (B, B [+ B*num_hn]) score matrix
        labels: (B,) target index for cross-entropy
    """
    B = q.size(0)
    device = q.device
    all_p = torch.cat([p, hn], dim=0) if hn is not None else p
    logits = (q @ all_p.t()) / temperature
    logits = logits.clamp(min=-100, max=100)  # prevent exp() overflow in cross_entropy
    labels = torch.arange(B, device=device)
    return logits, labels


# Default pooling for V_SPLADE (single backbone): max over sequence positions.
DEFAULT_POOLING = "max"


@dataclass
class RetrievalOutput:
    """Container for forward() outputs."""
    loss: Optional[torch.Tensor] = None
    rank_loss: Optional[torch.Tensor] = None
    reg_loss: Optional[torch.Tensor] = None        # total reg (passage FLOPS + caption FLOPS)
    reg_loss_p: Optional[torch.Tensor] = None      # passage FLOPS only
    reg_loss_cap: Optional[torch.Tensor] = None    # caption FLOPS only
    cap_loss: Optional[torch.Tensor] = None        # caption-gated token supervision loss
    cap_sparse_rank_loss: Optional[torch.Tensor] = None  # caption-gated ranking loss
    # Sparsity / retrieval-cost diagnostics (no_grad, mean over batch).
    train_p_nnz: Optional[torch.Tensor] = None     # avg nonzero per passage
    train_p_max: Optional[torch.Tensor] = None     # avg max|p| per passage
    train_flops_qd: Optional[torch.Tensor] = None  # sum_v mean|q|_v * mean|p|_v
    train_cap_bow_killed_pct: Optional[torch.Tensor] = None  # % bow-active tokens zeroed by model
    query_reps: Optional[torch.Tensor] = None
    passage_reps: Optional[torch.Tensor] = None
    hard_neg_reps: Optional[torch.Tensor] = None
    debug_tokens: Optional[dict] = None


class UnifiedRetriever(nn.Module):
    """V_SPLADE retriever: VBert encoder + SPLADE sparse head + (BOW | Li-LSR) query encoder."""

    def __init__(
        self,
        encoder_type: str = "vbert",
        pooling_type: str = None,
        head_type: str = "sparse",
        query_encoder_type: str = "bow",
        # Model paths
        model_name: str = "",
        lm_head_model: str = None,
        # Training config
        temperature: float = 1.0,
        # Caption-gated token supervision
        cap_weight: float = 0.0,
        cap_sparse_rank_weight: float = 0.0,
        cap_loss_mode: str = "logsigmoid_h",
        overlap_type: str = "passage_mean",
        p_mean_alpha: float = 1.0,
        use_zipfian_pushup: bool = False,
        push_focus_tau: float = 1.0,
        # Regularization weights
        reg_weight_p: float = 0.0,
        reg_weight_cap: float = 0.0,
        # SPLADE pooling
        splade_pooling: str = "max",
        # Encoder-specific kwargs
        lm_head_lora_r: int = 32,
        encoder_lora_r: int = 32,
        lm_head_full: bool = False,
        # Li-LSR query encoder kwargs
        query_lsr_lora_r: int = 0,
        query_lsr_activation: str = "relu",
        # Misc
        **kwargs,
    ):
        super().__init__()

        # V_SPLADE only supports the sparse head with a single (vbert) backbone.
        assert encoder_type == "vbert", "V_SPLADE only supports the vbert encoder."
        assert head_type == "sparse", "V_SPLADE only supports the sparse head."

        if pooling_type is None:
            pooling_type = DEFAULT_POOLING

        self.encoder_type = encoder_type
        self.head_type = head_type
        self.query_encoder_type = query_encoder_type
        self.splade_pooling = splade_pooling

        # Build encoder.
        encoder_kwargs = dict(
            model_name=model_name,
            lm_head_model=lm_head_model or model_name,
            lm_head_lora_r=lm_head_lora_r,
            encoder_lora_r=encoder_lora_r,
            lm_head_full=lm_head_full,
            **kwargs,
        )
        self.encoder = build_encoder(encoder_type, **encoder_kwargs)

        # Build pooling and sparse head.
        self.pooling = Pooling(pooling_type)
        lm_head = self.encoder.get_lm_head()
        self.head = SparseHead(lm_head, self.encoder.hidden_size)

        self.vocab_size = self.encoder.vocab_size
        self.hidden_size = self.encoder.hidden_size

        # Build query encoder: BOW (cheapest) or Li-LSR (inference-free learned).
        assert query_encoder_type in ("bow", "li_lsr"), (
            "V_SPLADE only supports the BOW or Li-LSR query encoders."
        )
        if query_encoder_type == "li_lsr":
            embed_layer = self.encoder.get_text_embeddings()
            if embed_layer is not None:
                embed_weight = embed_layer.weight
            else:
                # Fallback: use lm_head weights (tied with embed_tokens in causal LMs).
                _lm_head = self.encoder.get_lm_head()
                embed_weight = _lm_head.weight if _lm_head is not None else None
            self.query_encoder = build_query_encoder(
                "li_lsr",
                vocab_size=self.vocab_size,
                hidden_size=self.hidden_size,
                embed_weight=embed_weight,
                lora_r=query_lsr_lora_r,
                activation=query_lsr_activation,
            )
        else:
            self.query_encoder = build_query_encoder("bow", vocab_size=self.vocab_size)

        # Loss / regularization config.
        self.temperature = temperature
        self.cap_weight = cap_weight
        self.cap_sparse_rank_weight = cap_sparse_rank_weight
        self.cap_loss_mode = cap_loss_mode
        self.overlap_type = overlap_type
        self.reg_weight_p = reg_weight_p
        self.reg_weight_cap = reg_weight_cap
        self.loss_fn = nn.CrossEntropyLoss()
        self.reg_fn = FLOPSLoss()
        if cap_weight > 0:
            if use_zipfian_pushup:
                self.cap_push_up = ZipfianPushUpLoss(
                    push_focus_tau=push_focus_tau,
                    p_mean_alpha=p_mean_alpha,
                    cap_loss_mode=cap_loss_mode,
                )
            else:
                self.cap_push_up = CaptionPushUpLoss(
                    cap_loss_mode, p_mean_alpha=p_mean_alpha,
                )
        else:
            self.cap_push_up = None

        # Special-token mask for the VBert vocabulary. Prevents tokens such as
        # [CLS]/[SEP]/[PAD]/[MASK] (and out-of-MLM image tokens) from activating
        # in the sparse representation.
        _special_ids = [
            50280,  # [UNK]
            50281,  # [CLS]
            50282,  # [SEP]
            50283,  # [PAD]
            50284,  # [MASK]
            # Additional image/layout tokens beyond MLM head vocab (50368);
            # kept here for safety in case downstream code changes vocab_size.
            *range(50368, 50408),
        ]
        mask = torch.ones(self.vocab_size)
        for sid in _special_ids:
            if sid < self.vocab_size:
                mask[sid] = 0.0
        self.register_buffer("special_token_mask", mask, persistent=False)

    @classmethod
    def from_hf_export(cls, hf_dir: str,
                       query_lsr_activation: str = "softplus",
                       dtype: torch.dtype = torch.bfloat16) -> "UnifiedRetriever":
        """Build an empty V-SPLADE retriever shell from a V-SPLADE HF export.

        Only the module *structure* is created (random weights); call
        :func:`models.load_hf_export` afterwards to populate every tensor
        from the export's `model.safetensors` in one pass.
        """
        from models.encoder import VBertEncoder

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.encoder_type      = "vbert"
        instance.head_type         = "sparse"
        instance.query_encoder_type = "li_lsr"
        instance.splade_pooling    = "max"
        instance.temperature       = 1.0
        instance.cap_weight        = 0.0
        instance.cap_sparse_rank_weight = 0.0
        instance.cap_loss_mode     = "logsigmoid_h"
        instance.overlap_type      = "passage_mean"
        instance.reg_weight_p      = 0.0
        instance.reg_weight_cap    = 0.0
        instance.cap_push_up       = None
        instance.loss_fn           = nn.CrossEntropyLoss()
        instance.reg_fn            = FLOPSLoss()

        # Encoder + sparse head (shares the encoder's LM head via SparseHead).
        instance.encoder      = VBertEncoder.from_hf_export(hf_dir, dtype=dtype)
        instance.vocab_size   = instance.encoder.vocab_size
        instance.hidden_size  = instance.encoder.hidden_size
        instance.pooling      = Pooling("max")
        instance.head         = SparseHead(instance.encoder.get_lm_head(),
                                           instance.encoder.hidden_size)

        # Li-LSR inference-free query encoder (softplus activation, LoRA off).
        embed_layer = instance.encoder.get_text_embeddings()
        embed_weight = (embed_layer.weight if embed_layer is not None
                        else instance.encoder.get_lm_head().weight)
        instance.query_encoder = build_query_encoder(
            "li_lsr",
            vocab_size=instance.vocab_size,
            hidden_size=instance.hidden_size,
            embed_weight=embed_weight,
            lora_r=0,
            activation=query_lsr_activation,
        )

        # Special-token mask (same as training).
        _special_ids = [50280, 50281, 50282, 50283, 50284, *range(50368, 50408)]
        mask = torch.ones(instance.vocab_size)
        for sid in _special_ids:
            if sid < instance.vocab_size:
                mask[sid] = 0.0
        instance.register_buffer("special_token_mask", mask, persistent=False)

        for p in instance.parameters():
            p.requires_grad = False
        return instance.eval()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    # ---- Encoding helpers ----------------------------------------------------

    def _encode_passage_sparse(self, **kwargs):
        """Encode passage -> (h_raw, w_sparse) via encoder + sparse head."""
        hidden, mask = self.encoder.encode_passage(**kwargs)
        return self._apply_sparse_head(hidden, mask)

    def _encode_text_sparse(self, **kwargs):
        """Encode text (e.g. caption) -> (h_raw, w_sparse)."""
        hidden, mask = self.encoder.encode_text(**kwargs)
        return self._apply_sparse_head(hidden, mask)

    def _apply_sparse_head(self, hidden_states, attention_mask):
        """SPLADE sparse head: LM head (per position) -> max/mean pool over sequence."""
        scale = self.hidden_size ** -0.25

        # max/mean pooling: apply LM head over all positions, then pool.
        h = self.head.lm_head(hidden_states) * scale  # (B, seq, vocab)
        w = torch.log1p(torch.relu(h))
        mask = attention_mask.unsqueeze(-1).to(w.dtype)

        if self.splade_pooling == "max":
            w_out = (w * mask).max(dim=1).values
            h_masked = h * mask + (~mask.bool()) * (-1e9)
            h_out = h_masked.max(dim=1).values
        else:  # mean
            seq_len = mask.sum(dim=1, keepdim=True).clamp(min=1)
            w_out = (w * mask).sum(dim=1) / seq_len.squeeze(1)
            h_out = (h * mask).sum(dim=1) / seq_len.squeeze(1)

        # Mask special tokens.
        w_out = w_out * self.special_token_mask.to(w_out.dtype)
        return h_out, w_out

    def _encode_query(self, input_ids, attention_mask):
        """Encode query via BOW or Li-LSR (training: forward, eval: lookup)."""
        if isinstance(self.query_encoder, InferenceFreeQueryEncoder):
            if self.training:
                q = self.query_encoder(input_ids, attention_mask)
            else:
                q = self.query_encoder.encode_with_lookup(input_ids, attention_mask)
        else:
            q = self.query_encoder(input_ids, attention_mask)
        # Apply special-token mask to the query output.
        if q.dim() == 2 and q.size(-1) == self.special_token_mask.size(0):
            q = q * self.special_token_mask.to(q.dtype).to(q.device)
        return q

    def encode_query(self, input_ids, attention_mask) -> torch.Tensor:
        """Public API for query encoding."""
        return self._encode_query(input_ids, attention_mask)

    def encode_passage(
        self, input_ids=None, attention_mask=None,
        pixel_values=None, pixel_attention_mask=None,
    ) -> torch.Tensor:
        """Public API for passage encoding."""
        kwargs = {}
        if input_ids is not None:
            kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if pixel_values is not None:
            kwargs["pixel_values"] = pixel_values
        if pixel_attention_mask is not None:
            kwargs["pixel_attention_mask"] = pixel_attention_mask
        _, w = self._encode_passage_sparse(**kwargs)
        return w

    # ---- Forward -------------------------------------------------------------

    def forward(
        self,
        query_input_ids, query_attention_mask,
        passage_input_ids, passage_attention_mask,
        passage_pixel_values=None, passage_pixel_attention_mask=None,
        caption_input_ids=None, caption_attention_mask=None,
        hard_neg_passage_input_ids=None, hard_neg_passage_attention_mask=None,
        hard_neg_passage_pixel_values=None, hard_neg_passage_pixel_attention_mask=None,
        **kwargs,
    ) -> RetrievalOutput:
        """V_SPLADE forward pass.

        Computes NCE rank loss + FLOPS regularization + (optional) caption-gated token
        loss. Multi-vector / self-distillation / dense / multi-backbone paths
        are out of scope for this code release.
        """
        has_caption = caption_input_ids is not None
        use_cap = (
            self.cap_sparse_rank_weight > 0
            or self.cap_weight > 0
            or self.reg_weight_cap > 0
        ) and has_caption

        # ---- Query (BOW) ----
        q_reps = self._encode_query(query_input_ids, query_attention_mask)

        # ---- Positive passage ----
        p_kwargs = dict(input_ids=passage_input_ids, attention_mask=passage_attention_mask)
        if passage_pixel_values is not None:
            p_kwargs["pixel_values"] = passage_pixel_values
        if passage_pixel_attention_mask is not None:
            p_kwargs["pixel_attention_mask"] = passage_pixel_attention_mask
        p_h, p_reps = self._encode_passage_sparse(**p_kwargs)

        # ---- Hard negatives (optional) ----
        hn_reps = None
        if hard_neg_passage_input_ids is not None:
            if hard_neg_passage_pixel_values is not None:
                # Image hard negatives.
                hn_kwargs = dict(
                    input_ids=hard_neg_passage_input_ids,
                    attention_mask=hard_neg_passage_attention_mask,
                    pixel_values=hard_neg_passage_pixel_values,
                    pixel_attention_mask=hard_neg_passage_pixel_attention_mask,
                )
                _, hn_reps = self._encode_passage_sparse(**hn_kwargs)
            else:
                # Text hard negatives.
                _, hn_reps = self._encode_text_sparse(
                    input_ids=hard_neg_passage_input_ids,
                    attention_mask=hard_neg_passage_attention_mask,
                )

        # ---- Caption encoding ----
        cap_sparse = None
        cap_sparse_raw = None  # kept for diagnostics
        cap_bow = None
        if use_cap:
            _, cap_sparse_raw = self._encode_text_sparse(
                input_ids=caption_input_ids, attention_mask=caption_attention_mask,
            )
            # Use plain caption tokens for BOW if available (so that any instruction
            # tokens added by a caption prompt are not counted in the BOW mask).
            bow_ids = kwargs.get("caption_bow_input_ids", caption_input_ids)
            bow_mask = kwargs.get("caption_bow_attention_mask", caption_attention_mask)
            cap_bow = BOWQueryEncoder(self.vocab_size).to(q_reps.device)(bow_ids, bow_mask)
            cap_bow = cap_bow * self.special_token_mask.to(cap_bow.dtype).to(cap_bow.device)
            cap_sparse = cap_sparse_raw * cap_bow.to(cap_sparse_raw.dtype)

        # ---- dtype alignment (BOW query is float32, passage is bf16) ----
        if q_reps.dtype != p_reps.dtype:
            q_reps = q_reps.to(p_reps.dtype)

        # ---- Ranking loss (NCE) — cross-GPU gather for in-batch negatives ----
        # Symmetric gather across DDP ranks: every gathered query sees
        # ``world_size × B`` passages as negatives. The local rank's slice
        # keeps gradient via ``all_gather_2d_with_grad``; other ranks are
        # detached. Asymmetric gathers (only p) under-train the passage
        # encoder because each p sees gradients from only ``B`` queries.
        import torch.distributed as _dist_nce
        if _dist_nce.is_initialized() and _dist_nce.get_world_size() > 1:
            from models.losses import all_gather_2d_with_grad as _gather_nce
            q_reps_g = _gather_nce(q_reps)
            p_reps_g = _gather_nce(p_reps)
            hn_reps_g = _gather_nce(hn_reps) if hn_reps is not None else None
            logits, labels = compute_logits(
                q_reps_g, p_reps_g, hn_reps_g, self.temperature,
            )
        else:
            logits, labels = compute_logits(
                q_reps, p_reps, hn_reps, self.temperature,
            )
        rank_loss = self.loss_fn(logits, labels)

        # ---- FLOPS regularization ----
        reg_loss_p = self.reg_weight_p * self.reg_fn(p_reps)
        if hn_reps is not None:
            reg_loss_p = reg_loss_p + self.reg_weight_p * self.reg_fn(hn_reps)
        reg_loss_cap = q_reps.new_tensor(0.0)
        if cap_sparse is not None:
            reg_loss_cap = self.reg_weight_cap * self.reg_fn(cap_sparse)
        reg_loss = reg_loss_p + reg_loss_cap

        # ---- Caption loss ----
        cap_loss = q_reps.new_tensor(0.0)
        cap_sparse_rank_loss = q_reps.new_tensor(0.0)

        if use_cap and self.cap_sparse_rank_weight > 0 and cap_sparse is not None:
            # Cross-GPU gather for caption-side ranking too (parallel to the
            # passage-side NCE above): caption push-up still uses LOCAL p_reps.
            import torch.distributed as _dist_capr
            if _dist_capr.is_initialized() and _dist_capr.get_world_size() > 1:
                from models.losses import all_gather_2d_with_grad as _gather_capr
                q_reps_capr = _gather_capr(q_reps)
                cap_sparse_capr = _gather_capr(cap_sparse)
            else:
                q_reps_capr = q_reps
                cap_sparse_capr = cap_sparse
            cap_s_logits, cap_s_labels = compute_logits(
                q_reps_capr, cap_sparse_capr, None, self.temperature,
            )
            cap_sparse_rank_loss = self.loss_fn(cap_s_logits, cap_s_labels) * self.cap_sparse_rank_weight

        if use_cap and self.cap_weight > 0 and cap_sparse is not None and self.cap_push_up is not None:
            cap_loss = self.cap_push_up(p_h, p_reps, cap_sparse, self.overlap_type) * self.cap_weight

        # ---- Total loss ----
        loss = rank_loss + reg_loss + cap_loss + cap_sparse_rank_loss

        # ---- Diagnostics (no_grad) ----
        debug_tokens = None
        train_p_nnz = train_p_max = train_flops_qd = None
        train_cap_bow_killed_pct = None
        with torch.no_grad():
            def _topk_lowk(vec, k=5):
                """Return (top-k values, indices, low-k nonzero values, indices)."""
                nz_mask = vec > 0
                nz_n = int(nz_mask.sum().item())
                if nz_n == 0:
                    return (torch.empty(0), torch.empty(0, dtype=torch.long),
                            torch.empty(0), torch.empty(0, dtype=torch.long))
                t_k = min(k, vec.size(0))
                l_k = min(k, nz_n)
                top_v, top_i = vec.topk(t_k)
                masked = torch.where(nz_mask, vec, torch.full_like(vec, float('inf')))
                low_vals_asc, low_idx_asc = masked.topk(l_k, largest=False)
                return top_v.cpu(), top_i.cpu(), low_vals_asc.cpu(), low_idx_asc.cpu()

            q0, p0 = q_reps[0], p_reps[0]
            q_top_v, q_top_i, q_low_v, q_low_i = _topk_lowk(q0)
            p_top_v, p_top_i, p_low_v, p_low_i = _topk_lowk(p0)
            debug_tokens = {
                "q_indices": q_top_i, "q_values": q_top_v,
                "q_low_indices": q_low_i, "q_low_values": q_low_v,
                "p_indices": p_top_i, "p_values": p_top_v,
                "p_low_indices": p_low_i, "p_low_values": p_low_v,
                "q_input_ids": query_input_ids[0].cpu(),
                "q_attention_mask": query_attention_mask[0].cpu(),
                "q_nonzero": (q0 > 0).sum().item(),
                "p_nonzero": (p0 > 0).sum().item(),
            }
            # Batch sparsity / FLOPs diagnostics.
            train_p_nnz = (p_reps > 0).float().sum(-1).mean().detach()
            train_p_max = p_reps.max(dim=-1).values.mean().detach()
            train_flops_qd = (q_reps.abs().mean(dim=0) * p_reps.abs().mean(dim=0)).sum().detach()

            if cap_sparse is not None:
                cap0 = cap_sparse[0]
                c_top_v, c_top_i, c_low_v, c_low_i = _topk_lowk(cap0)
                debug_tokens["cap_indices"] = c_top_i
                debug_tokens["cap_values"] = c_top_v
                debug_tokens["cap_low_indices"] = c_low_i
                debug_tokens["cap_low_values"] = c_low_v
                debug_tokens["cap_nonzero"] = (cap0 > 0).sum().item()

                # Push-up overlap top-5 / low-5.
                p_w_d = p0.detach()
                p_mean_d = p_w_d.mean(dim=-1, keepdim=True)
                alpha = self.cap_push_up.p_mean_alpha if self.cap_push_up is not None else 1.0
                overlap = (p_w_d + alpha * p_mean_d) * cap0.detach()
                overlap_norm = overlap / (overlap.sum() + 1e-8)
                ov_top_v, ov_top_i, ov_low_v, ov_low_i = _topk_lowk(overlap_norm)
                debug_tokens["pushup_indices"] = ov_top_i
                debug_tokens["pushup_values"] = ov_top_v
                debug_tokens["pushup_low_indices"] = ov_low_i
                debug_tokens["pushup_low_values"] = ov_low_v

                # cap_bow_killed_pct: among bow-active caption tokens, what fraction
                # did the SPLADE head zero out? (Higher = model treats more cap
                # tokens as noise.)
                if cap_bow is not None and cap_sparse_raw is not None:
                    bow_active = (cap_bow > 0).float()
                    raw_active = (cap_sparse_raw > 0).float()
                    killed = bow_active * (1.0 - raw_active)
                    denom = bow_active.sum(-1).clamp(min=1.0)
                    train_cap_bow_killed_pct = (killed.sum(-1) / denom).mean().detach()

        return RetrievalOutput(
            loss=loss,
            rank_loss=rank_loss,
            reg_loss=reg_loss,
            reg_loss_p=reg_loss_p,
            reg_loss_cap=reg_loss_cap,
            cap_loss=cap_loss,
            cap_sparse_rank_loss=cap_sparse_rank_loss,
            train_p_nnz=train_p_nnz,
            train_p_max=train_p_max,
            train_flops_qd=train_flops_qd,
            train_cap_bow_killed_pct=train_cap_bow_killed_pct,
            query_reps=q_reps,
            passage_reps=p_reps,
            hard_neg_reps=hn_reps,
            debug_tokens=debug_tokens,
        )
