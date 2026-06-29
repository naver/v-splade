# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
V_SPLADE loss functions.

- FLOPSLoss:           Expected retrieval cost regularization (vocab-wise L2).
- NCELoss:             InfoNCE contrastive loss with in-batch + hard negatives.
- CaptionPushUpLoss:   Caption-gated token supervision loss — biases passage
                       sparse activations toward the vocab positions activated
                       by the paired caption.
- ZipfianPushUpLoss:   Same as ``CaptionPushUpLoss`` with a temperature
                       applied to the overlap distribution
                       (``push_focus_tau``): τ<1 hard-mines a few
                       strongly-weighted tokens; τ=1 recovers
                       ``CaptionPushUpLoss``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FLOPSLoss(nn.Module):
    """Expected retrieval cost: sum_v mean(|w|_v)^2."""

    def forward(self, reps: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.mean(torch.abs(reps), dim=0) ** 2)


class NCELoss(nn.Module):
    """InfoNCE contrastive loss with in-batch negatives + optional hard negatives."""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        q_reps: torch.Tensor,
        p_reps: torch.Tensor,
        hn_reps: torch.Tensor = None,
    ) -> torch.Tensor:
        B = q_reps.size(0)
        device = q_reps.device
        all_p = torch.cat([p_reps, hn_reps], dim=0) if hn_reps is not None else p_reps
        logits = (q_reps @ all_p.t()) / self.temperature
        labels = torch.arange(B, device=device)
        return self.loss_fn(logits, labels)


class CaptionPushUpLoss(nn.Module):
    """Caption-gated token supervision loss on passage sparse activations.

    Given the passage's sparse weights p_w (post log1p(relu)) and the
    caption's sparse weights cap_w (detached), an overlap distribution is
    formed from (p_w + alpha * mean(p_w)) * cap_w and renormalized.  The
    loss then maximizes the expected log-prob of activation under that
    distribution, biasing the passage to put mass on the vocab positions
    activated by the paired caption while remaining sparse elsewhere.
    """

    def __init__(self, cap_loss_mode: str = "logsigmoid_h", p_mean_alpha: float = 1.0):
        super().__init__()
        assert cap_loss_mode in ("logsigmoid_h", "raw_h", "raw_reps")
        self.cap_loss_mode = cap_loss_mode
        self.p_mean_alpha = p_mean_alpha

    def forward(
        self,
        p_h: torch.Tensor,
        p_reps: torch.Tensor,
        cap_sparse: torch.Tensor,
        overlap_type: str = "passage_mean",
    ) -> torch.Tensor:
        cap_w = cap_sparse.detach()
        p_w = p_reps.detach()

        if overlap_type == "passage_mean":
            p_mean = p_w.mean(dim=-1, keepdim=True)
            overlap = (p_w + self.p_mean_alpha * p_mean) * cap_w
        else:
            overlap = p_w * cap_w

        overlap = overlap / (overlap.sum(dim=-1, keepdim=True) + 1e-8)

        if self.cap_loss_mode == "raw_h":
            push_up_target_rep = p_h
        elif self.cap_loss_mode == "raw_reps":
            push_up_target_rep = p_reps
        else:  # logsigmoid_h (default) — BCE-style log P(active)
            push_up_target_rep = F.logsigmoid(p_h)

        per_sample = -(overlap * push_up_target_rep).sum(dim=-1)
        return per_sample.mean()


class ZipfianPushUpLoss(nn.Module):
    """Caption-gated token supervision — identical objective to
    ``CaptionPushUpLoss`` with a temperature on the overlap distribution.
    The name is a dev-time convenience for distinguishing the tempered
    variant in experiment configs.

    Applies a per-sample log-space softmax with ``push_focus_tau`` to the
    overlap distribution:

        o_sharp[v] = overlap[v]^(1/τ) / Σ_v' overlap[v']^(1/τ)

    - ``push_focus_tau < 1``: hard-mines a few strongly-weighted tokens.
    - ``push_focus_tau = 1``: identical to ``CaptionPushUpLoss``.

    The rest of the push-up pipeline matches ``CaptionPushUpLoss``.
    """

    def __init__(
        self,
        push_focus_tau: float = 1.0,
        p_mean_alpha: float = 1.0,
        cap_loss_mode: str = "logsigmoid_h",
    ):
        super().__init__()
        assert cap_loss_mode in ("logsigmoid_h", "raw_h", "raw_reps")
        self.push_focus_tau = push_focus_tau
        self.p_mean_alpha = p_mean_alpha
        self.cap_loss_mode = cap_loss_mode

    def forward(
        self,
        p_h: torch.Tensor,
        p_reps: torch.Tensor,
        cap_sparse: torch.Tensor,
        overlap_type: str = "passage_mean",
    ) -> torch.Tensor:
        cap_w = cap_sparse.detach()
        p_w = p_reps.detach()

        if overlap_type == "passage_mean":
            p_mean = p_w.mean(dim=-1, keepdim=True)
            overlap = (p_w + self.p_mean_alpha * p_mean) * cap_w
        else:  # "passage"
            overlap = p_w * cap_w
        # overlap is non-negative and zero wherever cap_w == 0.

        # Target sharpening via push_focus_tau (log-space, numerically stable).
        mask = (overlap > 0)
        neg_inf = torch.full_like(overlap, float("-inf"))
        log_overlap = torch.where(mask, torch.log(overlap.clamp(min=1e-30)), neg_inf)
        log_o_scaled = (1.0 / self.push_focus_tau) * log_overlap
        lse = torch.logsumexp(log_o_scaled, dim=-1, keepdim=True)
        o_sharp = torch.exp(log_o_scaled - lse)
        # Rows with no caption-active tokens → all -inf → exp(nan) → 0.
        o_sharp = torch.nan_to_num(o_sharp, nan=0.0, posinf=0.0, neginf=0.0)

        if self.cap_loss_mode == "raw_h":
            push_up_target_rep = p_h
        elif self.cap_loss_mode == "raw_reps":
            push_up_target_rep = p_reps
        else:  # logsigmoid_h (default)
            push_up_target_rep = F.logsigmoid(p_h)

        per_sample = -(o_sharp * push_up_target_rep).sum(dim=-1)
        return per_sample.mean()


def all_gather_2d_with_grad(local_tensor: torch.Tensor) -> torch.Tensor:
    """All-gather (B, D) across ranks, keeping gradient on the local slot.

    Used to give each query access to all ranks' passages as in-batch
    negatives during contrastive training.  Returns the input unchanged
    when distributed is not initialized or world_size <= 1.
    """
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return local_tensor
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, local_tensor)
    gathered[rank] = local_tensor  # preserve grad on local slot
    return torch.cat(gathered, dim=0)
