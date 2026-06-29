# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
V-SPLADE — Trainer, regularization scheduler, and evaluator.

RetrievalTrainer   : HuggingFace Trainer subclass with reg-weight scheduling
                     and per-step loss logging.
RegWeightScheduler : quadratic warmup for FLOPS regularization weights
                     (separate q / p / cap targets).
RetrievalMetrics   : Recall@k, NDCG@k, sparsity / FLOPS proxy.
BiEncoderEvaluator : encodes queries/passages in a retrieval dataset and
                     computes the metrics above (supports single-GPU + DDP).
ConvergenceStoppingCallback : early stop when the contrastive rank loss
                              stops changing.
"""

import logging
import os
import numpy as np
import torch
import torch.distributed as dist
from transformers import Trainer, TrainerCallback
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Regularization weight scheduler
# ──────────────────────────────────────────────────────────────

class RegWeightScheduler:
    """Quadratic warmup: weight = target * (step / warmup_steps)^2.

    Supports either a single target or a dict of separate
    {"q", "p", "cap"} weights.
    """

    def __init__(self, target_weight, warmup_steps: int):
        self.warmup_steps = warmup_steps
        self._step = 0
        if isinstance(target_weight, dict):
            self.targets = target_weight
            self.single = False
        else:
            self.targets = target_weight
            self.single = True

    def _scale(self) -> float:
        if self._step >= self.warmup_steps:
            return 1.0
        return (self._step / self.warmup_steps) ** 2

    def step(self):
        s = self._scale()
        self._step += 1
        if self.single:
            return self.targets * s
        return {k: v * s for k, v in self.targets.items()}


# ──────────────────────────────────────────────────────────────
# Convergence-based early stopping
# ──────────────────────────────────────────────────────────────

class ConvergenceStoppingCallback(TrainerCallback):
    """Stop training when rank_loss varies by less than `min_delta`
    over the most recent `window` steps. DDP-safe: the decision is
    broadcast from rank 0 to all ranks at a safe sync point.
    """

    def __init__(self, window: int = 100, min_delta: float = 0.01):
        self.window = window
        self.min_delta = min_delta
        self.history: list = []  # list of (global_step, rank_loss)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "rank_loss" not in logs:
            return
        self.history.append((state.global_step, logs["rank_loss"]))

        if len(self.history) < 2:
            return
        oldest_step = self.history[-1][0] - self.window
        recent = [v for step, v in self.history if step >= oldest_step]
        if len(recent) < 2:
            return

        if max(recent) - min(recent) < self.min_delta:
            logger.info(
                f"ConvergenceStopping: rank_loss range "
                f"{max(recent)-min(recent):.4f} < {self.min_delta} "
                f"over last {self.window} steps. Stopping."
            )
            self._should_stop = True

    def on_step_end(self, args, state, control, **kwargs):
        """Broadcast stop decision to all ranks at a safe sync point."""
        should_stop = getattr(self, "_should_stop", False)

        if dist.is_initialized():
            flag = torch.tensor([1 if should_stop else 0],
                                dtype=torch.long, device="cuda")
            dist.broadcast(flag, src=0)
            should_stop = flag.item() == 1

        if should_stop:
            control.should_training_stop = True
            self._should_stop = False


# ──────────────────────────────────────────────────────────────
# Custom Trainer
# ──────────────────────────────────────────────────────────────

class RetrievalTrainer(Trainer):
    """HuggingFace Trainer with FLOPS reg-weight scheduling and
    detailed per-step loss logging for the V-SPLADE objective.
    """

    def __init__(self, reg_weight_scheduler: RegWeightScheduler = None,
                 task_batch_sampler=None, wsd_decay_ratio: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.reg_weight_scheduler = reg_weight_scheduler
        self.task_batch_sampler = task_batch_sampler
        self.wsd_decay_ratio = wsd_decay_ratio
        # Per-task loss accumulators (visual vs. text-only RLHN batches are
        # logged under separate prefixes).
        self._task_loss_accum: dict[str, dict[str, float]] = {}
        self._task_loss_count: dict[str, int] = {}
        self._last_debug_tokens: dict | None = None

    def _save(self, output_dir=None, state_dict=None):
        """Override save to handle shared tensors safely
        (encoder.mlm_head shares weights with the SPLADE head's lm_head)."""
        if state_dict is None:
            raw = self.model.module if hasattr(self.model, "module") else self.model
            state_dict = {k: v.clone() for k, v in raw.state_dict().items()}
        super()._save(output_dir, state_dict=state_dict)

    def get_train_dataloader(self):
        """Use TaskAwareBatchSampler when provided.

        DDP data splitting is handled inside TaskAwareBatchSampler itself
        (rank / world_size aware), NOT via HuggingFace's BatchSamplerShard,
        because round-robin sharding would break the task-ratio
        interleaving pattern.
        """
        if self.task_batch_sampler is None:
            return super().get_train_dataloader()

        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_sampler=self.task_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """Use a WSD (Warmup-Stable-Decay) scheduler when
        ``wsd_decay_ratio > 0``, otherwise fall back to the HF default.
        """
        if self.wsd_decay_ratio <= 0:
            return super().create_scheduler(num_training_steps, optimizer=optimizer)

        from torch.optim.lr_scheduler import LambdaLR
        opt = optimizer or self.optimizer
        total = num_training_steps
        warmup = self.args.get_warmup_steps(total)
        decay_steps = int(total * self.wsd_decay_ratio)

        def wsd_lambda(step):
            if step < warmup:
                return step / max(warmup, 1)
            elif step < total - decay_steps:
                return 1.0
            else:
                return max(
                    0.0,
                    1.0 - (step - (total - decay_steps)) / max(decay_steps, 1),
                )

        self.lr_scheduler = LambdaLR(opt, wsd_lambda)
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        # Accumulate diagnostic losses per task type (visual = "colpali" vs.
        # text-only = "rlhn"). Logs are emitted with a "rlhn/" prefix for
        # text batches; visual batches use the default (unprefixed) keys.
        _LOSS_KEYS = (
            "rank_loss", "reg_loss_p", "reg_loss_cap",
            "cap_loss",
            # Diagnostic scalars
            "train_p_nnz", "train_p_max", "train_flops_qd",
        )
        task = inputs.get("task_type", "colpali")
        if task not in self._task_loss_accum:
            self._task_loss_accum[task] = {}
        self._task_loss_count[task] = self._task_loss_count.get(task, 0) + 1
        for key in _LOSS_KEYS:
            val = getattr(outputs, key, None)
            if val is not None:
                self._task_loss_accum[task][key] = (
                    self._task_loss_accum[task].get(key, 0.0) + val.item()
                )
        # Visual batches are the only ones with caption/image debug tokens.
        if task != "rlhn" and getattr(outputs, "debug_tokens", None) is not None:
            self._last_debug_tokens = outputs.debug_tokens

        return (loss, outputs) if return_outputs else loss

    def _log_top5_tokens(self, debug_tokens: dict):
        """Decode and log top-5 weighted tokens for query / caption+ / passage+."""
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            dc = getattr(self, "data_collator", None)
            if dc is not None:
                if hasattr(dc, "bow_tokenizer") and dc.bow_tokenizer is not None:
                    tokenizer = dc.bow_tokenizer
                elif hasattr(dc, "processor"):
                    tokenizer = dc.processor.tokenizer
        if tokenizer is None:
            return

        def _clean(s: str) -> str:
            return (
                s.replace("\\", "\\\\")
                 .replace("\n", "\\n")
                 .replace("\r", "\\r")
                 .replace("\t", "\\t")
            )

        lines = []
        if "q_input_ids" in debug_tokens and "q_attention_mask" in debug_tokens:
            ids = debug_tokens["q_input_ids"]
            mask = debug_tokens["q_attention_mask"]
            text = tokenizer.decode(ids[mask.bool()], skip_special_tokens=True)
            lines.append(f"  text: {_clean(text)[:120]}")
        for tag, idx_key, val_key, low_idx_key, low_val_key, nz_key in [
            ("query", "q_indices", "q_values", "q_low_indices", "q_low_values", "q_nonzero"),
            ("cap+",  "cap_indices", "cap_values", "cap_low_indices", "cap_low_values", "cap_nonzero"),
            ("img+",  "p_indices", "p_values", "p_low_indices", "p_low_values", "p_nonzero"),
        ]:
            if idx_key not in debug_tokens:
                if nz_key is None:
                    continue
                lines.append(f"  {tag}: (no caption input)")
                continue
            indices = debug_tokens[idx_key]
            values = debug_tokens[val_key]
            tokens = [_clean(tokenizer.decode([idx.item()])) for idx in indices]
            pairs = [f"{t}({v:.3f})" for t, v in zip(tokens, values.tolist())]
            nz = debug_tokens.get(nz_key, "") if nz_key else ""
            nz_str = f" [nz={nz}]" if nz else ""
            lines.append(f"  {tag}{nz_str}: {', '.join(pairs)}")
            if low_idx_key in debug_tokens and len(debug_tokens[low_idx_key]) > 0:
                low_idx = debug_tokens[low_idx_key]
                low_val = debug_tokens[low_val_key]
                low_toks = [_clean(tokenizer.decode([idx.item()])) for idx in low_idx]
                low_pairs = [f"{t}({v:.3f})" for t, v in zip(low_toks, low_val.tolist())]
                lines.append(f"  {tag:<5s} low: {', '.join(low_pairs)}")
        logger.info("Top-5 tokens [step %d]:\n%s",
                    self.state.global_step, "\n".join(lines))

    def _flush_loss_logs(self):
        """Emit accumulated loss averages once per logging step.

        Visual ("colpali") losses are logged with bare keys (e.g.
        ``rank_loss``); RLHN text-only losses use a ``rlhn/`` prefix so
        the two streams can be inspected independently in TensorBoard.
        """
        if not self._task_loss_count:
            return
        logs: dict = {}
        # Stable ordering: visual first, then text-only RLHN.
        for t in ["colpali"] + [k for k in self._task_loss_accum if k != "colpali"]:
            accum = self._task_loss_accum.get(t)
            if accum is None:
                continue
            cnt = self._task_loss_count.get(t, 1)
            prefix = "" if t == "colpali" else f"{t}/"
            for key, total in accum.items():
                logs[f"{prefix}{key}"] = total / cnt
            logs[f"{prefix}batches"] = cnt
        if logs:
            self.log(logs)
        if self.args.local_process_index == 0 and self._last_debug_tokens is not None:
            self._log_top5_tokens(self._last_debug_tokens)
        self._task_loss_accum.clear()
        self._task_loss_count.clear()
        self._last_debug_tokens = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs)

        if (self.state.global_step > 0
                and self.state.global_step % self.args.logging_steps == 0):
            self._flush_loss_logs()

        if self.reg_weight_scheduler is not None:
            new_w = self.reg_weight_scheduler.step()
            actual = model.module if hasattr(model, "module") else model
            if isinstance(new_w, dict):
                if hasattr(actual, "reg_weight_q"):
                    actual.reg_weight_q = new_w["q"]
                    actual.reg_weight_p = new_w["p"]
                    actual.reg_weight_cap = new_w["cap"]
            elif hasattr(actual, "reg_weight"):
                actual.reg_weight = new_w

        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs)
        if prediction_loss_only:
            return (outputs.loss, None, None)
        return (outputs.loss, (outputs.query_reps, outputs.passage_reps), None)


# ──────────────────────────────────────────────────────────────
# Retrieval metrics
# ──────────────────────────────────────────────────────────────

class RetrievalMetrics:

    @staticmethod
    def compute_retrieval_metrics(
        query_reps: torch.Tensor,
        passage_reps: torch.Tensor,
        ks: List[int] = [1, 5, 10],
        compute_sparsity: bool = True,
    ) -> Dict[str, float]:
        """Assumes query i matches passage i (diagonal ground truth)."""
        # Align dtypes (BOW query is float32, passage may be bfloat16)
        if query_reps.dtype != passage_reps.dtype:
            passage_reps = passage_reps.to(query_reps.dtype)
        sim = torch.matmul(query_reps, passage_reps.t())
        n = sim.size(0)
        _, indices = sim.sort(dim=1, descending=True)

        metrics = {}
        for k in ks:
            top_k = indices[:, :k]
            gt = torch.arange(n, device=indices.device).unsqueeze(1)
            hits = (top_k == gt).any(dim=1).float()
            metrics[f"recall@{k}"] = hits.mean().item()

            ndcg = 0.0
            for i in range(n):
                ranked = indices[i, :k].tolist()
                if i in ranked:
                    ndcg += 1.0 / np.log2(ranked.index(i) + 2)
            metrics[f"ndcg@{k}"] = ndcg / n

        if compute_sparsity:
            q_nz = (query_reps > 0).float().sum(dim=-1).mean().item()
            p_nz = (passage_reps > 0).float().sum(dim=-1).mean().item()
            metrics["query_avg_nonzero"] = q_nz
            metrics["passage_avg_nonzero"] = p_nz

            p_q = (query_reps > 0).float().mean(dim=0)
            p_d = (passage_reps > 0).float().mean(dim=0)
            metrics["eval_flops"] = (p_q * p_d).sum().item()

        return metrics


# ──────────────────────────────────────────────────────────────
# Bi-encoder evaluator
# ──────────────────────────────────────────────────────────────

class BiEncoderEvaluator:
    """Encode all queries and passages in a retrieval-style dataset and
    compute retrieval metrics. Supports single-GPU and DDP evaluation."""

    def __init__(
        self,
        model,
        processor,
        dataset,
        batch_size: int = 8,
        device: str = "cuda",
        world_size: int = 1,
        rank: int = 0,
    ):
        self.model = model
        self.processor = processor
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.world_size = world_size
        self.rank = rank

    @torch.no_grad()
    def evaluate(
        self, ks: List[int] = [1, 5, 10], compute_sparsity: bool = True
    ) -> Dict[str, float]:
        self.model.eval()
        orig_padding_side = self.processor.tokenizer.padding_side
        self.processor.tokenizer.padding_side = "left"
        n = len(self.dataset)

        # Distribute across ranks
        per_gpu = (n + self.world_size - 1) // self.world_size
        start = self.rank * per_gpu
        end = min(start + per_gpu, n)
        local_indices = list(range(start, end))

        # Encode queries
        local_q_reps = []
        for i in range(0, len(local_indices), self.batch_size):
            idx_batch = local_indices[i: i + self.batch_size]
            queries = [self.dataset[j]["query"] for j in idx_batch]

            enc = self.processor.process_queries(queries)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                reps = self.model.encode_query(
                    enc["input_ids"], enc["attention_mask"]
                )
            local_q_reps.append(reps.cpu())

        local_q = torch.cat(local_q_reps, dim=0) if local_q_reps else \
            torch.zeros(0, getattr(self.model, "vocab_size", 2048))

        # Encode passages (images)
        local_p_reps = []
        for i in range(0, len(local_indices), self.batch_size):
            idx_batch = local_indices[i: i + self.batch_size]
            images = [self.dataset[j]["image"] for j in idx_batch]

            enc = self.processor.process_images(images)
            enc = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                   for k, v in enc.items()}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                reps = self.model.encode_passage(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    pixel_values=enc.get("pixel_values"),
                    pixel_attention_mask=enc.get("pixel_attention_mask"),
                )
            local_p_reps.append(reps.cpu())

        local_p = torch.cat(local_p_reps, dim=0) if local_p_reps else \
            torch.zeros(0, getattr(self.model, "vocab_size", 2048))

        # Gather across GPUs
        if self.world_size > 1 and dist.is_initialized():
            all_q, all_p = self._gather_reps(local_q, local_p)
        else:
            all_q, all_p = local_q, local_p

        # Restore padding side
        self.processor.tokenizer.padding_side = orig_padding_side

        # Rank-0-only metric computation
        if self.rank == 0:
            return RetrievalMetrics.compute_retrieval_metrics(
                all_q, all_p, ks=ks, compute_sparsity=compute_sparsity
            )
        return {}

    def _gather_reps(self, local_q, local_p):
        """All-gather representations across DDP ranks."""
        local_size = torch.tensor([local_q.size(0)],
                                  dtype=torch.long, device=self.device)
        all_sizes_t = [torch.zeros(1, dtype=torch.long, device=self.device)
                       for _ in range(self.world_size)]
        dist.all_gather(all_sizes_t, local_size)
        all_sizes = [s.item() for s in all_sizes_t]
        max_size = max(all_sizes)
        dim = local_q.size(1)

        def _gather(local_tensor):
            padded = torch.zeros(max_size, dim,
                                 dtype=local_tensor.dtype, device=self.device)
            padded[:local_tensor.size(0)] = local_tensor.to(self.device)
            gathered = [torch.zeros_like(padded) for _ in range(self.world_size)]
            dist.all_gather(gathered, padded)
            return torch.cat([g[:s].cpu() for g, s in zip(gathered, all_sizes)], dim=0)

        return _gather(local_q), _gather(local_p)
