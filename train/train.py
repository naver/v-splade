# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
V-SPLADE Training Script

Trains a sparse visual retriever (V-SPLADE) with:
  - BiModernVBERT backbone (vbert encoder)
  - SPLADE-style sparse head (LM-head + log1p(relu))
  - Bag-of-words query encoder
  - Caption-gated token supervision
  - FLOPS regularization (passage)

Supported configuration (this submission):
  --encoder_type vbert
  --head_type sparse
  --query_encoder_type bow
  --loss_mode legacy   (default; the only retained loss path)
"""

import os
import json
import argparse
import logging
from datetime import datetime

import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch.distributed as dist
from torch.utils.data import Subset
from transformers import TrainingArguments
from datasets import load_dataset

from models import build_model
from dataset import (
    VisualDataset, RLHNDataset, TaskAwareBatchSampler,
    RetrieverCollator, ModernVBertRetrieverCollator,
)
from trainer import (
    RetrievalTrainer, RegWeightScheduler, BiEncoderEvaluator,
    ConvergenceStoppingCallback,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="V-SPLADE Training")

    # Architecture
    p.add_argument("--encoder_type", type=str, default="vbert",
                   choices=["vbert"],
                   help="Encoder backbone (BiModernVBERT).")
    p.add_argument("--head_type", type=str, default="sparse",
                   choices=["sparse"],
                   help="Head type: sparse (LM head + log1p(relu)).")
    p.add_argument("--query_encoder_type", type=str, default="bow",
                   choices=["bow", "li_lsr"],
                   help="Query encoder: bow (bag-of-words) or li_lsr (learned "
                        "sparse retrieval with LoRA-tuned embedding lookup).")
    p.add_argument("--model_name", type=str, default="ModernVBERT/ModernVBERT")

    # Data
    p.add_argument("--dataset_path", type=str,
                   default="${DATA_ROOT}/docmatix_caption_train")
    p.add_argument("--caption_column", type=str, default="caption",
                   help="Caption column name.")
    p.add_argument("--shuffle_dataset", action="store_true", default=False)
    p.add_argument("--output_dir", type=str, default="./checkpoints/vsplade")

    # Training
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Max training steps (-1 = full epoch).")
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=5.0)

    # Model
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Contrastive temperature (sparse default = 1.0).")

    # FLOPS regularization
    p.add_argument("--reg_warmup_steps", type=int, default=250,
                   help="Quadratic warmup steps for reg weight (0 = none).")
    p.add_argument("--reg_weight_q", type=float, default=0.0,
                   help="FLOPS reg weight for query (0 when BOW).")
    p.add_argument("--reg_weight_p", type=float, default=0.0,
                   help="FLOPS reg weight for passage.")
    p.add_argument("--reg_weight_cap", type=float, default=0.0,
                   help="FLOPS reg weight for caption.")

    # Caption-gated token supervision
    p.add_argument("--cap_weight", type=float, default=0.0,
                   help="Caption-gated supervision weight.")
    p.add_argument("--overlap_type", type=str, default="passage_mean",
                   choices=["passage_mean", "passage"],
                   help="Caption-gated overlap formulation.")
    p.add_argument("--cap_loss_mode", type=str, default="logsigmoid_h",
                   choices=["logsigmoid_h", "raw_h", "raw_reps"],
                   help="Caption loss activation mode.")
    p.add_argument("--p_mean_alpha", type=float, default=1.0,
                   help="Multiplier for p_mean in passage_mean overlap.")
    p.add_argument("--cap_sparse_rank_weight", type=float, default=0.0,
                   help="Caption-side InfoNCE ranking loss weight λ_cap-sr.")
    p.add_argument("--use_zipfian_pushup", action="store_true", default=False,
                   help="Use ZipfianPushUpLoss (temperature-sharpened target) "
                        "instead of the linear CaptionPushUpLoss.")
    p.add_argument("--push_focus_tau", type=float, default=1.0,
                   help="Sharpening temperature for ZipfianPushUpLoss. τ<1 "
                        "hard-mines a few strongly-weighted tokens; τ=1 "
                        "recovers the linear push-up.")

    # VBert-specific
    p.add_argument("--lm_head_model", type=str, default="ModernVBERT/ModernVBERT",
                   help="Model with MLM head weights.")
    p.add_argument("--lm_head_lora_r", type=int, default=32,
                   help="LoRA rank for the MLM head.")
    p.add_argument("--lm_head_full", action="store_true",
                   help="Full parameter tuning for MLM head (no LoRA).")
    p.add_argument("--encoder_lora_r", type=int, default=32,
                   help="LoRA rank for the encoder backbone.")
    p.add_argument("--splade_pooling", type=str, default="max",
                   choices=["max", "mean", "eos", "cls"],
                   help="SPLADE pooling.")

    # Query encoder (li_lsr) options
    p.add_argument("--query_lsr_lora_r", type=int, default=0,
                   help="LoRA rank for the li_lsr query embedding lookup "
                        "(0 = no LoRA on the query encoder).")
    p.add_argument("--query_lsr_activation", type=str, default="relu",
                   choices=["relu", "softplus"],
                   help="Activation for the li_lsr query encoder.")

    # RLHN mixed-modality training
    p.add_argument("--rlhn_dataset_path", type=str, default=None,
                   help="Path to the RLHN text-only retrieval dataset "
                        "(load_from_disk). When set, the trainer mixes "
                        "RLHN text batches with visual batches at a fixed "
                        "ratio (TaskAwareBatchSampler).")
    p.add_argument("--rlhn_num_samples", type=int, default=300_000,
                   help="Number of RLHN rows to sample at startup.")
    p.add_argument("--rlhn_num_hard_negatives", type=int, default=2,
                   help="Hard negatives per RLHN row.")
    p.add_argument("--ratio_text_to_image", type=int, default=2,
                   help="Per-epoch ratio of RLHN (text) batches to visual "
                        "(image) batches in TaskAwareBatchSampler.")

    # WSD (Warmup-Stable-Decay) scheduler
    p.add_argument("--lr_scheduler_type", type=str, default="linear",
                   choices=["linear", "cosine", "wsd"],
                   help="Override HF scheduler. 'wsd' enables the "
                        "Warmup-Stable-Decay scheduler with --lr_decay_ratio.")
    p.add_argument("--lr_decay_ratio", type=float, default=0.0,
                   help="WSD final decay fraction of total steps "
                        "(0 = disabled; paper uses 0.2).")

    # Negatives
    p.add_argument("--num_hard_negatives", type=int, default=0,
                   help="Number of hard negatives (0 = in-batch only).")
    p.add_argument("--off_in_batch_negative", action="store_true", default=False,
                   help="Disable in-batch negatives (require hard negatives).")

    # Evaluation
    p.add_argument("--num_eval_samples", type=int, default=500)
    p.add_argument("--eval_dataset_path", type=str, default=None)
    p.add_argument("--eval_vidore_dir", type=str, default=None,
                   help="Directory containing multiple vidore subdatasets.")
    p.add_argument("--eval_subset", type=str, nargs="+", default=None,
                   help="Only evaluate these subdatasets.")
    p.add_argument("--skip_eval", action="store_true", default=False)

    # Convergence early stopping
    p.add_argument("--convergence_window", type=int, default=0,
                   help="Stop if rank_loss stagnates (0 = off).")
    p.add_argument("--convergence_min_delta", type=float, default=0.01)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--ddp_find_unused_parameters", action="store_true", default=False)
    p.add_argument("--dataloader_num_workers", type=int, default=6)
    p.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    p.add_argument("--save_model", action="store_true", default=False)
    p.add_argument("--save_strategy", type=str, default="no",
                   choices=["no", "epoch", "steps"])

    return p.parse_args()


def broadcast_timestamp(local_rank: int) -> str:
    """Make all ranks share the same output-directory timestamp."""
    if local_rank == 0:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        ts = None

    if dist.is_initialized():
        if local_rank == 0:
            ts_tensor = torch.tensor([ord(c) for c in ts], dtype=torch.long).cuda()
            len_tensor = torch.tensor([len(ts)], dtype=torch.long).cuda()
        else:
            len_tensor = torch.tensor([0], dtype=torch.long).cuda()

        dist.broadcast(len_tensor, src=0)
        L = len_tensor.item()
        if local_rank != 0:
            ts_tensor = torch.zeros(L, dtype=torch.long).cuda()
        dist.broadcast(ts_tensor, src=0)
        ts = "".join(chr(c) for c in ts_tensor.tolist())

    return ts


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize distributed
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    # Validation
    if args.off_in_batch_negative and args.num_hard_negatives == 0:
        raise ValueError(
            "--off_in_batch_negative requires --num_hard_negatives > 0"
        )

    # Output dir
    ts = broadcast_timestamp(local_rank)
    run_name = f"vsplade_{ts}"
    output_dir = os.path.join(args.output_dir, run_name)
    if local_rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"encoder=vbert head=sparse "
                    f"query={args.query_encoder_type} output={output_dir}")

    # Processor (BiModernVBertProcessor)
    from colpali_engine.models import BiModernVBertProcessor
    if local_rank == 0:
        processor = BiModernVBertProcessor.from_pretrained(args.model_name)
    if dist.is_initialized():
        dist.barrier()
    if local_rank != 0:
        processor = BiModernVBertProcessor.from_pretrained(args.model_name)

    # Model
    model_kwargs = dict(
        model_name=args.model_name,
        temperature=args.temperature,
        # Caption (caption-gated token supervision)
        cap_weight=args.cap_weight,
        cap_sparse_rank_weight=args.cap_sparse_rank_weight,
        cap_loss_mode=args.cap_loss_mode,
        overlap_type=args.overlap_type,
        p_mean_alpha=args.p_mean_alpha,
        use_zipfian_pushup=args.use_zipfian_pushup,
        push_focus_tau=args.push_focus_tau,
        # Regularization
        reg_weight_q=args.reg_weight_q,
        reg_weight_p=args.reg_weight_p,
        reg_weight_cap=args.reg_weight_cap,
        # SPLADE pooling
        splade_pooling=args.splade_pooling,
        # Loss path
        loss_mode="legacy",
        # Query encoder (li_lsr)
        query_lsr_lora_r=args.query_lsr_lora_r,
        query_lsr_activation=args.query_lsr_activation,
        # VBert
        lm_head_model=args.lm_head_model,
        lm_head_lora_r=args.lm_head_lora_r,
        lm_head_full=args.lm_head_full,
        encoder_lora_r=args.encoder_lora_r,
    )

    model = build_model(
        mode="from_scratch",
        encoder_type="vbert",
        head_type="sparse",
        query_encoder_type=args.query_encoder_type,
        **model_kwargs,
    )
    if local_rank == 0:
        logger.info(
            f"Model: vbert/sparse/{args.query_encoder_type}"
        )

    # Dataset
    visual_ds = VisualDataset(
        dataset_path=args.dataset_path,
        caption_column=args.caption_column,
    )

    # Mixed-modality: RLHN (text) + Visual (image) with TaskAwareBatchSampler
    task_batch_sampler = None
    if args.rlhn_dataset_path:
        from torch.utils.data import ConcatDataset
        rlhn_ds = RLHNDataset(
            dataset_path=args.rlhn_dataset_path,
            num_samples=args.rlhn_num_samples,
            num_hard_negatives=args.rlhn_num_hard_negatives,
            seed=args.seed,
        )
        train_ds = ConcatDataset([rlhn_ds, visual_ds])
        # RLHN rows always carry >=1 hard negatives → make sure the
        # collator allocates space for them.
        args.num_hard_negatives = max(
            args.num_hard_negatives, args.rlhn_num_hard_negatives
        )

        task_batch_sampler = TaskAwareBatchSampler(
            rlhn_size=len(rlhn_ds),
            colpali_size=len(visual_ds),
            rlhn_offset=0,
            colpali_offset=len(rlhn_ds),
            batch_size=args.batch_size,
            ratio_text_to_image=args.ratio_text_to_image,
            seed=args.seed,
            rank=local_rank,
            world_size=world_size,
        )
        if local_rank == 0:
            logger.info(
                f"Mixed-modality: RLHN={len(rlhn_ds)}, "
                f"Visual={len(visual_ds)}, "
                f"ratio={args.ratio_text_to_image}:1 (text:image), "
                f"batches/epoch/rank={len(task_batch_sampler)}"
            )
    else:
        train_ds = visual_ds

    if local_rank == 0:
        logger.info(f"Train: {len(train_ds)}")

    # Collator
    collator = ModernVBertRetrieverCollator(
        processor=processor,
        num_hard_negatives=args.num_hard_negatives,
        caption_field="caption" if args.caption_column else None,
    )

    # Reg weight scheduler
    reg_scheduler = None
    if args.reg_warmup_steps > 0:
        target = {"q": args.reg_weight_q, "p": args.reg_weight_p, "cap": args.reg_weight_cap}
        reg_scheduler = RegWeightScheduler(target, args.reg_warmup_steps)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        eval_strategy="no",
        save_strategy=args.save_strategy,
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor if args.dataloader_num_workers > 0 else None,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=["tensorboard"],
        seed=args.seed,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )

    # Callbacks
    callbacks = []
    if args.convergence_window > 0:
        callbacks.append(ConvergenceStoppingCallback(
            window=args.convergence_window,
            min_delta=args.convergence_min_delta,
        ))
        if local_rank == 0:
            logger.info(f"Convergence stopping: window={args.convergence_window}, "
                        f"min_delta={args.convergence_min_delta}")

    # WSD scheduler is selected via --lr_scheduler_type wsd OR
    # whenever --lr_decay_ratio > 0 is requested.
    use_wsd = args.lr_scheduler_type == "wsd" or args.lr_decay_ratio > 0.0
    wsd_decay_ratio = args.lr_decay_ratio if use_wsd else 0.0
    if use_wsd and local_rank == 0:
        logger.info(
            f"WSD scheduler enabled: warmup_ratio={args.warmup_ratio}, "
            f"decay_ratio={wsd_decay_ratio}"
        )

    trainer = RetrievalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=None,
        data_collator=collator,
        compute_metrics=None,
        reg_weight_scheduler=reg_scheduler,
        task_batch_sampler=task_batch_sampler,
        wsd_decay_ratio=wsd_decay_ratio,
        callbacks=callbacks,
    )

    # Train
    if local_rank == 0:
        logger.info("Starting training...")
    trainer.train()

    # Save model
    if args.save_model:
        trainer.save_model(output_dir)
        if local_rank == 0:
            logger.info(f"Model saved to {output_dir}")

    # Evaluation
    if not args.skip_eval:
        eval_model = model.module if hasattr(model, "module") else model
        del trainer
        import gc; gc.collect()
        torch.cuda.empty_cache()

        eval_datasets = {}
        if args.eval_vidore_dir:
            from datasets import load_from_disk
            for sub in sorted(os.listdir(args.eval_vidore_dir)):
                if args.eval_subset and sub not in args.eval_subset:
                    continue
                sub_path = os.path.join(args.eval_vidore_dir, sub)
                if os.path.isdir(sub_path) and not sub.startswith("."):
                    eval_datasets[sub] = load_from_disk(sub_path)
        elif args.eval_dataset_path:
            from datasets import load_from_disk
            eval_datasets["docvqa"] = load_from_disk(args.eval_dataset_path)
        else:
            eval_datasets["docvqa"] = load_dataset("vidore/docvqa_test_subsampled", split="test")

        all_metrics = {}
        for ds_name, ds in eval_datasets.items():
            if 0 < args.num_eval_samples < len(ds):
                torch.manual_seed(42)
                idxs = torch.randperm(len(ds))[:args.num_eval_samples].tolist()
                ds_eval = Subset(ds, idxs)
            else:
                ds_eval = ds

            if local_rank == 0:
                logger.info(f"Evaluating {ds_name}: {len(ds_eval)} samples")

            evaluator = BiEncoderEvaluator(
                model=eval_model, processor=processor, dataset=ds_eval,
                batch_size=min(args.batch_size, 8), device="cuda",
                world_size=world_size, rank=local_rank,
            )
            m = evaluator.evaluate(ks=[1, 5, 10], compute_sparsity=True)
            if local_rank == 0:
                logger.info(f"  {ds_name}: {m}")
            all_metrics[ds_name] = m

        if local_rank == 0:
            if len(all_metrics) > 1:
                avg_keys = ["recall@1", "recall@5", "recall@10",
                            "ndcg@5", "ndcg@10", "eval_flops"]
                avg = {}
                for k in avg_keys:
                    vals = [m[k] for m in all_metrics.values() if k in m]
                    if vals:
                        avg[k] = sum(vals) / len(vals)
                all_metrics["avg"] = avg
                logger.info(f"  AVG: {avg}")

            metrics_path = os.path.join(output_dir, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(all_metrics, f, indent=2)
            logger.info(f"Metrics saved to {metrics_path}")

    if not args.save_model and local_rank == 0:
        logger.info("Skipping model save (pass --save_model to enable).")


if __name__ == "__main__":
    main()
