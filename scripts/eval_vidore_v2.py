# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
ViDoRe v2 evaluation for V-SPLADE (sparse) retrievers.

Encodes the four ViDoRe v2 English datasets with the trained V-SPLADE model
and reports nDCG@{5,10}, MAP, Recall@{5,10,100}. Supports torchrun-based DDP
for parallel passage encoding and bf16 autocast.

Reported metrics (per-dataset + macro average):
    - nDCG@{5, 10}
    - MAP@{5, 10}
    - Recall@{5, 10, 100}
    - FLOPs (mean per-token activations)
    - Avg. non-zeros (query / passage)

Usage:
    torchrun --nproc_per_node=8 eval_vidore_v2.py \
        --model_path  $CHECKPOINT_DIR \
        --model_name  $BACKBONE_PATH \
        --output_path $OUTPUT_DIR/vidore_v2_metrics.json \
        --cache_dir   $DATA_ROOT/vidore_v2
"""

import argparse
import datetime
import json
import logging
import os
import sys
from typing import Dict, List

import numpy as np
import pytrec_eval
import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk

# Make the in-repo `train/models` package importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from models import build_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ViDoRe v2 English datasets (BEIR layout: corpus / queries / qrels splits).
VIDORE_V2_DATASETS = [
    "vidore/esg_reports_eng_v2",
    "vidore/biomedical_lectures_eng_v2",
    "vidore/economics_reports_eng_v2",
    "vidore/esg_reports_human_labeled_v2",
]


# ══════════════════════════════════════════════════════════════════════
#  Lazy image list (avoids decoding the entire corpus into RAM)
# ══════════════════════════════════════════════════════════════════════
class _LazyImageList:
    """Wraps an HF Dataset and decodes PIL images on demand."""

    def __init__(self, ds, indices=None):
        self._ds = ds
        self._indices = indices

    def __len__(self):
        return len(self._indices) if self._indices is not None else len(self._ds)

    def _ds_idx(self, i):
        return self._indices[i] if self._indices is not None else i

    def __getitem__(self, key):
        if isinstance(key, slice):
            n = len(self)
            sub = list(range(*key.indices(n)))
            return _LazyImageList(self._ds, [self._ds_idx(i) for i in sub])
        return self._ds[self._ds_idx(key)]["image"]

    def __iter__(self):
        for i in range(len(self)):
            yield self._ds[self._ds_idx(i)]["image"]


# ══════════════════════════════════════════════════════════════════════
#  BEIR loader
# ══════════════════════════════════════════════════════════════════════
def _load_split(dataset_name, config, splits, **kwargs):
    last = None
    for split in splits:
        try:
            return load_dataset(dataset_name, data_dir=config, split=split, **kwargs)
        except Exception as e:
            last = e
    for split in splits:
        try:
            return load_dataset(dataset_name, name=config, split=split, **kwargs)
        except Exception as e:
            last = e
    raise RuntimeError(f"Could not load {dataset_name}/{config} from any of {splits}") from last


def load_beir_dataset(dataset_name, cache_dir=None, language=None):
    """Return (query_ids, queries, corpus_ids, corpus_images, qrels)."""
    kwargs = {"trust_remote_code": True}
    short = dataset_name.split("/")[-1]
    local_dir = os.path.join(cache_dir, short) if cache_dir else None

    if local_dir and os.path.isdir(local_dir):
        corpus_ds = load_from_disk(os.path.join(local_dir, "corpus"))
        queries_ds = load_from_disk(os.path.join(local_dir, "queries"))
        qrels_ds = load_from_disk(os.path.join(local_dir, "qrels"))
    else:
        splits = ["test", "train"]
        corpus_ds = _load_split(dataset_name, "corpus", splits, **kwargs)
        queries_ds = _load_split(dataset_name, "queries", splits, **kwargs)
        qrels_ds = _load_split(dataset_name, "qrels", splits, **kwargs)

    qid_col = "query-id" if "query-id" in queries_ds.column_names else "query_id"
    query_col = "query" if "query" in queries_ds.column_names else "text"
    cid_col = "corpus-id" if "corpus-id" in corpus_ds.column_names else "corpus_id"

    if language and "language" in queries_ds.column_names:
        queries_ds = queries_ds.filter(lambda x: x["language"] == language)

    query_ids = [str(q) for q in queries_ds[qid_col]]
    queries = list(queries_ds[query_col])
    corpus_ids = [str(c) for c in corpus_ds[cid_col]]
    corpus_images = _LazyImageList(corpus_ds)

    qrel_qid = "query-id" if "query-id" in qrels_ds.column_names else "query_id"
    qrel_cid = "corpus-id" if "corpus-id" in qrels_ds.column_names else "corpus_id"
    query_id_set = set(query_ids)
    qrels: Dict[str, Dict[str, int]] = {}
    for item in qrels_ds:
        qid = str(item[qrel_qid])
        if qid not in query_id_set:
            continue
        cid = str(item[qrel_cid])
        qrels.setdefault(qid, {})[cid] = int(item["score"])

    return query_ids, queries, corpus_ids, corpus_images, qrels


# ══════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════
def compute_mteb_metrics(
    qrels: Dict[str, Dict[str, int]],
    results: Dict[str, Dict[str, float]],
    k_values: List[int] = [1, 3, 5, 10, 20, 50, 100],
) -> Dict[str, float]:
    """nDCG / MAP / Recall / Precision via pytrec_eval (MTEB-style)."""
    measures = {
        "map_cut." + ",".join(str(k) for k in k_values),
        "ndcg_cut." + ",".join(str(k) for k in k_values),
        "recall." + ",".join(str(k) for k in k_values),
        "P." + ",".join(str(k) for k in k_values),
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    scores = evaluator.evaluate(results)

    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"ndcg_at_{k}"] = round(np.mean([s[f"ndcg_cut_{k}"] for s in scores.values()]), 5)
        metrics[f"recall_at_{k}"] = round(np.mean([s[f"recall_{k}"] for s in scores.values()]), 5)
        metrics[f"map_at_{k}"] = round(np.mean([s[f"map_cut_{k}"] for s in scores.values()]), 5)
        metrics[f"precision_at_{k}"] = round(np.mean([s[f"P_{k}"] for s in scores.values()]), 5)
    return metrics


def _compute_sparse_stats(all_q: torch.Tensor, all_p: torch.Tensor) -> Dict[str, float]:
    """Sparsity diagnostics: per-vector non-zeros + FLOPs proxy."""
    q_nz = (all_q > 0).float().sum(dim=-1).mean().item()
    p_nz = (all_p > 0).float().sum(dim=-1).mean().item()
    p_q = (all_q > 0).float().mean(dim=0)
    p_d = (all_p > 0).float().mean(dim=0)
    flops = (p_q * p_d).sum().item()
    return {
        "query_avg_nonzero": q_nz,
        "passage_avg_nonzero": p_nz,
        "eval_flops": flops,
    }


def _gather_reps(local_tensor: torch.Tensor, world_size: int, rank: int, device) -> torch.Tensor:
    """All-gather variable-sized 2D tensors with zero padding."""
    local_size = torch.tensor([local_tensor.size(0)], dtype=torch.long, device=device)
    all_sizes_t = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes_t, local_size)
    all_sizes = [int(s.item()) for s in all_sizes_t]
    max_size = max(all_sizes) if all_sizes else 0
    dim = local_tensor.size(1)

    padded = torch.zeros(max_size, dim, dtype=local_tensor.dtype, device=device)
    padded[: local_tensor.size(0)] = local_tensor.to(device)
    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)
    return torch.cat([g[:s].cpu() for g, s in zip(gathered, all_sizes)], dim=0)


# ══════════════════════════════════════════════════════════════════════
#  Model loading (V-SPLADE: vbert + sparse + bow/li_lsr)
# ══════════════════════════════════════════════════════════════════════
def _load_vsplade_model(model_path, model_name, lm_head_model, device,
                        query_encoder_type="bow",
                        query_lsr_lora_r=0,
                        query_lsr_activation="relu"):
    """Load a V-SPLADE model for inference.

    Two layouts are supported automatically:
      * **HF export** (a ``vsplade_config.json`` is present): a self-contained,
        LoRA-merged package. Loaded in one pass via
        ``build_model(mode="inference_only")`` — no base download, no LoRA wrapping.
      * **Training checkpoint** (no ``vsplade_config.json``): built from the
        backbone in ``from_scratch`` mode, then the full state dict / LoRA adapter
        + extra modules are dispatched onto it.
    """
    # ── HF export: load the packaged model directly ──────────────────
    if os.path.isfile(os.path.join(model_path, "vsplade_config.json")):
        logger.info(f"Loading V-SPLADE HF export (inference_only): {model_path}")
        model = build_model(
            path=model_path, mode="inference_only",
            query_lsr_activation=query_lsr_activation,
        )
        if query_encoder_type == "li_lsr" and model.query_encoder is not None:
            model.query_encoder.build_lookup_table()
        return model.to(device)

    # ── Training checkpoint: build from backbone, then load weights ──
    model = build_model(
        mode="from_scratch",
        encoder_type="vbert",
        head_type="sparse",
        query_encoder_type=query_encoder_type,
        model_name=model_name,
        lm_head_model=lm_head_model,
        splade_pooling="max",
        query_lsr_lora_r=query_lsr_lora_r,
        query_lsr_activation=query_lsr_activation,
    )

    full_model_path = os.path.join(model_path, "model.safetensors")
    if os.path.exists(full_model_path):
        from safetensors.torch import load_file
        sd = load_file(full_model_path)

        def _remap(key: str) -> str:
            if key.startswith("backbone."):
                key = "encoder." + key[len("backbone."):]
            if key.startswith("lsr_query_encoder."):
                key = "query_encoder." + key[len("lsr_query_encoder."):]
            return key

        if any(k.startswith(("backbone.", "lsr_query_encoder.")) for k in sd):
            sd = {_remap(k): v for k, v in sd.items()}
            logger.info("Remapped legacy keys (backbone -> encoder, lsr_query_encoder -> query_encoder)")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded full state dict: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        adapter_path = os.path.join(model_path, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            from peft import PeftModel
            base_text = model.encoder.backbone.encoder.model.text_model.get_base_model()
            model.encoder.backbone.encoder.model.text_model = PeftModel.from_pretrained(
                base_text, model_path,
            )
            logger.info(f"Loaded LoRA adapter from {model_path}")
        extra_path = os.path.join(model_path, "extra_modules.pt")
        if os.path.exists(extra_path):
            extra_sd = torch.load(extra_path, map_location="cpu")
            if "mlm_head" in extra_sd and hasattr(model.encoder, "mlm_head"):
                model.encoder.mlm_head.load_state_dict(extra_sd["mlm_head"], strict=False)
            if "query_encoder" in extra_sd and model.query_encoder is not None:
                model.query_encoder.load_state_dict(extra_sd["query_encoder"])
            logger.info("Loaded extra modules (mlm_head / query_encoder)")

    if query_encoder_type == "li_lsr" and model.query_encoder is not None:
        model.query_encoder.build_lookup_table()
    return model.to(device)


# ══════════════════════════════════════════════════════════════════════
#  Encoding
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def _encode_queries(model, processor, queries, batch_size, device):
    reps_list = []
    for i in range(0, len(queries), batch_size):
        batch = queries[i:i + batch_size]
        enc = processor.process_queries(batch)
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            reps = model.encode_query(enc["input_ids"], enc["attention_mask"])
        reps_list.append(reps.cpu())
    return reps_list


@torch.no_grad()
def _encode_passages(model, processor, images, batch_size, device):
    reps_list = []
    for i in range(0, len(images), batch_size):
        batch_images = list(images[i:i + batch_size])
        enc = processor.process_images(batch_images)
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            reps = model.encode_passage(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                pixel_values=enc.get("pixel_values"),
                pixel_attention_mask=enc.get("pixel_attention_mask"),
            )
        reps_list.append(reps.cpu())
    return reps_list


# ══════════════════════════════════════════════════════════════════════
#  Per-dataset evaluation (sharded encoding + all-gather)
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_dataset(model, processor, dataset_name, batch_size, device,
                     world_size, rank, cache_dir=None, language=None):
    model.eval()
    if rank == 0:
        logger.info(f"Loading {dataset_name}...")
    query_ids, queries, corpus_ids, corpus_images, qrels = load_beir_dataset(
        dataset_name, cache_dir=cache_dir, language=language,
    )
    if rank == 0:
        logger.info(f"  {len(queries)} queries, {len(corpus_ids)} corpus images")

    # Shard queries and passages across ranks (DDP-style data parallel encoding).
    q_per = (len(queries) + world_size - 1) // world_size
    p_per = (len(corpus_images) + world_size - 1) // world_size
    q_start, q_end = rank * q_per, min((rank + 1) * q_per, len(queries))
    p_start, p_end = rank * p_per, min((rank + 1) * p_per, len(corpus_images))

    q_reps = _encode_queries(model, processor, queries[q_start:q_end], batch_size, device)
    p_reps = _encode_passages(model, processor, corpus_images[p_start:p_end], batch_size, device)

    hidden_dim = model.vocab_size
    local_q = torch.cat(q_reps, dim=0) if q_reps else torch.zeros(0, hidden_dim)
    local_p = torch.cat(p_reps, dim=0) if p_reps else torch.zeros(0, hidden_dim)

    if world_size > 1 and dist.is_initialized():
        all_q = _gather_reps(local_q, world_size, rank, device)
        all_p = _gather_reps(local_p, world_size, rank, device)
    else:
        all_q, all_p = local_q, local_p

    meta = {"query_ids": query_ids, "corpus_ids": corpus_ids, "qrels": qrels}
    return all_q, all_p, meta


def score_sparse(all_q: torch.Tensor, all_p: torch.Tensor, meta: dict) -> Dict[str, float]:
    """Dot-product scoring + MTEB metrics + sparse FLOPs / non-zero stats."""
    if all_q.dtype != all_p.dtype:
        all_p = all_p.to(all_q.dtype)
    sim = torch.matmul(all_q, all_p.t())
    query_ids = meta["query_ids"]
    corpus_ids = meta["corpus_ids"]
    qrels = meta["qrels"]

    results = {
        qid: {cid: float(sim[qi, ci].item()) for ci, cid in enumerate(corpus_ids)}
        for qi, qid in enumerate(query_ids)
    }
    metrics = compute_mteb_metrics(qrels, results)
    metrics.update(_compute_sparse_stats(all_q, all_p))
    return metrics


# ══════════════════════════════════════════════════════════════════════
#  Output
# ══════════════════════════════════════════════════════════════════════
def _print_table(results: dict):
    print("\n" + "=" * 96)
    print(f"{'Dataset':<50} {'nDCG@5':>8} {'nDCG@10':>8} {'R@5':>8} {'R@10':>8} {'MAP@10':>8}")
    print("-" * 96)
    for name, m in results.items():
        if name == "_average":
            continue
        print(f"{name:<50} {m.get('ndcg_at_5', 0):>8.4f} {m.get('ndcg_at_10', 0):>8.4f} "
              f"{m.get('recall_at_5', 0):>8.4f} {m.get('recall_at_10', 0):>8.4f} "
              f"{m.get('map_at_10', 0):>8.4f}")
    if "_average" in results:
        a = results["_average"]
        print("-" * 96)
        print(f"{'Average':<50} {a['ndcg_at_5']:>8.4f} {a.get('ndcg_at_10', 0):>8.4f} "
              f"{a.get('recall_at_5', 0):>8.4f} {a.get('recall_at_10', 0):>8.4f} "
              f"{a.get('map_at_10', 0):>8.4f}")
    print("=" * 96)


def _aggregate_and_save(all_metrics: dict, output_path: str):
    valid = [m for m in all_metrics.values() if m and "ndcg_at_5" in m]
    if valid:
        keys = ["ndcg_at_5", "ndcg_at_10", "map_at_5", "map_at_10",
                "recall_at_5", "recall_at_10", "recall_at_100", "eval_flops"]
        avg = {k: float(np.mean([m.get(k, 0) for m in valid])) for k in keys}
        avg["num_datasets"] = len(valid)
        all_metrics["_average"] = avg
    _print_table(all_metrics)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Results saved to {output_path}")


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="ViDoRe v2 sparse evaluation (V-SPLADE)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="V-SPLADE checkpoint directory (model.safetensors or PEFT layout).")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Base backbone path (BiModernVBERT-style).")
    parser.add_argument("--lm_head_model", type=str, default=None,
                        help="LM-head donor path. Defaults to --model_name.")
    parser.add_argument("--query_encoder_type", type=str, default="bow",
                        choices=["bow", "li_lsr"],
                        help="Query encoder used during training.")
    parser.add_argument("--query_lsr_lora_r", type=int, default=0)
    parser.add_argument("--query_lsr_activation", type=str, default="relu",
                        choices=["relu", "softplus"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Local BEIR cache (per-dataset subdirs).")
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Override which ViDoRe v2 datasets to run.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Output JSON file path.")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    device = f"cuda:{local_rank}"

    # Load model + processor (V-SPLADE expects BiModernVBertProcessor).
    from colpali_engine.models import BiModernVBertProcessor
    model = _load_vsplade_model(
        args.model_path,
        args.model_name,
        args.lm_head_model or args.model_name,
        device,
        query_encoder_type=args.query_encoder_type,
        query_lsr_lora_r=args.query_lsr_lora_r,
        query_lsr_activation=args.query_lsr_activation,
    )
    model.eval()
    processor = BiModernVBertProcessor.from_pretrained(args.model_name)
    if local_rank == 0:
        logger.info(f"V-SPLADE model loaded: {args.model_path}")

    dataset_names = args.datasets if args.datasets else VIDORE_V2_DATASETS

    all_metrics = {}
    for ds_name in dataset_names:
        all_q, all_p, meta = evaluate_dataset(
            model, processor, ds_name, args.batch_size, device,
            world_size, local_rank, cache_dir=args.cache_dir, language=args.language,
        )
        if local_rank == 0:
            metrics = score_sparse(all_q, all_p, meta)
            short = ds_name.split("/")[-1]
            all_metrics[short] = metrics
            logger.info(
                f"  {short}: nDCG@5={metrics['ndcg_at_5']:.4f} "
                f"nDCG@10={metrics['ndcg_at_10']:.4f} "
                f"R@10={metrics['recall_at_10']:.4f} "
                f"FLOPs={metrics['eval_flops']:.2f}"
            )
        if world_size > 1 and dist.is_initialized():
            dist.barrier()

    if local_rank == 0:
        out = args.output_path or os.path.join(args.model_path, "vidore_v2_metrics.json")
        _aggregate_and_save(all_metrics, out)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
