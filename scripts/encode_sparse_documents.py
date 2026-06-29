# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
GPU sparse document encoder for V-SPLADE.

Loads a trained V-SPLADE (BiModernVBERT + SPLADE head) checkpoint and encodes
an image corpus into sparse passage vectors. Auto-detects DDP via torchrun
environment variables; each rank encodes a shard and dumps a partial CSR
matrix. Rank 0 merges all shards into a single sparse_docs.npz.

Output layout (under --output_dir):
    sparse_docs.npz        scipy CSR (shape [N_docs, vocab_size])
    doc_ids.json           list of doc IDs aligned to row index

Usage:
    torchrun --nproc_per_node=8 encode_sparse_documents.py \
        --checkpoint    $CHECKPOINT_DIR \
        --backbone      $BACKBONE_PATH \
        --dataset_path  $DATA_ROOT/corpus_hf \
        --output_dir    $OUTPUT_DIR \
        --batch_size    8
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
from scipy import sparse as sp


def _looks_like_arrow_dataset(path: str) -> bool:
    """True if ``path`` itself is an HF Dataset folder (has dataset_info.json
    and arrow shards), so it should be loaded directly, not as a BEIR root."""
    if not os.path.isdir(path):
        return False
    has_info = os.path.isfile(os.path.join(path, "dataset_info.json"))
    has_arrow = any(fn.endswith(".arrow") for fn in os.listdir(path))
    return has_info and has_arrow

# Import the V-SPLADE model package from train/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from models import build_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  Model loading (mirrors eval_vidore_v2.py)
# ══════════════════════════════════════════════════════════════════════
def load_vsplade_model(checkpoint_dir, backbone, lm_head_model, device,
                       query_encoder_type="bow",
                       query_lsr_lora_r=0,
                       query_lsr_activation="relu"):
    """Build a V-SPLADE retriever and load weights from a training checkpoint."""
    model = build_model(
        mode="from_scratch",
        encoder_type="vbert",
        head_type="sparse",
        query_encoder_type=query_encoder_type,
        model_name=backbone,
        lm_head_model=lm_head_model,
        splade_pooling="max",
        query_lsr_lora_r=query_lsr_lora_r,
        query_lsr_activation=query_lsr_activation,
    )

    full_model_path = os.path.join(checkpoint_dir, "model.safetensors")
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
        missing, unexpected = model.load_state_dict(sd, strict=False)
        logger.info(f"Loaded full state dict: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            from peft import PeftModel
            base_text = model.encoder.backbone.encoder.model.text_model.get_base_model()
            model.encoder.backbone.encoder.model.text_model = PeftModel.from_pretrained(
                base_text, checkpoint_dir,
            )
            logger.info(f"Loaded LoRA adapter from {checkpoint_dir}")
        extra_path = os.path.join(checkpoint_dir, "extra_modules.pt")
        if os.path.exists(extra_path):
            extra_sd = torch.load(extra_path, map_location="cpu")
            if "mlm_head" in extra_sd and hasattr(model.encoder, "mlm_head"):
                model.encoder.mlm_head.load_state_dict(extra_sd["mlm_head"], strict=False)
            if "query_encoder" in extra_sd and model.query_encoder is not None:
                model.query_encoder.load_state_dict(extra_sd["query_encoder"])

    return model.to(device)


# ══════════════════════════════════════════════════════════════════════
#  Corpus loading
# ══════════════════════════════════════════════════════════════════════
def load_corpus(dataset_path, image_col="image", id_col=None):
    """Load a corpus from any of:
      (a) HF Dataset directory (``load_from_disk``).
      (b) BEIR-layout root directory containing a ``corpus/`` subdir
          (e.g. ViDoRe v2 / v3 / VisRAG / ViDoCOOD per-corpus folders).
      (c) HuggingFace Hub repo id (downloaded via ``load_dataset``;
          BEIR-style configs use ``data_dir='corpus'``).
    Detects (image, corpus-id) columns automatically.
    """
    if os.path.isdir(dataset_path):
        beir_corpus = os.path.join(dataset_path, "corpus")
        if os.path.isdir(beir_corpus) and not _looks_like_arrow_dataset(dataset_path):
            ds = load_from_disk(beir_corpus)
        else:
            ds = load_from_disk(dataset_path)
    else:
        # Not a local path; try HF Hub (BEIR-style: pull the 'corpus' split).
        try:
            ds = load_dataset(dataset_path, data_dir="corpus", split="test")
        except Exception:
            ds = load_dataset(dataset_path, split="test")
    if image_col not in ds.column_names:
        candidates = [c for c in ds.column_names if "image" in c.lower()]
        if not candidates:
            raise ValueError(f"No image column found in {dataset_path}: {ds.column_names}")
        image_col = candidates[0]

    if id_col is None:
        for c in ("corpus-id", "corpus_id", "doc_id", "id", "dataset_id"):
            if c in ds.column_names:
                id_col = c
                break
    if id_col is None:
        ids = [str(i) for i in range(len(ds))]
    else:
        ids = [str(x) for x in ds[id_col]]
    return ds, ids, image_col


# ══════════════════════════════════════════════════════════════════════
#  Sparse encoding
# ══════════════════════════════════════════════════════════════════════
@torch.no_grad()
def encode_passages(model, processor, ds, image_col, indices,
                    batch_size, device, rank, log_every=10):
    """Encode passages in batches. Returns scipy CSR (len(indices), vocab_size)."""
    rows, cols, vals = [], [], []
    n = len(indices)
    t0 = time.time()
    for bstart in range(0, n, batch_size):
        bend = min(bstart + batch_size, n)
        batch_idx = indices[bstart:bend]
        batch_images = [ds[i][image_col] for i in batch_idx]
        batch_images = [
            img.convert("RGB") if hasattr(img, "convert") and img.mode != "RGB" else img
            for img in batch_images
        ]
        enc = processor.process_images(batch_images)
        enc = {k: v.to(device) for k, v in enc.items() if isinstance(v, torch.Tensor)}
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            reps = model.encode_passage(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                pixel_values=enc.get("pixel_values"),
                pixel_attention_mask=enc.get("pixel_attention_mask"),
            )
        reps = reps.float().cpu()
        # Collect non-zero entries per row.
        for local_row, vec in enumerate(reps):
            nz = torch.nonzero(vec, as_tuple=False).squeeze(-1)
            if nz.numel() == 0:
                continue
            global_row = bstart + local_row
            rows.append(np.full(nz.numel(), global_row, dtype=np.int64))
            cols.append(nz.numpy().astype(np.int64))
            vals.append(vec[nz].numpy().astype(np.float32))

        if (bend // batch_size) % log_every == 0:
            elapsed = time.time() - t0
            rate = bend / max(elapsed, 1e-6)
            logger.info(f"  rank {rank}: {bend}/{n} ({rate:.1f} docs/s, {elapsed:.1f}s elapsed)")

    vocab_size = model.vocab_size
    if rows:
        row_arr = np.concatenate(rows)
        col_arr = np.concatenate(cols)
        val_arr = np.concatenate(vals)
    else:
        row_arr = np.zeros(0, dtype=np.int64)
        col_arr = np.zeros(0, dtype=np.int64)
        val_arr = np.zeros(0, dtype=np.float32)
    return sp.csr_matrix((val_arr, (row_arr, col_arr)), shape=(n, vocab_size))


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="V-SPLADE GPU sparse document encoder")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained V-SPLADE checkpoint directory.")
    parser.add_argument("--backbone", type=str, required=True,
                        help="Backbone model path (BiModernVBERT-style).")
    parser.add_argument("--lm_head_model", type=str, default=None,
                        help="LM-head donor path. Defaults to --backbone.")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="HF dataset directory (load_from_disk) containing the image corpus.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Reserved for tokenizer truncation (processor-managed).")
    parser.add_argument("--image_col", type=str, default="image")
    parser.add_argument("--id_col", type=str, default=None)
    parser.add_argument("--query_encoder_type", type=str, default="bow",
                        choices=["bow", "li_lsr"])
    parser.add_argument("--query_lsr_lora_r", type=int, default=0)
    parser.add_argument("--query_lsr_activation", type=str, default="relu",
                        choices=["relu", "softplus"])
    parser.add_argument("--fast_image_processor", action="store_true", default=True,
                        help="Use Idefics3ImageProcessorFast on CUDA (~50x faster "
                             "preprocessing). Set --no-fast_image_processor to disable.")
    parser.add_argument("--no-fast_image_processor", dest="fast_image_processor",
                        action="store_false")
    args = parser.parse_args()

    # DDP setup.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = f"cuda:{local_rank}"

    os.makedirs(args.output_dir, exist_ok=True)
    final_npz = os.path.join(args.output_dir, "sparse_docs.npz")
    final_ids = os.path.join(args.output_dir, "doc_ids.json")
    if os.path.exists(final_npz) and os.path.exists(final_ids):
        if local_rank == 0:
            logger.info(f"Already encoded: {final_npz}")
        if world_size > 1:
            dist.destroy_process_group()
        return

    # Corpus + ID list.
    if local_rank == 0:
        logger.info(f"Loading corpus from {args.dataset_path}...")
    ds, doc_ids, image_col = load_corpus(args.dataset_path, args.image_col, args.id_col)
    n_total = len(ds)
    if local_rank == 0:
        logger.info(f"  {n_total} documents, image column = {image_col!r}")

    # Shard by rank: contiguous slice of indices.
    per_rank = (n_total + world_size - 1) // world_size
    start = local_rank * per_rank
    end = min(start + per_rank, n_total)
    local_indices = list(range(start, end))
    if local_rank == 0:
        logger.info(f"Sharding across {world_size} ranks ({per_rank} docs/rank).")

    # Load model + processor.
    from colpali_engine.models import BiModernVBertProcessor
    model = load_vsplade_model(
        args.checkpoint, args.backbone,
        args.lm_head_model or args.backbone,
        device,
        query_encoder_type=args.query_encoder_type,
        query_lsr_lora_r=args.query_lsr_lora_r,
        query_lsr_activation=args.query_lsr_activation,
    )
    model.eval()
    processor = BiModernVBertProcessor.from_pretrained(args.backbone)
    # Swap the Idefics3 image processor for its CUDA-accelerated variant
    # (~50x faster per-page than the default PIL/NumPy path). BICUBIC
    # resize replaces LANCZOS — empirically a < 1 pp NDCG@5 difference.
    if args.fast_image_processor and torch.cuda.is_available():
        from transformers import Idefics3ImageProcessorFast
        processor.image_processor = Idefics3ImageProcessorFast.from_pretrained(
            args.backbone, device=device,
        )
        if local_rank == 0:
            logger.info("Image processor: Idefics3ImageProcessorFast (cuda)")
    if local_rank == 0:
        logger.info(f"V-SPLADE model loaded: {args.checkpoint}")

    # Encode local shard.
    t0 = time.time()
    local_csr = encode_passages(
        model, processor, ds, image_col, local_indices,
        args.batch_size, device, local_rank,
    )
    logger.info(f"rank {local_rank}: encoded {local_csr.shape[0]} docs in {time.time()-t0:.1f}s "
                f"(nnz={local_csr.nnz})")

    # Dump per-rank shard.
    shard_npz = os.path.join(args.output_dir, f"shard_{local_rank:04d}.npz")
    sp.save_npz(shard_npz, local_csr)

    if world_size > 1:
        dist.barrier()

    # Rank 0 merges shards.
    if local_rank == 0:
        shards = []
        for r in range(world_size):
            sp_path = os.path.join(args.output_dir, f"shard_{r:04d}.npz")
            shards.append(sp.load_npz(sp_path))
        merged = sp.vstack(shards, format="csr")
        if merged.shape[0] != n_total:
            logger.warning(f"Row count mismatch: merged={merged.shape[0]}, expected={n_total}")
        sp.save_npz(final_npz, merged)
        with open(final_ids, "w") as f:
            json.dump(doc_ids, f)
        # Clean up per-rank shards.
        for r in range(world_size):
            os.remove(os.path.join(args.output_dir, f"shard_{r:04d}.npz"))
        logger.info(f"Saved {final_npz}  shape={merged.shape}  nnz={merged.nnz}")
        logger.info(f"Saved {final_ids}  ({len(doc_ids)} ids)")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
