# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

"""
Generate captions for the ColPali training dataset using any Qwen3-VL model.

Designed for multi-GPU parallelism: launch one process per GPU with
CUDA_VISIBLE_DEVICES=N and --shard_id N --num_shards M.

Usage (8-GPU node, 2B model):
    for i in {0..7}; do
        CUDA_VISIBLE_DEVICES=$i python generate_captions_colpali.py \
            --model_name ${MODEL_PATH}/Qwen3-VL-2B-Instruct \
            --tp 1 --shard_id $i --num_shards 8 \
            --output_dir ${OUTPUT_DIR}/colpali_captions_2b &
    done
    wait

Output: {output_dir}/caption_{start_idx:06d}.jsonl
Each line: {"index": <row_index>, "caption": "...", "num_tokens": N}
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

from datasets import load_from_disk
from PIL import Image
from vllm import LLM, SamplingParams


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


DOCUMENT_PROMPT = (
    "You are an assistant specialized in document analysis.\n"
    "Given a table or a figure, provide a detailed summary\n"
    "(maximum 3000 characters).\n\n"
    "Your summary should be qualitative and not quantitative.\n\n"
    "Here is the table/figure:\n"
    "Answer ONLY with the caption."
)

DEFAULT_DATASET_PATH = os.environ.get(
    "COLPALI_DATASET_PATH",
    "./data/colpali_train",
)

BATCH_SIZE = 64


def _ensure_pil(img) -> Image.Image:
    """Convert HuggingFace image dict or bytes to PIL Image."""
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict) and "bytes" in img:
        import io
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    if isinstance(img, (str, bytes)):
        return Image.open(img).convert("RGB")
    return img


def build_prompt(img: Image.Image) -> dict:
    """Build vLLM prompt dict for Qwen3-VL."""
    return {
        "prompt": (
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            f"{DOCUMENT_PROMPT}"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "multi_modal_data": {"image": img},
    }


def generate_for_shard(
    llm: LLM,
    sampling_params: SamplingParams,
    dataset,
    indices,
    output_path: Path,
    batch_size: int,
    warmup_batches: int = 0,
    meta_path: Path = None,
):
    """Generate captions for the assigned row indices, with resume support."""
    done = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                done.add(rec["index"])
        log.info(f"Resume: {len(done)} already done")

    todo = [i for i in indices if i not in done]
    log.info(f"Shard: {len(todo)} remaining (of {len(indices)} total)")

    if not todo:
        log.info("Nothing to do, exiting.")
        return

    if warmup_batches > 0:
        wb = min(warmup_batches, max(1, len(todo) // 10))
        warm_indices = todo[: batch_size * wb]
        log.info(f"Warmup: {len(warm_indices)} images ({wb} batches)")
        for batch_start in range(0, len(warm_indices), batch_size):
            batch_indices = warm_indices[batch_start : batch_start + batch_size]
            prompts = [
                build_prompt(_ensure_pil(dataset[idx]["image"]))
                for idx in batch_indices
            ]
            _ = llm.generate(prompts, sampling_params)
        log.info("Warmup done")

    t_global = time.time()
    total_tokens = 0
    n_done_timed = 0

    for batch_start in range(0, len(todo), batch_size):
        batch_indices = todo[batch_start : batch_start + batch_size]

        prompts = []
        for idx in batch_indices:
            img = _ensure_pil(dataset[idx]["image"])
            prompts.append(build_prompt(img))

        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - t0

        batch_tokens = 0
        with open(output_path, "a") as f:
            for row_idx, out in zip(batch_indices, outputs):
                caption = out.outputs[0].text.strip()
                n_tokens = len(out.outputs[0].token_ids)
                batch_tokens += n_tokens
                record = {
                    "index": row_idx,
                    "caption": caption,
                    "num_tokens": n_tokens,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        total_tokens += batch_tokens
        n_done_timed += len(batch_indices)
        elapsed_total = time.time() - t_global
        done_so_far = batch_start + len(batch_indices)
        log.info(
            f"  {done_so_far}/{len(todo)} | "
            f"batch {elapsed:.1f}s ({batch_tokens} tok) | "
            f"total {elapsed_total:.0f}s | "
            f"{batch_tokens / max(elapsed, 0.1):.0f} tok/s"
        )

    total_elapsed = time.time() - t_global
    log.info(
        f"DONE: {len(todo)} images, {total_tokens} tokens, "
        f"{total_elapsed:.0f}s, "
        f"throughput {n_done_timed / max(total_elapsed, 1e-3):.3f} img/s"
    )

    if meta_path is not None:
        meta = {
            "n_images_timed": n_done_timed,
            "total_tokens": total_tokens,
            "elapsed_s": total_elapsed,
            "images_per_sec": n_done_timed / max(total_elapsed, 1e-3),
            "tokens_per_sec": total_tokens / max(total_elapsed, 1e-3),
            "warmup_batches_excluded": warmup_batches,
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ColPali training captions with Qwen3-VL"
    )
    parser.add_argument(
        "--model_name", type=str, required=True,
        help="Path to model dir (e.g. /path/to/Qwen3-VL-30B-A3B-Instruct)",
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--shard_id", type=int, required=True,
        help="Shard index (0..num_shards-1)",
    )
    parser.add_argument(
        "--num_shards", type=int, required=True,
        help="Total number of shards (= number of parallel workers)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for JSONL files",
    )
    parser.add_argument(
        "--dataset_path", type=str, default=DEFAULT_DATASET_PATH,
        help="Path to the ColPali training dataset (HF datasets save_to_disk format)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help="vLLM batch size",
    )
    parser.add_argument(
        "--warmup_batches", type=int, default=0,
        help="Number of warmup batches before starting the throughput timer.",
    )
    parser.add_argument(
        "--meta_path", type=str, default=None,
        help="Optional path to write a throughput summary JSON.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading dataset: {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)
    n_total = len(dataset)
    log.info(f"Dataset size: {n_total} rows")

    shard_size = math.ceil(n_total / args.num_shards)
    start = args.shard_id * shard_size
    end = min(start + shard_size, n_total)
    indices = list(range(start, end))
    log.info(
        f"Shard {args.shard_id}/{args.num_shards}: "
        f"rows {start}..{end-1} ({len(indices)} rows)"
    )

    output_path = output_dir / f"caption_{start:06d}.jsonl"
    log.info(f"Output: {output_path}")

    log.info(f"Initializing vLLM: {args.model_name} (tp={args.tp})")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=32768,
        mm_processor_kwargs={"max_pixels": 28 * 28 * 26000},
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.0,
        repetition_penalty=1.1,
        stop=["<|im_end|>"],
    )

    generate_for_shard(
        llm, sampling_params, dataset, indices, output_path, args.batch_size,
        warmup_batches=args.warmup_batches,
        meta_path=Path(args.meta_path) if args.meta_path else None,
    )

    log.info("Shard complete.")


if __name__ == "__main__":
    main()
