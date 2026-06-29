#!/usr/bin/env bash
# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

# Generate captions for the ColPali training dataset with Qwen3-VL.
#
# Required environment variables:
#   MODEL_PATH      — directory of the Qwen3-VL checkpoint
#                     (e.g. .../Qwen3-VL-30B-A3B-Instruct)
#   DATASET_PATH    — HF datasets directory (save_to_disk format) of ColPali
#   OUTPUT_DIR      — where to write caption_*.jsonl shards
#   NUM_GPUS        — number of GPUs on this node (default: 8)

set -euo pipefail

: "${MODEL_PATH:?set MODEL_PATH to the Qwen3-VL checkpoint dir}"
: "${DATASET_PATH:?set DATASET_PATH to the ColPali HF dataset dir}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"

mkdir -p "${OUTPUT_DIR}"
cd "$(dirname "$0")/.."

for ((i = 0; i < NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i python scripts/generate_captions_colpali.py \
        --model_name "${MODEL_PATH}" \
        --dataset_path "${DATASET_PATH}" \
        --tp 1 \
        --shard_id "$i" \
        --num_shards "${NUM_GPUS}" \
        --batch_size "${BATCH_SIZE}" \
        --output_dir "${OUTPUT_DIR}" \
        > "${OUTPUT_DIR}/worker_${i}.log" 2>&1 &
done
wait

echo "All ${NUM_GPUS} workers finished. Output: ${OUTPUT_DIR}"
