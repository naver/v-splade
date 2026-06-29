#!/usr/bin/env bash
# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

# Encode a document corpus into V-SPLADE sparse vectors.
#
# Required env vars:
#   CHECKPOINT      — path to a trained V-SPLADE checkpoint dir
#   BACKBONE_PATH   — BiModernVBERT checkpoint dir (preprocessor + LM head)
#   DATASET_PATH    — HF datasets dir for the document corpus
#   OUTPUT_DIR      — where to write sparse_docs.npz / doc_ids.json
#
# Optional env vars (defaults match the paper inference recipe):
#   NUM_GPUS=1                     CUDA processor scales well on a single GPU
#   BATCH_SIZE=16
#   MAX_LENGTH=1024
#   IMAGE_COL=image
#   ID_COL=corpus-id
#   QUERY_ENCODER_TYPE=li_lsr
#   QUERY_LSR_LORA_R=32
#   QUERY_LSR_ACTIVATION=softplus

set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT}"
# Defaults to the upstream ModernVBERT Hub id; downloaded + converted automatically
# (cached). Set to a local dir to use a pre-converted / custom backbone.
BACKBONE_PATH="${BACKBONE_PATH:-ModernVBERT/modernvbert}"
: "${DATASET_PATH:?set DATASET_PATH}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

NUM_GPUS="${NUM_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
IMAGE_COL="${IMAGE_COL:-image}"
ID_COL="${ID_COL:-corpus-id}"
QUERY_ENCODER_TYPE="${QUERY_ENCODER_TYPE:-li_lsr}"
QUERY_LSR_LORA_R="${QUERY_LSR_LORA_R:-32}"
QUERY_LSR_ACTIVATION="${QUERY_LSR_ACTIVATION:-softplus}"

mkdir -p "${OUTPUT_DIR}"
cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

torchrun --nproc_per_node="${NUM_GPUS}" scripts/encode_sparse_documents.py \
    --checkpoint "${CHECKPOINT}" \
    --backbone "${BACKBONE_PATH}" \
    --lm_head_model "${BACKBONE_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --max_length "${MAX_LENGTH}" \
    --image_col "${IMAGE_COL}" \
    --id_col "${ID_COL}" \
    --query_encoder_type "${QUERY_ENCODER_TYPE}" \
    --query_lsr_lora_r "${QUERY_LSR_LORA_R}" \
    --query_lsr_activation "${QUERY_LSR_ACTIVATION}"

echo "Encoded sparse docs written to ${OUTPUT_DIR}"
