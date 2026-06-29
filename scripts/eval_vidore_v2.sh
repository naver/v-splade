#!/usr/bin/env bash
# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

# Evaluate a trained V-SPLADE checkpoint on ViDoRe v2.
#
# Required env vars:
#   CHECKPOINT       — path to a trained V-SPLADE checkpoint dir
#   BACKBONE_PATH    — BiModernVBERT checkpoint dir (preprocessor + LM head)
#   VIDORE_V2_PATH   — local cache of the ViDoRe v2 BEIR corpora
#   OUTPUT_PATH      — output JSON file (per-corpus metrics + average)
#
# Optional env vars (defaults match the paper inference recipe):
#   NUM_GPUS=4
#   BATCH_SIZE=16
#   LANGUAGE=english                only score queries in this language
#   QUERY_ENCODER_TYPE=li_lsr
#   QUERY_LSR_LORA_R=32
#   QUERY_LSR_ACTIVATION=softplus
#   DATASETS                        override default 4-corpus list (space-separated)

set -euo pipefail

: "${CHECKPOINT:?set CHECKPOINT}"
# Defaults to the upstream ModernVBERT Hub id; downloaded + converted automatically
# (cached). Ignored when CHECKPOINT is a self-contained HF export. Set to a local
# dir to use a pre-converted / custom backbone.
BACKBONE_PATH="${BACKBONE_PATH:-ModernVBERT/modernvbert}"
: "${VIDORE_V2_PATH:?set VIDORE_V2_PATH to the local ViDoRe v2 cache root}"
: "${OUTPUT_PATH:?set OUTPUT_PATH (a .json file)}"

NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LANGUAGE="${LANGUAGE:-english}"
QUERY_ENCODER_TYPE="${QUERY_ENCODER_TYPE:-li_lsr}"
QUERY_LSR_LORA_R="${QUERY_LSR_LORA_R:-32}"
QUERY_LSR_ACTIVATION="${QUERY_LSR_ACTIVATION:-softplus}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"
cd "$(dirname "$0")/.."

EXTRA_ARGS=()
if [[ -n "${DATASETS:-}" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS+=(--datasets ${DATASETS})
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

torchrun --nproc_per_node="${NUM_GPUS}" scripts/eval_vidore_v2.py \
    --model_path "${CHECKPOINT}" \
    --model_name "${BACKBONE_PATH}" \
    --lm_head_model "${BACKBONE_PATH}" \
    --query_encoder_type "${QUERY_ENCODER_TYPE}" \
    --query_lsr_lora_r "${QUERY_LSR_LORA_R}" \
    --query_lsr_activation "${QUERY_LSR_ACTIVATION}" \
    --cache_dir "${VIDORE_V2_PATH}" \
    --language "${LANGUAGE}" \
    --batch_size "${BATCH_SIZE}" \
    --output_path "${OUTPUT_PATH}" \
    "${EXTRA_ARGS[@]}"

echo "ViDoRe v2 evaluation finished. Metrics: ${OUTPUT_PATH}"
