#!/usr/bin/env bash
# V-SPLADE
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0

# Train V-SPLADE on the captioned visual retrieval corpus.
#
# All defaults below reproduce the paper *quality* recipe (Table 1).
# Override any of them via the corresponding env var before running.
#
# Required env vars:
#   DATASET_PATH        — HF datasets dir for the captioned ColPali set
#   BACKBONE_PATH       — pretrained BiModernVBERT checkpoint dir
#   OUTPUT_DIR          — where to write the trained checkpoint
#
# Optional env vars (defaults = paper *quality* recipe):
#   NUM_GPUS=4
#   BATCH_SIZE=42                 per-GPU batch
#   GRAD_ACCUM=4
#   NUM_EPOCHS=3
#   LEARNING_RATE=5e-4
#   WEIGHT_DECAY=0.01
#   WARMUP_RATIO=0.05
#   LR_SCHEDULER_TYPE=wsd
#   LR_DECAY_RATIO=0.2
#   MAX_GRAD_NORM=5.0
#   SEED=1
#   TEMPERATURE=0.1               contrastive τ
#   ENCODER_LORA_R=32             LoRA on text encoder
#   LM_HEAD_LORA_R=32             LoRA on MLM head
#   QUERY_ENCODER_TYPE=li_lsr     bow | li_lsr
#   QUERY_LSR_LORA_R=0            LoRA rank on the inference-free query
#                                 encoder. 0 = frozen embedding + 1-dim
#                                 projection only (Li-LSR original recipe,
#                                 paper baseline). Set to a positive int to
#                                 adapt the embedding too.
#   QUERY_LSR_ACTIVATION=softplus relu | softplus
#   CAP_WEIGHT=5.0                λ_cap (caption-gated supervision)
#   REG_WEIGHT_P=0.01             λ_p  (passage FLOPS, quality variant)
#   CAP_SPARSE_RANK_WEIGHT=1.0    λ_cap-sr (caption InfoNCE, quality variant)
#   REG_WEIGHT_CAP=0.005          caption FLOPS regularizer
#   REG_WARMUP_STEPS=500
#   CAP_LOSS_MODE=logsigmoid_h    push-up target activation
#   OVERLAP_TYPE=passage          overlap = p_w * cap_w (no passage_mean)
#   P_MEAN_ALPHA=0.0
#   USE_ZIPFIAN_PUSHUP=1          1 → enable temperature-sharpened push-up
#   PUSH_FOCUS_TAU=0.5            sharpening τ (<1 hard-mines top tokens)
#   RLHN_DATASET_PATH             enables mixed-modality training
#   RLHN_NUM_SAMPLES=300000
#   RLHN_NUM_HARD_NEGATIVES=2
#   RATIO_TEXT_TO_IMAGE=3         3 RLHN batches : 1 image batch
#   DDP_FIND_UNUSED_PARAMETERS=1  expose params unused on text-only RLHN batches
#                                 so DDP gradient sync is matched to the paper
#                                 baseline launch.
#   SAVE_STRATEGY=epoch
#   LOGGING_STEPS=25
#   SKIP_EVAL=1                   skip in-training eval; evaluate separately

set -euo pipefail

: "${DATASET_PATH:?set DATASET_PATH}"
# Defaults to the upstream ModernVBERT Hub id; the loader downloads + converts it
# to the V-SPLADE-compatible layout automatically (cached). Set BACKBONE_PATH to a
# local dir to use a pre-converted / custom backbone.
BACKBONE_PATH="${BACKBONE_PATH:-ModernVBERT/modernvbert}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-42}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-wsd}"
LR_DECAY_RATIO="${LR_DECAY_RATIO:-0.2}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-5.0}"
SEED="${SEED:-1}"

TEMPERATURE="${TEMPERATURE:-0.1}"

ENCODER_LORA_R="${ENCODER_LORA_R:-32}"
LM_HEAD_LORA_R="${LM_HEAD_LORA_R:-32}"

QUERY_ENCODER_TYPE="${QUERY_ENCODER_TYPE:-li_lsr}"
QUERY_LSR_LORA_R="${QUERY_LSR_LORA_R:-0}"
QUERY_LSR_ACTIVATION="${QUERY_LSR_ACTIVATION:-softplus}"

CAP_WEIGHT="${CAP_WEIGHT:-5.0}"
REG_WEIGHT_P="${REG_WEIGHT_P:-0.01}"
CAP_SPARSE_RANK_WEIGHT="${CAP_SPARSE_RANK_WEIGHT:-1.0}"
REG_WEIGHT_CAP="${REG_WEIGHT_CAP:-0.005}"
REG_WARMUP_STEPS="${REG_WARMUP_STEPS:-500}"

CAP_LOSS_MODE="${CAP_LOSS_MODE:-logsigmoid_h}"
OVERLAP_TYPE="${OVERLAP_TYPE:-passage}"
P_MEAN_ALPHA="${P_MEAN_ALPHA:-0.0}"
USE_ZIPFIAN_PUSHUP="${USE_ZIPFIAN_PUSHUP:-1}"
PUSH_FOCUS_TAU="${PUSH_FOCUS_TAU:-0.5}"

RLHN_NUM_SAMPLES="${RLHN_NUM_SAMPLES:-300000}"
RLHN_NUM_HARD_NEGATIVES="${RLHN_NUM_HARD_NEGATIVES:-2}"
RATIO_TEXT_TO_IMAGE="${RATIO_TEXT_TO_IMAGE:-3}"

DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-1}"

SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
LOGGING_STEPS="${LOGGING_STEPS:-25}"
SKIP_EVAL="${SKIP_EVAL:-1}"

cd "$(dirname "$0")/../train"

EXTRA_ARGS=()
if [[ -n "${RLHN_DATASET_PATH:-}" ]]; then
    EXTRA_ARGS+=(--rlhn_dataset_path "${RLHN_DATASET_PATH}")
    EXTRA_ARGS+=(--rlhn_num_samples "${RLHN_NUM_SAMPLES}")
    EXTRA_ARGS+=(--rlhn_num_hard_negatives "${RLHN_NUM_HARD_NEGATIVES}")
    EXTRA_ARGS+=(--ratio_text_to_image "${RATIO_TEXT_TO_IMAGE}")
fi
if [[ "${USE_ZIPFIAN_PUSHUP}" == "1" ]]; then
    EXTRA_ARGS+=(--use_zipfian_pushup)
fi
if [[ "${DDP_FIND_UNUSED_PARAMETERS}" == "1" ]]; then
    EXTRA_ARGS+=(--ddp_find_unused_parameters)
fi
if [[ "${SKIP_EVAL}" == "1" ]]; then
    EXTRA_ARGS+=(--skip_eval)
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

torchrun --nproc_per_node="${NUM_GPUS}" train.py \
    --encoder_type vbert \
    --head_type sparse \
    --splade_pooling max \
    --query_encoder_type "${QUERY_ENCODER_TYPE}" \
    --query_lsr_lora_r "${QUERY_LSR_LORA_R}" \
    --query_lsr_activation "${QUERY_LSR_ACTIVATION}" \
    --encoder_lora_r "${ENCODER_LORA_R}" \
    --lm_head_lora_r "${LM_HEAD_LORA_R}" \
    --model_name "${BACKBONE_PATH}" \
    --lm_head_model "${BACKBONE_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --caption_column caption \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --num_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --lr_decay_ratio "${LR_DECAY_RATIO}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --seed "${SEED}" \
    --temperature "${TEMPERATURE}" \
    --cap_weight "${CAP_WEIGHT}" \
    --cap_sparse_rank_weight "${CAP_SPARSE_RANK_WEIGHT}" \
    --reg_weight_q 0.0 \
    --reg_weight_p "${REG_WEIGHT_P}" \
    --reg_weight_cap "${REG_WEIGHT_CAP}" \
    --reg_warmup_steps "${REG_WARMUP_STEPS}" \
    --cap_loss_mode "${CAP_LOSS_MODE}" \
    --overlap_type "${OVERLAP_TYPE}" \
    --p_mean_alpha "${P_MEAN_ALPHA}" \
    --push_focus_tau "${PUSH_FOCUS_TAU}" \
    --bf16 \
    --gradient_checkpointing \
    --save_model \
    --save_strategy "${SAVE_STRATEGY}" \
    --logging_steps "${LOGGING_STEPS}" \
    "${EXTRA_ARGS[@]}"

echo "Training done. Checkpoint: ${OUTPUT_DIR}"
