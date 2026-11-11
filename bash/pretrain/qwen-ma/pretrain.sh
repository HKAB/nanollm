#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the local Qwen3.5 base snapshot}"
GEMMA_TOKENIZER_PATH="${GEMMA_TOKENIZER_PATH:-${GEMMA_MODEL_PATH:-}}"
: "${GEMMA_TOKENIZER_PATH:?Set GEMMA_TOKENIZER_PATH to the local Gemma 4 tokenizer directory}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the Gemma-tokenized parquet dataset}"

# Resolve relative paths before switching to the repository root.
QWEN_MODEL_PATH="$(cd -- "$QWEN_MODEL_PATH" && pwd -P)"
GEMMA_TOKENIZER_PATH="$(cd -- "$GEMMA_TOKENIZER_PATH" && pwd -P)"
DATASET_ROOT="$(cd -- "$DATASET_ROOT" && pwd -P)"
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

export WANDB_MODE="${WANDB_MODE:-offline}"
export NANOLLM_CORE_DATASET_REVISION="${NANOLLM_CORE_DATASET_REVISION:-HKAB/nanollm-core}"

exec torchrun --standalone --nproc-per-node=8 -m scripts.base_train \
    --run qwen-ma-2b \
    --model-tag qwen-ma-2b \
    --wandb-project nanollm \
    --wandb-tags "qwen-ma,2B,pretrain" \
    --pretrained-model-path "$QWEN_MODEL_PATH" \
    --gemma-tokenizer-path "$GEMMA_TOKENIZER_PATH" \
    --dataset-root "$DATASET_ROOT" \
    --max-seq-len 4096 \
    --num-iterations 200 \
    --device-batch-size 1 \
    --total-batch-size 131072 \
    --gradient-checkpointing \
    --embedding-lr 5e-5 \
    --unembedding-lr 5e-5 \
    --matrix-lr 5e-5 \
    --scalar-lr 5e-5 \
    --warmdown-ratio 0.1 \
    --warmup-steps 20 \
    --optimizer muon \
    --weight-decay 0.1 \
    --final-lr-frac 0.1 \
    --eval-every 50 \
    --eval-tokens 131072 \
    --core-metric-every -1 \
    --sample-every 50 \
    --save-every 100 \
    "$@"
