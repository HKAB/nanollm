
#!/usr/bin/env bash
set -euo pipefail

: "${QWEN_MODEL_PATH:?Set QWEN_MODEL_PATH to the local Qwen3.5 base snapshot}"
GEMMA_TOKENIZER_PATH="${GEMMA_TOKENIZER_PATH:-${GEMMA_MODEL_PATH:-}}"
: "${GEMMA_TOKENIZER_PATH:?Set GEMMA_TOKENIZER_PATH to the local Gemma 4 tokenizer directory}"
: "${DATASET_ROOT:?Set DATASET_ROOT to the Gemma-tokenized parquet dataset}"

# Resolve paths
QWEN_MODEL_PATH="$(cd -- "$QWEN_MODEL_PATH" && pwd -P)"
GEMMA_TOKENIZER_PATH="$(cd -- "$GEMMA_TOKENIZER_PATH" && pwd -P)"
DATASET_ROOT="$(cd -- "$DATASET_ROOT" && pwd -P)"

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."

export WANDB_MODE="${WANDB_MODE:-offline}"
export NANOLLM_CORE_DATASET="${NANOLLM_CORE_DATASET:-HKAB/nanollm-core}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

exec torchrun \
    --standalone \
    --nproc-per-node=8 \
    -m scripts.base_train -- \
    --run qwen-ma-2b \
    --model-tag qwen-ma-2b \
    --wandb-project nanollm \
    --wandb-tags "qwen-ma,2B,pretrain,8xh100,60B" \
    --pretrained-model-path "$QWEN_MODEL_PATH" \
    --gemma-tokenizer-path "$GEMMA_TOKENIZER_PATH" \
    --dataset-root "$DATASET_ROOT" \
    --max-seq-len 8192 \
    --num-iterations 57220 \
    --device-batch-size 2 \
    --total-batch-size 1048576 \
    --embedding-lr 5e-5 \
    --unembedding-lr 5e-5 \
    --matrix-lr 5e-5 \
    --scalar-lr 5e-5 \
    --warmdown-ratio 0.1 \
    --warmup-steps 2000 \
    --optimizer muon \
    --linear-cross-entropy-backend liger \
    --weight-decay 0.1 \
    --final-lr-frac 0.1 \
    --eval-every 10 \
    --eval-tokens 1048576 \
    --core-metric-every 10 \
    --core-metric-max-per-task 1000 \
    --sample-every 10 \
    --save-every 5000 \
    "$@"
