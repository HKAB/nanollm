#!/usr/bin/env bash
# Single-node GSM8K RL for Qwen3-0.6B-Base. Requires transformers >=4.57.

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
export WANDB_MODE="${WANDB_MODE:-offline}"
export NANOLLM_CACHE_DIR="${NANOLLM_CACHE_DIR:-$PWD/.cache/nanollm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$NANOLLM_CACHE_DIR/inductor}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "${TORCHRUN:-.venv/bin/torchrun}" --standalone --nproc_per_node="${NPROC_PER_NODE:-1}" \
    -m scripts.chat_rl_grpo -- \
    --device-type cuda \
    --hf-model "${HF_MODEL:-Qwen/Qwen3.5-0.8B}" \
    --run "${RUN_NAME:-qwen3.5-0.8b-grpo}" \
    --rollout-prompts 16 \
    --num-samples 8 \
    --minibatch-size 16 \
    --device-batch-size 4 \
    --max-new-tokens 256 \
    --max-steps 100 \
    --eval-every 25 \
    --eval-examples 32 \
    --sample-every 2 --sample-max-chars 1200 \
    --save-every 25 \
    "$@"
