#!/usr/bin/env bash
# Single-node GSM8K RL for Qwen3.5-0.8B using nanoVLLM by default.

export WANDB_MODE="${WANDB_MODE:-offline}"
export NANOLLM_CACHE_DIR="${NANOLLM_CACHE_DIR:-$PWD/.cache/nanollm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$NANOLLM_CACHE_DIR/inductor}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -n "${MODEL_TAG:-}${MODEL_SOURCE:-}${MODEL_STEP:-}" ]]; then
    model_args=(--model-source "${MODEL_SOURCE:-sft}")
    [[ -z "${MODEL_TAG:-}" ]] || model_args+=(--model-tag "$MODEL_TAG")
    [[ -z "${MODEL_STEP:-}" ]] || model_args+=(--model-step "$MODEL_STEP")
else
    model_args=(--hf-model "${HF_MODEL:-Qwen/Qwen3.5-0.8B}" --hf-backend "${HF_BACKEND:-native}")
fi

torchrun --nproc_per_node="${NPROC_PER_NODE:-1}" \
    -m scripts.chat_rl_grpo -- \
    --device-type cuda \
    "${model_args[@]}" \
    --run "${RUN_NAME:-qwen3.5-0.8b-grpo}" \
    --rollout-prompts 16 \
    --num-samples 8 \
    --minibatch-size 16 \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-128}" \
    --device-batch-size 4 \
    --max-new-tokens 256 \
    --max-steps 100 \
    --eval-every 25 \
    --eval-examples 32 \
    --sample-every 2 --sample-max-chars 1200 \
    --save-every 25