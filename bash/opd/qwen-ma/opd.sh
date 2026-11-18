#!/usr/bin/env bash
# Qwen-MA OPD experiment on Vietnamese V-IFEval prompts.
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
export WANDB_MODE="${WANDB_MODE:-offline}"
export NANOLLM_CACHE_DIR="${NANOLLM_CACHE_DIR:-$PWD/.cache/nanollm}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$NANOLLM_CACHE_DIR/inductor}"

python_bin="${PYTHON:-python}"
if [[ -z "${PYTHON+x}" && -x .venv/bin/python ]]; then
    python_bin=.venv/bin/python
fi

if [[ -n "${QWEN_MODEL_PATH:-}" ]]; then
    : "${GEMMA_TOKENIZER_PATH:?Set GEMMA_TOKENIZER_PATH to the local Gemma 4 tokenizer directory}"
    student_kwargs="$("$python_bin" -c 'import json, sys; print(json.dumps({"gemma_tokenizer_path": sys.argv[1]}))' "$GEMMA_TOKENIZER_PATH")"
    student_args=(--hf-model "$QWEN_MODEL_PATH" --model-kwargs "$student_kwargs")
else
    student_args=(--model-source "${MODEL_SOURCE:-base}" --model-tag "${MODEL_TAG:-qwen-ma-2b}")
    if [[ -n "${MODEL_STEP:-}" ]]; then
        student_args+=(--model-step "$MODEL_STEP")
    fi
fi

if [[ -n "${TEACHER_HF_MODEL:-}" ]]; then
    teacher_args=(--teacher-hf-model "$TEACHER_HF_MODEL" --teacher-hf-backend "${TEACHER_HF_BACKEND:-transformers}")
else
    : "${TEACHER_MODEL_TAG:?Set TEACHER_MODEL_TAG or TEACHER_HF_MODEL to a checkpoint using the same Gemma 4 tokenizer}"
    teacher_args=(--teacher-model-source "${TEACHER_MODEL_SOURCE:-sft}" --teacher-model-tag "$TEACHER_MODEL_TAG")
    if [[ -n "${TEACHER_MODEL_STEP:-}" ]]; then
        teacher_args+=(--teacher-model-step "$TEACHER_MODEL_STEP")
    fi
fi

nproc="${NPROC_PER_NODE:-1}"
run_name="${RUN_NAME:-qwen-ma-opd-$(date -u +%Y%m%d-%H%M%S)}"
launch=("$python_bin")
if (( nproc > 1 )); then
    launch+=(-m torch.distributed.run --standalone --nproc-per-node "$nproc")
fi

exec "${launch[@]}" -m scripts.chat_opd \
    "${student_args[@]}" "${teacher_args[@]}" \
    --dataset hkab/vi-ifeval --dataset-split test \
    --prompt-format raw \
    --run "$run_name" \
    --rollout-prompts "${ROLLOUT_PROMPTS:-$((4 * nproc))}" \
    --num-samples "${NUM_SAMPLES:-1}" \
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}" --device-batch-size "${DEVICE_BATCH_SIZE:-1}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-128}" --max-steps "${MAX_STEPS:-100}" \
    --save-every "${SAVE_EVERY:-50}" --save-tag "${SAVE_TAG:-$run_name}" \
    "$@"
