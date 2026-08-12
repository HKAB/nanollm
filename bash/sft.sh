#!/usr/bin/env bash
# SFT training from a previous pretrain checkpoint.
#
# Expected pipeline:
#   raw ShareGPT jsonl -> bash/pretokenize_sft.sh -> this script
#
# Notes:
# - This script loads weights (and optimizer state) from base pretrain phase.
# - If your SFT data is raw messages parquet (not pretokenized), remove --pretokenized.

export WANDB_MODE=offline
export NANOLLM_CACHE_DIR=/mnt/data/users/truongnp5/uv_env/nanollm/.cache/nanollm

torchrun --nproc_per_node=8 --rdzv-conf "timeout=7200" -m scripts.chat_sft -- \
    --run qwen_0.8B_sft \
    --model-tag pretrained_0.8B \
    --model-step 5000 \
    --load-optimizer 1 \
    --no-compile \
    --dataset-root /mnt/data/users/truongnp5/final_clean_data/final_sft_pretokenized \
    --pretokenized \
    --max-seq-len 4096 \
    --num-iterations 3000 \
    --device-batch-size 2 \
    --total-batch-size 262144 \
    --embedding-lr 1e-5 \
    --unembedding-lr 1e-5 \
    --matrix-lr 1e-5 \
    --init-lr-frac 1.0 \
    --warmup-ratio 0.03 \
    --warmdown-ratio 0.6 \
    --final-lr-frac 0.05 \
    --eval-every 200 \
    --eval-tokens 1048576 \
    --chatcore-every -1
