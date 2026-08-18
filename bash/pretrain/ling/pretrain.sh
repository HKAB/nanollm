# Pre-training on pre-tokenized parquet files.
# Pipeline: raw jsonl → pretokenize.sh → this script.

export WANDB_MODE=offline
export NANOLLM_CACHE_DIR=/mnt/data/users/truongnp5/uv_env/nanoqwen35/.cache/nanollm

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m scripts.base_train \
    --run ling_tiny \
    --wandb-project nanollm \
    --wandb-entity hkab \
    --wandb-tags "ling,tiny,pretrain" \
    --gradient-checkpointing \
    --moe-bias-update-rate 1e-3 \
    --pretrained-model-path /mnt/data/huggingface/hub/models--inclusionAI--Ling-3.0-tiny/snapshots/a2ee06c0f2de5b171701aee7f73f70a1da75483b/ \
    --dataset-root /mnt/data/users/truongnp5/final_clean_data/vi_en_parquet_v1_pretokenized_ling \
    --max-seq-len 8192 \
    --num-iterations 57220 \
    --device-batch-size 1 \
    --total-batch-size 1048576 \
    --embedding-lr 5e-5 \
    --unembedding-lr 5e-5 \
    --matrix-lr 5e-5 \
    --scalar-lr 5e-5 \
    --warmdown-ratio 0.1 \
    --warmup-steps 2000 \
    --optimizer muon \
    --weight-decay 0.1 \
    --final-lr-frac 0.1 \
    --eval-every 1000 \
    --eval-tokens 1048576 \
    --core-metric-every 1000 \
    --core-metric-max-per-task 1000 \
    --sample-every 1000 \
    --save-every 5000