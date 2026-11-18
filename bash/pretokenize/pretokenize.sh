#!/usr/bin/env bash
# Pretokenize jsonl data into packed parquet shards.
# Each file is split into CPU_COUNT byte-ranges processed in parallel; documents
# are tokenized, concatenated with EOS, and packed into fixed-size blocks.
# State is tracked in a sqlite DB so the run is resumable.

INPUT_DIR=/mnt/data/users/truongnp5/final_clean_data/final_merge/**/*.jsonl
OUTPUT_DIR=/mnt/data/users/truongnp5/final_clean_data/vi_en_parquet_rm_leak_gemma4_pretokenized/
TOKENIZER=/mnt/data/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots/4d7ae4984b7db7de8f8457170b3f1a419ee76d52
HF_HOME=/mnt/data/users/truongnp5/cache/huggingface

# Optional arguments, e.g. --leakage-index path/to/index.json
python -m scripts.pretokenize \
    --input-dir       "$INPUT_DIR" \
    --output-dir      "$OUTPUT_DIR" \
    --tokenizer       "$TOKENIZER" \
    --hf-home         "$HF_HOME" \
    --chunk-size      8192 \
    --cpu-count       16 \
    --leakage-index   /mnt/data/users/truongnp5/uv_env/nanoqwen35/.cache/pretrain-leakage.json \
    --write-threshold 20000 "$@"
