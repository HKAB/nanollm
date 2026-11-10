#!/usr/bin/env bash
set -euo pipefail

# Usage: check_test_leakage.sh pretrain|sft 'path/**/*.jsonl' leakage.json [scanner options]
# Dependency: install pyahocorasick in the Python environment used below.
if (( $# < 3 )); then
    echo "Usage: $0 pretrain|sft 'path/**/*.jsonl' leakage.json [scanner options]" >&2
    exit 2
fi

mode=$1
input=$2
output=$3
shift 3
exec python -m scripts.check_test_leakage \
    --mode "$mode" --input-dir "$input" --output "$output" "$@"
