#!/usr/bin/env bash
set -euo pipefail

repo_dir="${1:-.cache/V-IFEval}"
commit="bc0083842e9562036a7d522fa348f6ca0861ac8e"
python_bin="${PYTHON:-python}"
if [[ -z "${PYTHON+x}" && -x .venv/bin/python ]]; then
    python_bin=.venv/bin/python
fi

if [[ ! -d "$repo_dir/.git" ]]; then
    git clone https://github.com/HKAB/V-IFEval.git "$repo_dir"
fi
if ! git -C "$repo_dir" cat-file -e "$commit^{commit}" 2>/dev/null; then
    git -C "$repo_dir" fetch origin "$commit"
fi
git -C "$repo_dir" checkout "$commit"

if "$python_bin" -m pip --version >/dev/null 2>&1; then
    "$python_bin" -m pip install absl-py==1.4.0 langdetect==1.0.9 nltk==3.9.1 underthesea huggingface-hub==0.34.4
elif command -v uv >/dev/null 2>&1; then
    uv pip install --python "$python_bin" absl-py==1.4.0 langdetect==1.0.9 nltk==3.9.1 underthesea huggingface-hub==0.34.4
else
    echo "Neither pip nor uv is available; cannot install V-IFEval dependencies." >&2
    exit 1
fi
"$python_bin" -m nltk.downloader punkt_tab

echo "V-IFEval is ready at $repo_dir"
echo "Set V_IFEVAL_PATH=$repo_dir if you use a different working directory."
