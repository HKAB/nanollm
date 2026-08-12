"""
GSM8K generative math task.

Backed by Hugging Face dataset `openai/gsm8k`.

Each example is exposed as a conversation in the same shape as
`tasks/global_mmlu.py`:

{
  "messages": [
    {"role": "user", "content": "...question..."},
    {"role": "assistant", "content": "...full reference solution..."},
  ],
  "answer": "...normalized final numeric answer..."
}
"""

from __future__ import annotations

import re
from typing import Any

from datasets import load_dataset


GSM_RE = re.compile(r"####\s*(\-?[0-9\.,]+)")
MAX_NEW_TOKENS = 256


def extract_answer(completion: str) -> str | None:
    """Extract the normalized numeric answer after the `####` marker.

    Follows the official GSM8K normalization convention:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    if not isinstance(completion, str):
        return None
    match = GSM_RE.search(completion)
    if match is None:
        return None
    return match.group(1).strip().replace(",", "")


class GSM8K:
    """GSM8K task compatible with the project's generative evaluation loops."""

    eval_type = "generative"
    max_new_tokens = MAX_NEW_TOKENS

    def __init__(
        self,
        subset: str,
        split: str,
        *,
        limit: int | None = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if subset not in {"main", "socratic"}:
            raise ValueError("GSM8K subset must be 'main' or 'socratic'")
        if split not in {"train", "test"}:
            raise ValueError("GSM8K split must be 'train' or 'test'")

        ds = load_dataset("openai/gsm8k", subset, split=split)
        if shuffle:
            ds = ds.shuffle(seed=seed)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.get_example(idx)

    def num_examples(self) -> int:
        return len(self.ds)

    def get_example(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
        question = row["question"]
        answer = row["answer"]
        final_answer = extract_answer(answer)
        if final_answer is None:
            raise ValueError(f"Could not extract GSM8K answer at index {index}")

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        return {
            "messages": messages,
            "answer": final_answer,
        }

    def evaluate(self, conversation: dict[str, Any], assistant_response: str) -> bool:
        if not isinstance(assistant_response, str):
            raise TypeError("assistant_response must be a string")

        ref_num = conversation.get("answer")
        if ref_num is None:
            assistant_message = conversation["messages"][-1]
            ref_num = extract_answer(assistant_message["content"])
        pred_num = extract_answer(assistant_response)
        return pred_num is not None and pred_num == ref_num

    def reward(self, conversation: dict[str, Any], assistant_response: str) -> float:
        return float(self.evaluate(conversation, assistant_response))
