"""Natural language inference on the test split of ``uitnlp/ViANLI``."""

from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset


DATASET_NAME = "uitnlp/ViANLI"
EVAL_SPLIT = "test"
MAX_NEW_TOKENS = 16
LABELS = ("entailment", "neutral", "contradiction")


def _render_prompt(premise: str, hypothesis: str) -> str:
    return (
        "Hãy xác định mối quan hệ giữa tiền đề và giả thuyết sau.\n\n"
        f"Tiền đề: {premise}\n\n"
        f"Giả thuyết: {hypothesis}\n\n"
        "Chọn một trong ba nhãn:\n"
        "- entailment: giả thuyết chắc chắn đúng dựa trên tiền đề.\n"
        "- contradiction: giả thuyết mâu thuẫn với tiền đề.\n"
        "- neutral: tiền đề không đủ để kết luận giả thuyết đúng hay sai.\n\n"
        "Chỉ trả lời bằng đúng một nhãn, không giải thích."
    )


class ViANLI:
    """Answer-only three-way NLI evaluation fixed to the official test split."""

    eval_type = "generative_classification"
    max_new_tokens = MAX_NEW_TOKENS
    primary_metric = "accuracy"
    labels = LABELS
    split = EVAL_SPLIT

    def __init__(
        self,
        *,
        limit: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")

        ds = load_dataset(DATASET_NAME, split=EVAL_SPLIT)
        if shuffle:
            if hasattr(ds, "shuffle"):
                ds = ds.shuffle(seed=seed)
            else:
                ds = list(ds)
                random.Random(seed).shuffle(ds)
        if limit is not None:
            if hasattr(ds, "select"):
                ds = ds.select(range(min(limit, len(ds))))
            else:
                ds = ds[:limit]
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.get_example(idx)

    def num_examples(self) -> int:
        return len(self.ds)

    def get_example(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
        uid = row["uid"]
        premise = row["premise"]
        hypothesis = row["hypothesis"]
        label = row["label"]

        for field_name, value in (("premise", premise), ("hypothesis", hypothesis)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"ViANLI example at index {index} has an invalid {field_name}"
                )
        if label not in self.labels:
            raise ValueError(f"ViANLI example at index {index} has unknown label {label!r}")

        return {
            "messages": [
                {"role": "user", "content": _render_prompt(premise, hypothesis)},
                {"role": "assistant", "content": label},
            ],
            "uid": uid,
            "premise": premise,
            "hypothesis": hypothesis,
            "answer": label,
            "label": label,
            "split": self.split,
        }

    def evaluate(self, conversation: dict[str, Any], assistant_response: str) -> bool:
        if not isinstance(assistant_response, str):
            return False
        return assistant_response.strip() == conversation["answer"]

    def reward(self, conversation: dict[str, Any], assistant_response: str) -> float:
        return float(self.evaluate(conversation, assistant_response))
