"""Sentiment classification on the UIT-VSFC test split."""

from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset


DATASET_NAME = "uitnlp/vietnamese_students_feedback"
EVAL_SPLIT = "test"
SENTIMENTS = ("negative", "neutral", "positive")


def _render_prompt(sentence: str) -> str:
    return (
        "Hãy phân loại cảm xúc của phản hồi sinh viên sau.\n\n"
        f"Phản hồi: {sentence}\n\n"
        "Nhãn hợp lệ: negative, neutral, positive.\n"
        "Chỉ trả lời bằng đúng một nhãn, không giải thích."
    )


class UITVSFCSentiment:
    """Answer-only sentiment task fixed to the official test split.

    The dataset's ``topic`` field is deliberately ignored: this task uses only
    ``sentence`` as input and ``sentiment`` as its target.
    """

    eval_type = "generative_classification"
    primary_metric = "macro_f1"
    labels = SENTIMENTS
    sentiments = SENTIMENTS
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
        sentence = row["sentence"]
        sentiment_id = row["sentiment"]
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(f"UIT-VSFC example at index {index} has an invalid sentence")
        if isinstance(sentiment_id, bool) or not isinstance(sentiment_id, int):
            raise ValueError(
                f"UIT-VSFC example at index {index} has invalid sentiment {sentiment_id!r}"
            )
        if not 0 <= sentiment_id < len(self.sentiments):
            raise ValueError(
                f"UIT-VSFC example at index {index} has unknown sentiment {sentiment_id!r}"
            )
        answer = self.sentiments[sentiment_id]

        return {
            "messages": [
                {"role": "user", "content": _render_prompt(sentence)},
                {"role": "assistant", "content": answer},
            ],
            "sentence": sentence,
            "sentiment": sentiment_id,
            "answer": answer,
            "split": self.split,
        }

    def evaluate(self, conversation: dict[str, Any], assistant_response: str) -> bool:
        if not isinstance(assistant_response, str):
            return False
        return assistant_response.strip() == conversation["answer"]

    def reward(self, conversation: dict[str, Any], assistant_response: str) -> float:
        return float(self.evaluate(conversation, assistant_response))
