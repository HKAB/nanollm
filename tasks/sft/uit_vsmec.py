"""Emotion classification on the test split of ``tridm/UIT-VSMEC``."""

from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset


DATASET_NAME = "tridm/UIT-VSMEC"
EVAL_SPLIT = "test"
MAX_NEW_TOKENS = 16
EMOTIONS = (
    "Anger",
    "Disgust",
    "Enjoyment",
    "Fear",
    "Other",
    "Sadness",
    "Surprise",
)


def _render_prompt(sentence: str) -> str:
    labels = ", ".join(EMOTIONS)
    return (
        "Hãy phân loại cảm xúc được thể hiện trong câu tiếng Việt sau.\n\n"
        f"Câu: {sentence}\n\n"
        f"Nhãn cảm xúc hợp lệ: {labels}.\n"
        "Chỉ trả lời bằng đúng một nhãn cảm xúc, không giải thích."
    )


class UITVSMEC:
    """Answer-only UIT-VSMEC evaluation task.

    The Hugging Face split is deliberately fixed to ``test``.  This class is
    an evaluation task and cannot accidentally load the train or validation
    examples.
    """

    eval_type = "generative_classification"
    max_new_tokens = MAX_NEW_TOKENS
    primary_metric = "macro_f1"
    labels = EMOTIONS
    emotions = EMOTIONS
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
        sentence = row["Sentence"]
        emotion = row["Emotion"]
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError(f"UIT-VSMEC example at index {index} has an invalid Sentence")
        if emotion not in self.emotions:
            raise ValueError(
                f"UIT-VSMEC example at index {index} has unknown Emotion {emotion!r}"
            )

        return {
            "messages": [
                {"role": "user", "content": _render_prompt(sentence)},
                {"role": "assistant", "content": emotion},
            ],
            "sentence": sentence,
            "answer": emotion,
            "emotion": emotion,
            "split": self.split,
        }

    def evaluate(self, conversation: dict[str, Any], assistant_response: str) -> bool:
        if not isinstance(assistant_response, str):
            return False
        # Preserve exact label spelling while allowing generation whitespace.
        return assistant_response.strip() == conversation["answer"]

    def reward(self, conversation: dict[str, Any], assistant_response: str) -> float:
        return float(self.evaluate(conversation, assistant_response))
