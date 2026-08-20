"""Vietnamese causal commonsense reasoning from AI Singapore's NLR suite."""

from __future__ import annotations

import random
from typing import Any

from datasets import load_dataset


DATASET_NAME = "aisingapore/NLR-Causal-Reasoning"
LANGUAGE_CONFIG = "vi"
EVAL_SPLIT = "eval"
MAX_NEW_TOKENS = 8
LABELS = ("A", "B")
QUESTIONS = ("nguyên nhân", "kết quả")


def _render_prompt(
    premise: str,
    choice1: str,
    choice2: str,
    question: str,
) -> str:
    if question == "nguyên nhân":
        instruction = (
            "Sự việc nào dưới đây có khả năng là nguyên nhân dẫn đến sự việc "
            "được nêu?"
        )
    else:
        instruction = (
            "Sự việc nào dưới đây có khả năng là kết quả xảy ra do sự việc "
            "được nêu?"
        )
    return (
        "Hãy suy luận mối quan hệ nhân quả.\n\n"
        f"Sự việc: {premise}\n\n"
        f"Câu hỏi ({question}): {instruction}\n\n"
        f"A. {choice1}\n"
        f"B. {choice2}\n\n"
        "Chỉ trả lời bằng một chữ cái: A hoặc B."
    )


class NLRCausalReasoningVI:
    """Two-choice causal reasoning on the Vietnamese evaluation split."""

    eval_type = "categorical"
    max_new_tokens = MAX_NEW_TOKENS
    letters = LABELS
    language_config = LANGUAGE_CONFIG
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

        ds = load_dataset(DATASET_NAME, LANGUAGE_CONFIG, split=EVAL_SPLIT)
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
        prompts = row["prompts"]
        if not isinstance(prompts, (list, tuple)) or len(prompts) != 1:
            raise ValueError(
                f"NLR causal example at index {index} must contain exactly one prompt"
            )
        prompt_data = prompts[0]
        if not isinstance(prompt_data, dict):
            raise ValueError(f"NLR causal example at index {index} has an invalid prompt")

        premise = prompt_data.get("premise")
        choice1 = prompt_data.get("choice1")
        choice2 = prompt_data.get("choice2")
        question = prompt_data.get("question_translated")
        for field_name, value in (
            ("premise", premise),
            ("choice1", choice1),
            ("choice2", choice2),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"NLR causal example at index {index} has an invalid {field_name}"
                )
        if question not in QUESTIONS:
            raise ValueError(
                f"NLR causal example at index {index} has unknown question type {question!r}"
            )

        answer = row["label"]
        if answer not in self.letters:
            raise ValueError(
                f"NLR causal example at index {index} has unknown label {answer!r}"
            )

        return {
            "messages": [
                {
                    "role": "user",
                    "content": _render_prompt(premise, choice1, choice2, question),
                },
                {"role": "assistant", "content": answer},
            ],
            "id": row.get("id"),
            "premise": premise,
            "choices": [choice1, choice2],
            "question_type": question,
            "letters": self.letters,
            "answer": answer,
            "language_config": self.language_config,
            "split": self.split,
        }

    def evaluate(self, conversation: dict[str, Any], assistant_response: str) -> bool:
        if not isinstance(assistant_response, str):
            return False
        return assistant_response.strip() == conversation["answer"]

    def reward(self, conversation: dict[str, Any], assistant_response: str) -> float:
        return float(self.evaluate(conversation, assistant_response))
