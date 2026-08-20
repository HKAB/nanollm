"""Chat evaluation adapter for the normalized Vietnamese Global-MMLU task."""

from __future__ import annotations

import random
from typing import Any

from tasks.pretrain.core import load_core_data

_LABELS = ("A", "B", "C", "D")
MAX_NEW_TOKENS = 8


def _render_mc(question: str, letters: tuple[str, ...], choices: list[str]) -> str:
    options = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices))
    return (
        f"{question}\n\n{options}\n\n"
        "Hãy chọn đáp án đúng. Chỉ trả lời bằng một chữ cái: A, B, C hoặc D."
    )


class GlobalMMLU:
    """Global-MMLU task compatible with categorical chat evaluation."""
    eval_type = "categorical"
    max_new_tokens = MAX_NEW_TOKENS
    letters = _LABELS

    def __init__(
        self,
        *,
        limit: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.ds = load_core_data()["global_mmlu"]
        if shuffle:
            rnd = random.Random(seed)
            rnd.shuffle(self.ds)
        if limit is not None:
            self.ds = self.ds[:limit]

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.get_example(idx)

    def num_examples(self) -> int:
        return len(self.ds)

    def get_example(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
        answer_letter = self.letters[row["gold"]]
        user_message = _render_mc(row["query"], self.letters, row["choices"])
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_letter},
        ]
        return {
            "messages": messages,
            "letters": self.letters,
            "answer": answer_letter,
        }

    def evaluate(self, conversation: dict[str, Any], assistant_response: str) -> bool:
        assert assistant_response in self.letters, (
            f"assistant_response must be one of {self.letters}, got {assistant_response!r}"
        )
        return assistant_response == conversation["messages"][-1]["content"]
