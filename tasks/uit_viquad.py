"""SFT/evaluation tasks backed by ``taidng/UIT-ViQuAD2.0``.

Two complementary tasks are provided:

* :class:`UITViQuADHallucination` asks the model to answer only from the
  supplied context and requires an exact refusal when the context is
  insufficient. ``UITViQuADAnswerability`` remains as a compatibility alias.
* :class:`UITViQuADQA` is extractive QA over answerable examples only.  Its
  target is an answer span with no explanation or surrounding prose.

Both classes expose examples in the conversation format used by the other
tasks in this repository, so the returned ``messages`` can also be serialized
directly into an SFT dataset.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from datasets import load_dataset


DATASET_NAME = "taidng/UIT-ViQuAD2.0"
_SPLITS = {"train", "validation", "test"}
INSUFFICIENT_CONTEXT_RESPONSE = "Không đủ thông tin để trả lời."


@dataclass(frozen=True)
class HallucinationResult:
    correct: bool
    is_impossible: bool
    hallucinated: bool
    refused: bool


@dataclass(frozen=True)
class ExtractiveQAScore:
    exact_match: bool
    f1: float


def _answer_texts(answers: Any) -> list[str]:
    """Return non-empty answer texts for either common HF Sequence shape."""
    if not answers:
        return []
    if isinstance(answers, dict):
        texts = answers.get("text", [])
    elif isinstance(answers, (list, tuple)):
        texts = [item.get("text") for item in answers if isinstance(item, dict)]
    else:
        return []
    if isinstance(texts, str):
        texts = [texts]
    return [text.strip() for text in texts if isinstance(text, str) and text.strip()]


def _normalize_text(text: str) -> str:
    """Normalize harmless formatting differences for extractive QA scoring."""
    text = unicodedata.normalize("NFC", text).strip()
    return re.sub(r"\s+", " ", text).casefold()


def _qa_tokens(text: str) -> list[str]:
    """Vietnamese-safe SQuAD-style tokens for EM/token-F1."""

    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S"} else character
        for character in normalized
    )
    return normalized.split()


def _score_qa_answer(prediction: str, reference: str) -> ExtractiveQAScore:
    prediction_tokens = _qa_tokens(prediction)
    reference_tokens = _qa_tokens(reference)
    exact_match = prediction_tokens == reference_tokens
    if not prediction_tokens or not reference_tokens:
        return ExtractiveQAScore(exact_match, float(exact_match))
    overlap = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return ExtractiveQAScore(exact_match, 0.0)
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return ExtractiveQAScore(
        exact_match,
        2 * precision * recall / (precision + recall),
    )


def _render_prompt(context: str, question: str, instruction: str) -> str:
    return f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}\n\n{instruction}"


class _UITViQuADBase:
    eval_type = "generative"

    def __init__(
        self,
        split: str = "train",
        *,
        limit: int | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {sorted(_SPLITS)}, got {split!r}")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")

        ds = load_dataset(DATASET_NAME, split=split)
        self.ds = self._prepare_dataset(ds)
        if shuffle:
            if hasattr(self.ds, "shuffle"):
                self.ds = self.ds.shuffle(seed=seed)
            else:
                self.ds = list(self.ds)
                random.Random(seed).shuffle(self.ds)
        if limit is not None:
            if hasattr(self.ds, "select"):
                self.ds = self.ds.select(range(min(limit, len(self.ds))))
            else:
                self.ds = self.ds[:limit]

    def _prepare_dataset(self, ds: Any) -> Any:
        return ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.get_example(idx)

    def num_examples(self) -> int:
        return len(self.ds)

    @staticmethod
    def _metadata(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row.get("id"),
            "uit_id": row.get("uit_id"),
            "title": row.get("title"),
        }

    def reward(self, conversation: dict[str, Any], assistant_response: str) -> float:
        return float(self.evaluate(conversation, assistant_response))


class UITViQuADHallucination(_UITViQuADBase):
    """Answer from context or emit the exact refusal instead of hallucinating."""

    eval_type = "hallucination"

    def get_example(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
        is_impossible = bool(row["is_impossible"])
        answers = _answer_texts(row.get("answers"))
        if not is_impossible and not answers:
            raise ValueError(f"Answerable example at index {index} has no gold answer")
        answer = INSUFFICIENT_CONTEXT_RESPONSE if is_impossible else answers[0]
        prompt = _render_prompt(
            row["context"],
            row["question"],
            "Chỉ trả lời câu hỏi dựa trên thông tin có trong ngữ cảnh. "
            "Nếu ngữ cảnh không cung cấp đủ thông tin, hãy trả lời chính xác: "
            f'\"{INSUFFICIENT_CONTEXT_RESPONSE}\" '
            "Nếu có đủ thông tin, chỉ trả lời bằng đoạn trích ngắn nhất chứa đáp án. "
            "Không giải thích và không sử dụng kiến thức bên ngoài.",
        )
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ],
            "answer": answer,
            "answers": answers,
            "is_impossible": is_impossible,
            "plausible_answers": _answer_texts(row.get("plausible_answers")),
            **self._metadata(row),
        }

    def evaluate_details(
        self, conversation: dict[str, Any], assistant_response: str
    ) -> HallucinationResult:
        is_impossible = bool(conversation["is_impossible"])
        if not isinstance(assistant_response, str):
            return HallucinationResult(False, is_impossible, is_impossible, False)

        if is_impossible:
            # Only the instructed refusal is accepted. Plausible answers are
            # deliberately not gold answers for impossible questions.
            refused = assistant_response.strip() == INSUFFICIENT_CONTEXT_RESPONSE
            return HallucinationResult(refused, True, not refused, refused)

        prediction = _normalize_text(assistant_response)
        refused = assistant_response.strip() == INSUFFICIENT_CONTEXT_RESPONSE
        correct = bool(prediction) and prediction in {
            _normalize_text(answer) for answer in conversation["answers"]
        }
        return HallucinationResult(correct, False, False, refused)

    def evaluate(self, conversation: dict[str, Any], assistant_response: str) -> bool:
        return self.evaluate_details(conversation, assistant_response).correct


# Preserve existing imports while changing the benchmark semantics in place.
UITViQuADAnswerability = UITViQuADHallucination


class UITViQuADQA(_UITViQuADBase):
    """Answer-only extractive QA using the answerable ViQuAD examples."""

    eval_type = "extractive_qa"

    def _prepare_dataset(self, ds: Any) -> Any:
        if hasattr(ds, "filter"):
            return ds.filter(lambda row: not bool(row["is_impossible"]))
        return [row for row in ds if not bool(row["is_impossible"])]

    def get_example(self, index: int) -> dict[str, Any]:
        row = self.ds[index]
        answers = _answer_texts(row.get("answers"))
        if not answers:
            raise ValueError(f"Answerable example at index {index} has no gold answer")
        prompt = _render_prompt(
            row["context"],
            row["question"],
            "Trả lời câu hỏi bằng một đoạn trích ngắn từ ngữ cảnh. "
            "Chỉ trả lời đáp án, không giải thích.",
        )
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answers[0]},
            ],
            "answer": answers[0],
            "answers": answers,
            "is_impossible": False,
            **self._metadata(row),
        }

    def evaluate_details(
        self, conversation: dict[str, Any], assistant_response: str
    ) -> ExtractiveQAScore:
        if not isinstance(assistant_response, str):
            return ExtractiveQAScore(False, 0.0)
        scores = [
            _score_qa_answer(assistant_response, answer)
            for answer in conversation["answers"]
        ]
        return ExtractiveQAScore(
            any(score.exact_match for score in scores),
            max((score.f1 for score in scores), default=0.0),
        )

    def evaluate(self, conversation: dict[str, Any], assistant_response: str) -> bool:
        return self.evaluate_details(conversation, assistant_response).exact_match


def iter_sft_messages(task: _UITViQuADBase) -> Iterable[list[dict[str, str]]]:
    """Yield just the ``messages`` payloads for parquet/JSONL conversion."""
    for index in range(len(task)):
        yield task[index]["messages"]
