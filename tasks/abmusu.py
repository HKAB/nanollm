"""Vietnamese abstractive multi-document summarization (VLSP 2022 AbMusu)."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from datasets import load_dataset


DATASET_NAME = "truongnp5/abmusu"
EVAL_SPLIT = "validation"
SPLITS = {"train", "validation", "test"}
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class Rouge2Score:
    precision: float
    recall: float
    f1: float


def _word_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return _WORD_RE.findall(normalized)


def rouge2_score(prediction: str, reference: str) -> Rouge2Score:
    """Calculate multiset ROUGE-2 with Unicode-aware word tokenization."""

    if not isinstance(prediction, str) or not isinstance(reference, str):
        return Rouge2Score(0.0, 0.0, 0.0)
    prediction_tokens = _word_tokens(prediction)
    reference_tokens = _word_tokens(reference)
    prediction_bigrams = Counter(zip(prediction_tokens, prediction_tokens[1:]))
    reference_bigrams = Counter(zip(reference_tokens, reference_tokens[1:]))
    prediction_total = sum(prediction_bigrams.values())
    reference_total = sum(reference_bigrams.values())
    if prediction_total == 0 or reference_total == 0:
        return Rouge2Score(0.0, 0.0, 0.0)

    matched = sum((prediction_bigrams & reference_bigrams).values())
    precision = matched / prediction_total
    recall = matched / reference_total
    denominator = precision + recall
    f1 = 0.0 if denominator == 0.0 else 2 * precision * recall / denominator
    return Rouge2Score(precision, recall, f1)


def aggregate_rouge2(scores: Sequence[Rouge2Score]) -> dict[str, float]:
    """Macro-average ROUGE-2 over document clusters."""

    if not scores:
        raise ValueError("Cannot aggregate an empty AbMusu result set")
    count = len(scores)
    return {
        "rouge2_precision": sum(score.precision for score in scores) / count,
        "rouge2_recall": sum(score.recall for score in scores) / count,
        "rouge2_f1": sum(score.f1 for score in scores) / count,
    }


def _render_prompt(documents: Sequence[Mapping[str, str]]) -> str:
    overview_sections = []
    document_sections = []
    for index, document in enumerate(documents, start=1):
        title = document.get("title", "").strip()
        anchor = document.get("anchor_text", "").strip()
        raw_text = document.get("raw_text", "").strip()
        overview_sections.append(
            f"Tài liệu {index}:\nTiêu đề: {title}\nThông tin mở đầu: {anchor}"
        )
        document_sections.append(f"### Tài liệu {index}: {title}\n{raw_text}")

    # Put every document's title/lead first. If the model context truncates the
    # full articles, it still sees a balanced overview of the complete cluster.
    overview = "\n\n".join(overview_sections)
    full_text = "\n\n".join(document_sections)
    return (
        "Hãy viết một bản tóm tắt trừu tượng bằng tiếng Việt cho nhóm bài báo "
        "cùng chủ đề dưới đây. Tổng hợp các thông tin quan trọng từ tất cả tài liệu, "
        "loại bỏ chi tiết trùng lặp và không thêm thông tin không có trong nguồn. "
        "Chỉ trả lời bằng bản tóm tắt, không giải thích.\n\n"
        f"## Tổng quan các tài liệu\n{overview}\n\n"
        f"## Nội dung đầy đủ\n{full_text}"
    )


class AbMusu:
    """AbMusu evaluation on validation, the only official labeled eval split."""

    eval_type = "summarization"

    def __init__(
        self,
        *,
        dataset_name: str = DATASET_NAME,
        split: str = EVAL_SPLIT,
        dataset: Any | None = None,
        limit: int | None = None,
    ):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {sorted(SPLITS)}, got {split!r}")
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative or None")
        ds = dataset if dataset is not None else load_dataset(dataset_name, split=split)
        if limit is not None:
            if hasattr(ds, "select"):
                ds = ds.select(range(min(limit, len(ds))))
            else:
                ds = ds[:limit]
        self.dataset_name = dataset_name
        self.split = split
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.ds[index])
        documents = row.get("single_documents")
        if not isinstance(documents, (list, tuple)) or not documents:
            raise ValueError(f"AbMusu example at index {index} has no documents")
        summary = row.get("summary")
        if self.split != "test" and (
            not isinstance(summary, str) or not summary.strip()
        ):
            raise ValueError(f"AbMusu example at index {index} has no reference summary")
        row["messages"] = [
            {"role": "user", "content": _render_prompt(documents)},
            {"role": "assistant", "content": summary or ""},
        ]
        row["reference_summary"] = summary
        return row

    def evaluate_details(
        self, conversation: Mapping[str, Any], assistant_response: str
    ) -> Rouge2Score:
        reference = conversation.get("reference_summary")
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError("ROUGE evaluation requires a reference summary")
        return rouge2_score(assistant_response, reference)

    def evaluate(
        self, conversation: Mapping[str, Any], assistant_response: str
    ) -> float:
        return self.evaluate_details(conversation, assistant_response).f1

    def reward(self, conversation: Mapping[str, Any], assistant_response: str) -> float:
        return self.evaluate(conversation, assistant_response)
