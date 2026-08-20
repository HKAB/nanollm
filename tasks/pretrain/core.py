"""Task definitions and data loading for the Vietnamese CORE benchmark."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from datasets import load_dataset


CoreTaskType = Literal["multiple_choice", "schema", "language_modeling"]


@dataclass(frozen=True)
class CoreTask:
    label: str
    task_type: CoreTaskType
    num_fewshot: int
    continuation_delimiter: str
    random_baseline: float


CORE_TASKS = (
    CoreTask("global_mmlu", "multiple_choice", 0, "\nĐáp án: ", 0.25),
    CoreTask("wikipediaqa_vi", "multiple_choice", 0, "\nĐáp án: ", 0.25),
    CoreTask("dia_ly", "multiple_choice", 0, "\nĐáp án: ", 0.25),
    CoreTask("hoa_hoc", "multiple_choice", 0, "\nĐáp án: ", 0.25),
    CoreTask("lich_su", "multiple_choice", 0, "\nĐáp án: ", 0.25),
    CoreTask("sinh_hoc", "multiple_choice", 0, "\nĐáp án: ", 0.25),
    CoreTask("toan", "multiple_choice", 0, "\nĐáp án: ", 0.25),
    CoreTask("van", "multiple_choice", 0, "\nĐáp án: ", 0.25),
    CoreTask("vat_ly", "multiple_choice", 0, "\nĐáp án: ", 0.25),
)


def load_core_data() -> dict[str, list[dict]]:
    """Load the normalized CORE test split and group rows by logical task."""
    dataset_name = os.environ.get("NANOLLM_CORE_DATASET")
    if not dataset_name:
        raise RuntimeError(
            "NANOLLM_CORE_DATASET must name the normalized Hugging Face "
            "CORE dataset repository"
        )

    revision = os.environ.get("NANOLLM_CORE_DATASET_REVISION")
    load_kwargs = {"path": dataset_name, "split": "test"}
    if revision:
        load_kwargs["revision"] = revision
    dataset = load_dataset(**load_kwargs)

    tasks_by_label = {task.label: task for task in CORE_TASKS}
    grouped = {task.label: [] for task in CORE_TASKS}
    for index, row in enumerate(dataset):
        label = row.get("task")
        if label not in tasks_by_label:
            raise ValueError(f"CORE row {index} has unknown task label: {label!r}")

        task = tasks_by_label[label]
        gold = row.get("gold")
        if task.task_type == "multiple_choice":
            query = row.get("query")
            choices = row.get("choices")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"CORE row {index} has an invalid query")
            if not isinstance(choices, list) or len(choices) < 2 or not all(
                isinstance(choice, str) and choice for choice in choices
            ):
                raise ValueError(f"CORE row {index} has invalid choices")
            if not isinstance(gold, int) or not 0 <= gold < len(choices):
                raise ValueError(f"CORE row {index} has invalid gold index: {gold!r}")
            item = {"query": query, "choices": choices, "gold": gold}
        elif task.task_type == "schema":
            context_options = row.get("context_options")
            continuation = row.get("continuation")
            if (
                not isinstance(context_options, list)
                or len(context_options) < 2
                or not all(
                    isinstance(context, str) and context
                    for context in context_options
                )
            ):
                raise ValueError(f"CORE row {index} has invalid context options")
            if not isinstance(continuation, str) or not continuation:
                raise ValueError(f"CORE row {index} has an invalid continuation")
            if not isinstance(gold, int) or not 0 <= gold < len(context_options):
                raise ValueError(f"CORE row {index} has invalid gold index: {gold!r}")
            item = {
                "context_options": context_options,
                "continuation": continuation,
                "gold": gold,
            }
        else:
            context = row.get("context")
            continuation = row.get("continuation")
            if not isinstance(context, str) or not context:
                raise ValueError(f"CORE row {index} has an invalid context")
            if not isinstance(continuation, str) or not continuation:
                raise ValueError(f"CORE row {index} has an invalid continuation")
            item = {"context": context, "continuation": continuation}

        grouped[label].append(item)

    missing = [label for label, rows in grouped.items() if not rows]
    if missing:
        raise ValueError(f"CORE dataset has no rows for tasks: {', '.join(missing)}")
    return grouped
