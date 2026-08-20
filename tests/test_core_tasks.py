import pytest

from nanollm.core_eval import render_prompts_mc
from tasks.pretrain import core


def test_core_manifest_uses_newline_delimiter_and_unique_labels():
    labels = [task.label for task in core.CORE_TASKS]

    assert len(labels) == len(set(labels))
    assert all(task.continuation_delimiter == "\nĐáp án: " for task in core.CORE_TASKS)


def test_core_multiple_choice_prompt_puts_answer_on_newline():
    prompts = render_prompts_mc(
        {"query": "Câu hỏi?", "choices": ["A", "B"], "gold": 0},
        "\nĐáp án: ",
    )

    assert prompts == ["Câu hỏi?\nĐáp án: A", "Câu hỏi?\nĐáp án: B"]


def test_load_core_data_groups_and_normalizes_rows(monkeypatch):
    rows = [
        {"task": task.label, "query": f"Question {index}", "choices": ["A", "B", "C", "D"], "gold": index % 4}
        for index, task in enumerate(core.CORE_TASKS)
    ]
    calls = []
    monkeypatch.setenv("NANOLLM_CORE_DATASET", "owner/core-vi")
    monkeypatch.setenv("NANOLLM_CORE_DATASET_REVISION", "abc123")
    monkeypatch.setattr(core, "load_dataset", lambda **kwargs: calls.append(kwargs) or rows)

    grouped = core.load_core_data()

    assert calls == [{"path": "owner/core-vi", "split": "test", "revision": "abc123"}]
    assert set(grouped) == {task.label for task in core.CORE_TASKS}
    assert grouped["global_mmlu"][0] == {
        "query": "Question 0",
        "choices": ["A", "B", "C", "D"],
        "gold": 0,
    }


def test_load_core_data_requires_repository(monkeypatch):
    monkeypatch.delenv("NANOLLM_CORE_DATASET", raising=False)

    with pytest.raises(RuntimeError, match="NANOLLM_CORE_DATASET"):
        core.load_core_data()


def test_load_core_data_rejects_missing_task(monkeypatch):
    monkeypatch.setenv("NANOLLM_CORE_DATASET", "owner/core-vi")
    monkeypatch.delenv("NANOLLM_CORE_DATASET_REVISION", raising=False)
    monkeypatch.setattr(
        core,
        "load_dataset",
        lambda **kwargs: [
            {"task": "global_mmlu", "query": "Question", "choices": ["A", "B"], "gold": 0}
        ],
    )

    with pytest.raises(ValueError, match="no rows for tasks"):
        core.load_core_data()


def test_load_core_data_keeps_future_schema_and_language_modeling_rows(monkeypatch):
    tasks = (
        core.CoreTask("schema", "schema", 0, " ", 0.5),
        core.CoreTask("lm", "language_modeling", 0, " ", 0.0),
    )
    rows = [
        {
            "task": "schema",
            "context_options": ["one", "two"],
            "continuation": "continues",
            "gold": 1,
        },
        {"task": "lm", "context": "prefix", "continuation": "suffix"},
    ]
    monkeypatch.setattr(core, "CORE_TASKS", tasks)
    monkeypatch.setenv("NANOLLM_CORE_DATASET", "owner/core-vi")
    monkeypatch.setattr(core, "load_dataset", lambda **kwargs: rows)

    grouped = core.load_core_data()

    assert grouped["schema"] == [{
        "context_options": ["one", "two"],
        "continuation": "continues",
        "gold": 1,
    }]
    assert grouped["lm"] == [{"context": "prefix", "continuation": "suffix"}]
