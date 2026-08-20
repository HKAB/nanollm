import pytest

from scripts import base_eval
from tasks.pretrain.core import CoreTask


def test_evaluate_core_uses_task_metadata_and_fractional_baseline(monkeypatch):
    task = CoreTask("example", "multiple_choice", 0, "\nĐáp án: ", 0.25)
    rows = [
        {"query": "Q1", "choices": ["A", "B"], "gold": 0},
        {"query": "Q2", "choices": ["A", "B"], "gold": 1},
    ]
    captured = {}

    monkeypatch.setattr(base_eval, "CORE_TASKS", (task,))
    monkeypatch.setattr(base_eval, "load_core_data", lambda: {"example": rows})

    def fake_evaluate_task(model, tokenizer, data, device, task_meta):
        captured.update(task_meta)
        assert len(data) == 1
        return 0.625

    monkeypatch.setattr(base_eval, "evaluate_task", fake_evaluate_task)

    result = base_eval.evaluate_core(None, None, "cpu", max_per_task=1)

    assert captured == {
        "task_type": "multiple_choice",
        "num_fewshot": 0,
        "continuation_delimiter": "\nĐáp án: ",
    }
    assert result["results"] == {"example": 0.625}
    assert result["centered_results"]["example"] == pytest.approx(0.5)
    assert result["core_metric"] == pytest.approx(0.5)
