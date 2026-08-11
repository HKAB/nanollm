import pytest
import torch

from scripts import chat_eval


class FakeModel:
    def get_device(self):
        return torch.device("cpu")


class FakeTokenizer:
    def __init__(self, responses):
        self.responses = responses

    def render_for_completion(self, conversation, *, enable_thinking):
        assert not enable_thinking
        return [1]

    def decode(self, tokens):
        return self.responses[tokens[0]]


class FakeEngine:
    def __init__(self, response_ids):
        self.response_ids = iter(response_ids)

    def generate_batch(self, prompt, **kwargs):
        return [[*prompt, next(self.response_ids)]], None


class FakeClassificationTask:
    labels = ("A", "B")
    primary_metric = "macro_f1"

    def __init__(self):
        self.rows = [{"answer": "A"}, {"answer": "A"}, {"answer": "B"}]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def test_generated_classification_reports_accuracy_and_macro_f1(monkeypatch):
    monkeypatch.setattr(chat_eval, "get_dist_info", lambda: (False, 0, 0, 1))
    monkeypatch.setattr(chat_eval, "print0", lambda *args, **kwargs: None)
    task = FakeClassificationTask()
    score = chat_eval.run_generative_classification_eval(
        task,
        FakeTokenizer({10: "A", 11: "B", 12: "invalid"}),
        FakeModel(),
        FakeEngine([10, 12, 11]),
        num_samples=1,
        max_new_tokens=16,
        temperature=0.0,
        top_k=50,
    )

    assert task.metrics["accuracy"] == pytest.approx(2 / 3)
    assert task.metrics["macro_f1"] == pytest.approx(5 / 6)
    assert score == task.metrics["macro_f1"]
