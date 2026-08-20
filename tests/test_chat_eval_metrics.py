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

    def generate_prompts(self, prompts, **kwargs):
        return [[*prompt, next(self.response_ids)] for prompt in prompts]


class FakeClassificationTask:
    labels = ("A", "B")
    primary_metric = "macro_f1"

    def __init__(self):
        self.rows = [{"answer": "A"}, {"answer": "A"}, {"answer": "B"}]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class FakeCategoricalTask:
    eval_type = "categorical"
    max_new_tokens = 1

    def __init__(self):
        self.rows = [
            {"messages": [], "letters": ("A", "B"), "answer": "A"},
            {"messages": [], "letters": ("A", "B"), "answer": "B"},
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def evaluate(self, conversation, prediction):
        return prediction == conversation["answer"]


def test_categorical_eval_projects_only_answer_positions(monkeypatch):
    class Tokenizer:
        def token_to_id(self, token):
            return 0

        def render_for_completion(self, conversation):
            return [1, 2] if conversation["answer"] == "A" else [3]

        def encode(self, letter):
            return [{"A": 4, "B": 5}[letter]]

    class Model(FakeModel):
        def __call__(self, prompt_ids, *, logit_positions):
            assert prompt_ids.shape == (2, 2)
            assert logit_positions.tolist() == [1, 0]
            logits = torch.zeros(2, 1, 6)
            logits[0, 0, 4] = 1
            logits[1, 0, 5] = 1
            return logits

    monkeypatch.setattr(chat_eval, "get_dist_info", lambda: (False, 0, 0, 1))

    score = chat_eval.run_categorical_eval(
        FakeCategoricalTask(), Tokenizer(), Model(), batch_size=2
    )

    assert score == 1.0


def test_categorical_eval_uses_exact_packed_forward_when_supported(monkeypatch):
    class Tokenizer:
        def token_to_id(self, token):
            return 0

        def render_for_completion(self, conversation):
            return [1, 2] if conversation["answer"] == "A" else [3]

        def encode(self, letter):
            return [{"A": 4, "B": 5}[letter]]

    class Model(FakeModel):
        def supports_packed_prefill(self):
            return True

        def __call__(self, prompt_ids, *, cu_seqlens, position_ids,
                     logit_positions):
            assert prompt_ids.tolist() == [[1, 2, 3]]
            assert cu_seqlens.tolist() == [0, 2, 3]
            assert position_ids.tolist() == [[0, 1, 0]]
            assert logit_positions.tolist() == [1, 2]
            logits = torch.zeros(2, 1, 6)
            logits[0, 0, 4] = 1
            logits[1, 0, 5] = 1
            return logits

    monkeypatch.setattr(chat_eval, "get_dist_info", lambda: (False, 0, 0, 1))
    score = chat_eval.run_categorical_eval(
        FakeCategoricalTask(), Tokenizer(), Model(), batch_size=2
    )
    assert score == 1.0


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


def test_run_chat_eval_dispatches_categorical_with_engine_model(monkeypatch):
    inference_model = FakeModel()

    class Engine:
        model = inference_model

    monkeypatch.setattr(
        chat_eval,
        "create_chat_task",
        lambda *args, **kwargs: FakeCategoricalTask(),
    )
    monkeypatch.setattr(chat_eval, "get_dist_info", lambda: (False, 0, 0, 1))
    monkeypatch.setattr(chat_eval, "print0", lambda *args, **kwargs: None)

    captured = {}

    def fake_categorical(task_object, tokenizer, model, batch_size,
                         max_problems=None):
        captured.update(
            task=task_object,
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
            max_problems=max_problems,
        )
        return 1.0

    monkeypatch.setattr(chat_eval, "run_categorical_eval", fake_categorical)
    training_wrapper = FakeModel()
    tokenizer = object()
    score = chat_eval.run_chat_eval(
        "GlobalMMLU",
        training_wrapper,
        tokenizer,
        Engine(),
        batch_size=7,
        max_problems=2,
    )

    assert score == 1.0
    assert captured["model"] is inference_model
    assert captured["tokenizer"] is tokenizer
    assert captured["batch_size"] == 7
    assert captured["max_problems"] == 2
