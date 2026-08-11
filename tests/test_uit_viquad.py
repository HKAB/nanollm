from tasks import uit_viquad


class FakeDataset(list):
    def filter(self, fn):
        return FakeDataset(row for row in self if fn(row))

    def select(self, indices):
        return FakeDataset(self[index] for index in indices)

    def shuffle(self, seed):
        import random

        result = FakeDataset(self)
        random.Random(seed).shuffle(result)
        return result


ROWS = FakeDataset(
    [
        {
            "id": "answerable",
            "uit_id": "uit_1",
            "title": "Tiếng Anh",
            "context": "Tiếng Anh có bảy lớp từ chính.",
            "question": "Tiếng Anh có bao nhiêu lớp từ chính?",
            "answers": {"text": ["bảy"], "answer_start": [14]},
            "is_impossible": False,
            "plausible_answers": {"text": [], "answer_start": []},
        },
        {
            "id": "impossible",
            "uit_id": "uit_2",
            "title": "Tiếng Anh",
            "context": "Tiếng Anh có bảy lớp từ chính.",
            "question": "Ngôn ngữ Ấn-Âu có bao nhiêu lớp từ?",
            "answers": {"text": [], "answer_start": []},
            "is_impossible": True,
            "plausible_answers": {"text": ["bảy"], "answer_start": [14]},
        },
    ]
)


def _patch_dataset(monkeypatch):
    monkeypatch.setattr(uit_viquad, "load_dataset", lambda *args, **kwargs: FakeDataset(ROWS))


def test_hallucination_task_answers_or_refuses(monkeypatch):
    _patch_dataset(monkeypatch)
    task = uit_viquad.UITViQuADHallucination(split="train")

    assert task[0]["answer"] == "bảy"
    assert task[1]["answer"] == uit_viquad.INSUFFICIENT_CONTEXT_RESPONSE
    assert task[1]["plausible_answers"] == ["bảy"]
    assert uit_viquad.INSUFFICIENT_CONTEXT_RESPONSE in task[1]["messages"][0]["content"]

    assert task.evaluate(task[0], "  BẢY  ")
    assert not task.evaluate(task[0], "Có bảy lớp từ.")
    assert task.evaluate(task[1], f" {uit_viquad.INSUFFICIENT_CONTEXT_RESPONSE} ")


def test_non_refusal_on_impossible_question_is_hallucination(monkeypatch):
    _patch_dataset(monkeypatch)
    task = uit_viquad.UITViQuADHallucination()
    example = task[1]

    result = task.evaluate_details(example, "bảy")
    assert not result.correct
    assert result.is_impossible
    assert result.hallucinated
    assert not task.evaluate(example, "Không đủ thông tin.")


def test_old_answerability_name_is_compatibility_alias():
    assert uit_viquad.UITViQuADAnswerability is uit_viquad.UITViQuADHallucination


def test_hallucination_runner_reports_balanced_metrics(monkeypatch):
    import torch

    from scripts import chat_eval

    _patch_dataset(monkeypatch)
    task = uit_viquad.UITViQuADHallucination()

    class FakeModel:
        def get_device(self):
            return torch.device("cpu")

    class FakeTokenizer:
        responses = {
            7: "bảy",
            8: uit_viquad.INSUFFICIENT_CONTEXT_RESPONSE,
        }

        def render_for_completion(self, conversation, *, enable_thinking):
            assert not enable_thinking
            return [1]

        def decode(self, tokens):
            return self.responses[tokens[0]]

    class FakeEngine:
        def __init__(self):
            self.responses = iter([7, 8])

        def generate_batch(self, prompt, **kwargs):
            return [[*prompt, next(self.responses)]], None

    monkeypatch.setattr(chat_eval, "get_dist_info", lambda: (False, 0, 0, 1))
    monkeypatch.setattr(chat_eval, "print0", lambda *args, **kwargs: None)
    score = chat_eval.run_hallucination_eval(
        task,
        FakeTokenizer(),
        FakeModel(),
        FakeEngine(),
        num_samples=1,
        max_new_tokens=64,
        temperature=0.0,
        top_k=50,
    )

    assert score == 1.0
    assert task.metrics == {
        "overall_accuracy": 1.0,
        "answerable_accuracy": 1.0,
        "refusal_accuracy": 1.0,
        "hallucination_rate": 0.0,
    }


def test_qa_filters_impossible_examples_and_scores_gold_variants(monkeypatch):
    _patch_dataset(monkeypatch)
    task = uit_viquad.UITViQuADQA(split="validation")

    assert len(task) == 1
    assert task[0]["messages"][-1]["content"] == "bảy"
    assert task.evaluate(task[0], "  BẢY  ")
    assert not task.evaluate(task[0], "Có bảy lớp từ.")


def test_limit_is_applied_after_qa_filter(monkeypatch):
    _patch_dataset(monkeypatch)
    task = uit_viquad.UITViQuADQA(limit=1)
    assert len(task) == 1
