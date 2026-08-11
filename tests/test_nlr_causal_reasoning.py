from tasks import nlr_causal_reasoning


class FakeDataset(list):
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
            "id": "cause-example",
            "prompts": [
                {
                    "choice1": "Trận đấu phải bước vào phút bù giờ.",
                    "choice2": "Trọng tài ra quyết định sai.",
                    "premise": "Những người hâm mộ la ó.",
                    "question_translated": "nguyên nhân",
                }
            ],
            "label": "B",
        },
        {
            "id": "effect-example",
            "prompts": [
                {
                    "choice1": "Mặt đất trở nên ướt.",
                    "choice2": "Mặt trời sáng hơn.",
                    "premise": "Trời bắt đầu mưa lớn.",
                    "question_translated": "kết quả",
                }
            ],
            "label": "A",
        },
    ]
)


def _patch_dataset(monkeypatch):
    calls = []

    def fake_load_dataset(name, config, *, split):
        calls.append((name, config, split))
        return FakeDataset(ROWS)

    monkeypatch.setattr(nlr_causal_reasoning, "load_dataset", fake_load_dataset)
    return calls


def test_task_loads_only_vietnamese_eval_split(monkeypatch):
    calls = _patch_dataset(monkeypatch)
    task = nlr_causal_reasoning.NLRCausalReasoningVI()

    assert calls == [("aisingapore/NLR-Causal-Reasoning", "vi", "eval")]
    assert task.language_config == "vi"
    assert task.split == "eval"


def test_cause_prompt_and_answer(monkeypatch):
    _patch_dataset(monkeypatch)
    example = nlr_causal_reasoning.NLRCausalReasoningVI(limit=1)[0]
    prompt = example["messages"][0]["content"]

    assert "nguyên nhân dẫn đến" in prompt
    assert "A. Trận đấu phải bước vào phút bù giờ." in prompt
    assert "B. Trọng tài ra quyết định sai." in prompt
    assert example["answer"] == "B"
    assert example["letters"] == ("A", "B")


def test_effect_prompt_and_strict_evaluation(monkeypatch):
    _patch_dataset(monkeypatch)
    task = nlr_causal_reasoning.NLRCausalReasoningVI()
    example = task[1]
    prompt = example["messages"][0]["content"]

    assert "kết quả xảy ra do" in prompt
    assert task.evaluate(example, " A ")
    assert not task.evaluate(example, "A. Mặt đất trở nên ướt.")


def test_unknown_question_type_is_rejected(monkeypatch):
    bad_row = {**ROWS[0], "prompts": [{**ROWS[0]["prompts"][0], "question_translated": "lý do"}]}
    monkeypatch.setattr(
        nlr_causal_reasoning,
        "load_dataset",
        lambda *args, **kwargs: FakeDataset([bad_row]),
    )
    task = nlr_causal_reasoning.NLRCausalReasoningVI()

    try:
        task[0]
    except ValueError as error:
        assert "unknown question type" in str(error)
    else:
        raise AssertionError("Expected an unknown question type to raise ValueError")
