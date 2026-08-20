import pytest

from tasks.sft import abmusu
from nanollm.tokenizers.qwen_tokenizer import QwenTokenizer


ROWS = [
    {
        "single_documents": [
            {
                "title": "Bài báo thứ nhất",
                "anchor_text": "Thông tin mở đầu thứ nhất.",
                "raw_text": "Nội dung đầy đủ thứ nhất.",
            },
            {
                "title": "Bài báo thứ hai",
                "anchor_text": "Thông tin mở đầu thứ hai.",
                "raw_text": "Nội dung đầy đủ thứ hai.",
            },
        ],
        "summary": "Hà Nội đón nắng đẹp hôm nay.",
        "category": "Xã hội",
    }
]


def test_rouge2_exact_match_and_unicode_normalization():
    score = abmusu.rouge2_score(
        "HÀ NỘI đón nắng đẹp hôm nay!", "Hà Nội đón nắng đẹp hôm nay."
    )
    assert score.precision == 1.0
    assert score.recall == 1.0
    assert score.f1 == 1.0


def test_rouge2_uses_multiset_overlap_and_harmonic_mean():
    score = abmusu.rouge2_score("a b c", "a b d e")
    assert score.precision == 0.5
    assert score.recall == pytest.approx(1 / 3)
    assert score.f1 == pytest.approx(0.4)


def test_task_defaults_to_labeled_validation_and_renders_all_documents(monkeypatch):
    calls = []

    def fake_load_dataset(name, *, split):
        calls.append((name, split))
        return ROWS

    monkeypatch.setattr(abmusu, "load_dataset", fake_load_dataset)
    task = abmusu.AbMusu()
    example = task[0]

    assert calls == [(abmusu.DATASET_NAME, "validation")]
    prompt = example["messages"][0]["content"]
    assert all(document["title"] in prompt for document in ROWS[0]["single_documents"])
    assert all(document["raw_text"] in prompt for document in ROWS[0]["single_documents"])
    assert example["messages"][-1]["content"] == ROWS[0]["summary"]


def test_unlabeled_test_example_cannot_be_evaluated():
    row = dict(ROWS[0])
    row.pop("summary")
    task = abmusu.AbMusu(split="test", dataset=[row])
    example = task[0]

    with pytest.raises(ValueError, match="reference summary"):
        task.evaluate(example, "Bản tóm tắt dự đoán.")


def test_macro_aggregation():
    metrics = abmusu.aggregate_rouge2(
        [abmusu.Rouge2Score(1.0, 0.5, 2 / 3), abmusu.Rouge2Score(0.0, 0.5, 0.0)]
    )
    assert metrics == {
        "rouge2_precision": 0.5,
        "rouge2_recall": 0.5,
        "rouge2_f1": pytest.approx(1 / 3),
    }


def test_completion_renderer_honors_larger_context_budget():
    tokenizer = object.__new__(QwenTokenizer)
    observed = []
    tokenizer.render_conversation = lambda conversation, max_tokens: (
        observed.append(max_tokens) or [11],
        [0],
    )
    tokenizer.encode_special = lambda token: 12
    tokenizer.encode = lambda text: [13]

    result = tokenizer.render_for_completion(
        {"messages": [{"role": "assistant", "content": "reference"}]},
        enable_thinking=False,
        max_tokens=3500,
    )

    assert observed == [3500]
    assert result[0] == 11
