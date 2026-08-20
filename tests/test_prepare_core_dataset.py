import pytest

from scripts import prepare_core_dataset
from scripts.prepare_core_dataset import ordered_choices


def test_ordered_choices_respects_labels_not_source_order():
    assert ordered_choices(
        ["C", "A", "D", "B"],
        ["third", "first", "fourth", "second"],
        "row-1",
    ) == ["first", "second", "third", "fourth"]


def test_ordered_choices_rejects_malformed_labels():
    with pytest.raises(ValueError, match="labels A, B, C, D"):
        ordered_choices(["A", "B", "C"], ["one", "two", "three"], "row-1")


def test_build_rows_skips_invalid_samples(monkeypatch, capsys):
    global_mmlu = [
        {
            "sample_id": "invalid",
            "question": "Bad question",
            "option_a": "",
            "option_b": "B",
            "option_c": "C",
            "option_d": "D",
            "answer": "A",
        },
        {
            "sample_id": "global/1",
            "question": "Global question",
            "option_a": "A",
            "option_b": "B",
            "option_c": "C",
            "option_d": "D",
            "answer": "A",
            "subject": "subject",
        },
    ]
    wikipedia = [{
        "question": "Wikipedia question",
        "choices": {
            "labels": ["A", "B", "C", "D"],
            "text": ["A", "B", "C", "D"],
        },
        "answerKey": "B",
        "metadata": "category",
    }]
    exams = [
        {
            "id": f"exam/{index}",
            "question": f"Exam question {index}",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
            "metadata": {"subject": subject},
        }
        for index, subject in enumerate(prepare_core_dataset.EXAM_SUBJECTS)
    ]

    def fake_load_dataset(path, *args, **kwargs):
        return {
            "CohereLabs/Global-MMLU": global_mmlu,
            "vlsp-2023-vllm/wikipediaqa_vi": wikipedia,
            "vlsp-2023-vllm/exams_vi": exams,
        }[path]

    monkeypatch.setattr(prepare_core_dataset, "load_dataset", fake_load_dataset)

    rows = prepare_core_dataset.build_rows()

    assert len(rows) == 9
    assert all(row["id"] != "invalid" for row in rows)
    assert "Skipped invalid rows: global_mmlu=1" in capsys.readouterr().out
