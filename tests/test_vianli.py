from tasks import vianli


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
            "uid": "uit_1",
            "premise": "Một buổi tọa đàm được tổ chức vào đầu tháng 4.",
            "hypothesis": "Đầu tháng 4 có một buổi tọa đàm.",
            "label": "entailment",
        },
        {
            "uid": "uit_2",
            "premise": "Mức phí thấp nhất là 20.000 đồng.",
            "hypothesis": "Mức phí cao nhất là 20.000 đồng.",
            "label": "contradiction",
        },
        {
            "uid": "uit_3",
            "premise": "Một căn hộ được bán vào tháng hai.",
            "hypothesis": "Chủ căn hộ rất hài lòng.",
            "label": "neutral",
        },
    ]
)


def _patch_dataset(monkeypatch):
    calls = []

    def fake_load_dataset(name, *, split):
        calls.append((name, split))
        return FakeDataset(ROWS)

    monkeypatch.setattr(vianli, "load_dataset", fake_load_dataset)
    return calls


def test_task_always_loads_test_split(monkeypatch):
    calls = _patch_dataset(monkeypatch)
    task = vianli.ViANLI()

    assert calls == [("uitnlp/ViANLI", "test")]
    assert task.split == "test"


def test_example_contains_pair_label_and_uid(monkeypatch):
    _patch_dataset(monkeypatch)
    task = vianli.ViANLI(limit=1)
    example = task[0]

    assert example["uid"] == "uit_1"
    assert example["answer"] == "entailment"
    assert example["messages"][-1]["content"] == "entailment"
    prompt = example["messages"][0]["content"]
    assert example["premise"] in prompt
    assert example["hypothesis"] in prompt
    assert all(label in prompt for label in task.labels)


def test_strict_answer_only_evaluation(monkeypatch):
    _patch_dataset(monkeypatch)
    task = vianli.ViANLI(limit=1)
    example = task[0]

    assert task.evaluate(example, " entailment ")
    assert not task.evaluate(example, "Entailment")
    assert not task.evaluate(example, "The answer is entailment")


def test_unknown_label_is_rejected(monkeypatch):
    monkeypatch.setattr(
        vianli,
        "load_dataset",
        lambda *args, **kwargs: FakeDataset(
            [
                {
                    "uid": "uit_bad",
                    "premise": "Tiền đề.",
                    "hypothesis": "Giả thuyết.",
                    "label": "unknown",
                }
            ]
        ),
    )
    task = vianli.ViANLI()

    try:
        task[0]
    except ValueError as error:
        assert "unknown label" in str(error)
    else:
        raise AssertionError("Expected an unknown label to raise ValueError")
