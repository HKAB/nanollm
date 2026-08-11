from tasks import uit_vsfc


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
        {"sentence": "thời lượng học quá dài .", "sentiment": 0, "topic": 1},
        {"sentence": "em sẽ học lại ở học kỳ sau .", "sentiment": 1, "topic": 3},
        {"sentence": "thầy giảng bài hay .", "sentiment": 2, "topic": 0},
    ]
)


def _patch_dataset(monkeypatch):
    calls = []

    def fake_load_dataset(name, *, split):
        calls.append((name, split))
        return FakeDataset(ROWS)

    monkeypatch.setattr(uit_vsfc, "load_dataset", fake_load_dataset)
    return calls


def test_task_always_loads_test_split(monkeypatch):
    calls = _patch_dataset(monkeypatch)
    task = uit_vsfc.UITVSFCSentiment()

    assert calls == [("uitnlp/vietnamese_students_feedback", "test")]
    assert task.split == "test"


def test_sentiment_mapping_and_topic_is_ignored(monkeypatch):
    _patch_dataset(monkeypatch)
    task = uit_vsfc.UITVSFCSentiment()

    assert [task[index]["answer"] for index in range(3)] == [
        "negative",
        "neutral",
        "positive",
    ]
    example = task[0]
    assert "topic" not in example
    assert "training_program" not in example["messages"][0]["content"]


def test_strict_answer_only_evaluation(monkeypatch):
    _patch_dataset(monkeypatch)
    task = uit_vsfc.UITVSFCSentiment(limit=1)
    example = task[0]

    assert task.evaluate(example, " negative ")
    assert not task.evaluate(example, "Negative")
    assert not task.evaluate(example, "Sentiment: negative")


def test_invalid_sentiment_is_rejected(monkeypatch):
    monkeypatch.setattr(
        uit_vsfc,
        "load_dataset",
        lambda *args, **kwargs: FakeDataset(
            [{"sentence": "Một phản hồi", "sentiment": 3, "topic": 0}]
        ),
    )
    task = uit_vsfc.UITVSFCSentiment()

    try:
        task[0]
    except ValueError as error:
        assert "unknown sentiment" in str(error)
    else:
        raise AssertionError("Expected an unknown sentiment to raise ValueError")
