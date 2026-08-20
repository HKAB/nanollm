from tasks.sft import uit_vsmec


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
        {"Sentence": "người ta có bạn bè nhìn vui thật", "Emotion": "Sadness"},
        {"Sentence": "kinh vãi 😡", "Emotion": "Disgust"},
    ]
)


def _patch_dataset(monkeypatch):
    calls = []

    def fake_load_dataset(name, *, split):
        calls.append((name, split))
        return FakeDataset(ROWS)

    monkeypatch.setattr(uit_vsmec, "load_dataset", fake_load_dataset)
    return calls


def test_task_always_loads_test_split(monkeypatch):
    calls = _patch_dataset(monkeypatch)
    task = uit_vsmec.UITVSMEC()

    assert calls == [("tridm/UIT-VSMEC", "test")]
    assert task.split == "test"


def test_example_and_strict_answer_only_evaluation(monkeypatch):
    _patch_dataset(monkeypatch)
    task = uit_vsmec.UITVSMEC(limit=1)
    example = task[0]

    assert example["messages"][-1]["content"] == "Sadness"
    assert all(label in example["messages"][0]["content"] for label in task.emotions)
    assert task.evaluate(example, " Sadness ")
    assert not task.evaluate(example, "The emotion is Sadness")
    assert not task.evaluate(example, "sadness")


def test_unknown_emotion_is_rejected(monkeypatch):
    monkeypatch.setattr(
        uit_vsmec,
        "load_dataset",
        lambda *args, **kwargs: FakeDataset(
            [{"Sentence": "Một câu", "Emotion": "Happiness"}]
        ),
    )
    task = uit_vsmec.UITVSMEC()

    try:
        task[0]
    except ValueError as error:
        assert "unknown Emotion" in str(error)
    else:
        raise AssertionError("Expected an unknown emotion to raise ValueError")
