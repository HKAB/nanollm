import json

import pytest

from scripts.upload_v_ifeval import build_dataset, read_and_validate


def test_read_and_validate_accepts_v_ifeval_jsonl(tmp_path):
    path = tmp_path / "test.jsonl"
    row = {
        "prompt": "Mention alpha.",
        "instruction_id_list": ["keywords:existence"],
        "kwargs": [{"keywords": ["alpha"]}],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    assert read_and_validate(path) == [row]

    dataset = build_dataset([row])
    assert dataset.features["kwargs"].feature.dtype == "string"
    assert json.loads(dataset[0]["kwargs"][0]) == {"keywords": ["alpha"]}


@pytest.mark.parametrize(
    "row",
    [
        {"prompt": "", "instruction_id_list": ["x"], "kwargs": [{}]},
        {"prompt": "p", "instruction_id_list": [], "kwargs": []},
        {"prompt": "p", "instruction_id_list": ["x"], "kwargs": []},
    ],
)
def test_read_and_validate_rejects_invalid_rows(tmp_path, row):
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_and_validate(path)
