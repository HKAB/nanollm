"""Create a datasets-4.0-compatible, data-only mirror of V-IFEval.

The source repository's exported metadata uses the newer ``Json`` feature,
which datasets 4.0.0 cannot deserialize.  This script deliberately downloads
the raw JSONL instead and recreates the Arrow schema with the installed
datasets version.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, Features, List, Value
from huggingface_hub import hf_hub_download


SOURCE_REPO = "hkab/vi-ifeval"
SOURCE_FILE = "data/test.jsonl"
DESTINATION_REPO = "hkab/vi-ifeval"
EXPECTED_EXAMPLES = 1_134


FEATURES = Features(
    {
        "prompt": Value("string"),
        "instruction_id_list": List(Value("string")),
        # Explicit strings avoid the unsupported Json feature while preserving
        # every kwargs value and its original JSON type losslessly.
        "kwargs": List(Value("string")),
    }
)


def read_and_validate(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error

            prompt = row.get("prompt")
            instruction_ids = row.get("instruction_id_list")
            kwargs = row.get("kwargs")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Line {line_number} has an invalid prompt")
            if not isinstance(instruction_ids, list) or not instruction_ids:
                raise ValueError(f"Line {line_number} has no instructions")
            if not isinstance(kwargs, list) or len(kwargs) != len(instruction_ids):
                raise ValueError(
                    f"Line {line_number} has mismatched instructions and kwargs"
                )
            if not all(isinstance(item, str) for item in instruction_ids):
                raise ValueError(f"Line {line_number} has a non-string instruction ID")
            if not all(isinstance(item, dict) for item in kwargs):
                raise ValueError(f"Line {line_number} has non-object kwargs")
            rows.append(row)
    return rows


def build_dataset(rows: list[dict[str, Any]]) -> Dataset:
    columns = {
        "prompt": [row["prompt"] for row in rows],
        "instruction_id_list": [row["instruction_id_list"] for row in rows],
        "kwargs": [
            [json.dumps(item, ensure_ascii=False) for item in row["kwargs"]]
            for row in rows
        ],
    }
    return Dataset.from_dict(columns, features=FEATURES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", default=SOURCE_REPO)
    parser.add_argument("--source-file", default=SOURCE_FILE)
    parser.add_argument("--input", type=Path, help="Use a local JSONL file instead")
    parser.add_argument("--repo-id", default=DESTINATION_REPO)
    parser.add_argument(
        "--expected-examples",
        type=int,
        default=EXPECTED_EXAMPLES,
        help="Refuse upload if the row count differs; use 0 to disable",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    source_path = args.input or Path(
        hf_hub_download(
            repo_id=args.source_repo,
            filename=args.source_file,
            repo_type="dataset",
        )
    )
    rows = read_and_validate(source_path)
    if args.expected_examples and len(rows) != args.expected_examples:
        raise ValueError(
            f"Expected {args.expected_examples:,} examples, found {len(rows):,}; "
            "refusing to upload"
        )

    dataset = build_dataset(rows)
    print(f"Uploading {len(dataset):,} examples with features: {dataset.features}")
    DatasetDict({"test": dataset}).push_to_hub(
        args.repo_id,
        private=args.private,
    )
    print(f"Uploaded test split to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
