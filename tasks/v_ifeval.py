"""Vietnamese instruction-following evaluation using HKAB/V-IFEval."""

from __future__ import annotations

import importlib
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from datasets import load_dataset


DATASET_NAME = "truongnp5/vi-ifeval"
EVAL_SPLIT = "test"
V_IFEVAL_ENV = "V_IFEVAL_PATH"
DEFAULT_V_IFEVAL_PATH = Path(".cache/V-IFEval")


def _load_eval_dataset(dataset_name: str, split: str) -> Any:
    """Load a Hub dataset, bypassing incompatible exported feature metadata.

    Some newer Hub datasets describe arbitrary dictionaries with the ``Json``
    feature.  datasets 4.0.0 cannot deserialize that feature metadata even
    though its Arrow/Parquet reader can read the underlying values.  Loading
    the pushed Parquet shard directly avoids the metadata compatibility issue.
    """

    try:
        return load_dataset(dataset_name, split=split)
    except ValueError as error:
        if "Feature type 'Json' not found" not in str(error):
            raise

    data_file = f"hf://datasets/{dataset_name}/data/{split}-*.parquet"
    try:
        return load_dataset("parquet", data_files={split: data_file}, split=split)
    except Exception as fallback_error:
        raise RuntimeError(
            f"Could not load {dataset_name!r} after bypassing its incompatible "
            "Json feature metadata"
        ) from fallback_error


@dataclass(frozen=True)
class VIFEvalResult:
    """Per-example results matching V-IFEval's strict and loose evaluators."""

    strict: tuple[bool, ...]
    loose: tuple[bool, ...]

    @property
    def strict_prompt(self) -> bool:
        return all(self.strict)

    @property
    def loose_prompt(self) -> bool:
        return all(self.loose)


def _find_v_ifeval_root(explicit_path: str | os.PathLike[str] | None) -> Path:
    configured = explicit_path or os.environ.get(V_IFEVAL_ENV)
    root = Path(configured) if configured else DEFAULT_V_IFEVAL_PATH
    registry_path = root / "instructions_registry.py"
    if not registry_path.is_file():
        raise RuntimeError(
            f"V-IFEval was not found at {root}. Run "
            "`bash scripts/setup_v_ifeval.sh` or set V_IFEVAL_PATH to its checkout."
        )
    return root.resolve()


def load_instruction_registry(
    v_ifeval_path: str | os.PathLike[str] | None = None,
) -> Mapping[str, type]:
    """Load the registry from a V-IFEval source checkout."""

    root = _find_v_ifeval_root(v_ifeval_path)
    root_string = str(root)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)
    try:
        # The upstream utility calls nltk.download() at import time even though
        # its checkers use underthesea for tokenization. Setup downloads the
        # resource once; suppress that repeated network side effect at runtime.
        import nltk

        nltk_download = nltk.download
        nltk.download = lambda *args, **kwargs: True
        try:
            module = importlib.import_module("instructions_registry")
        finally:
            nltk.download = nltk_download
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "V-IFEval dependencies are missing. Run `bash scripts/setup_v_ifeval.sh`."
        ) from error
    return module.INSTRUCTION_DICT


def _check_instructions(
    example: Mapping[str, Any],
    response: str,
    registry: Mapping[str, type],
) -> tuple[bool, ...]:
    outcomes = []
    for instruction_id, kwargs in zip(
        example["instruction_id_list"], example["kwargs"], strict=True
    ):
        instruction = registry[instruction_id](instruction_id)
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=example["prompt"])
        outcomes.append(bool(response.strip() and instruction.check_following(response)))
    return tuple(outcomes)


def _loose_response_variants(response: str) -> tuple[str, ...]:
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    variants = (response, remove_first, remove_last, remove_both)
    return variants + tuple(variant.replace("*", "") for variant in variants)


def evaluate_response(
    example: Mapping[str, Any],
    response: str,
    registry: Mapping[str, type],
) -> VIFEvalResult:
    """Apply the official strict and loose V-IFEval checking semantics."""

    if not isinstance(response, str):
        count = len(example["instruction_id_list"])
        return VIFEvalResult((False,) * count, (False,) * count)

    strict = _check_instructions(example, response, registry)
    variants = _loose_response_variants(response)
    loose = tuple(
        any(result)
        for result in zip(
            *(_check_instructions(example, variant, registry) for variant in variants),
            strict=True,
        )
    )
    return VIFEvalResult(strict, loose)


def aggregate_results(results: Sequence[VIFEvalResult]) -> dict[str, float]:
    """Return the four headline metrics printed by V-IFEval."""

    if not results:
        raise ValueError("Cannot aggregate an empty V-IFEval result set")
    instruction_total = sum(len(result.strict) for result in results)
    if instruction_total == 0:
        raise ValueError("V-IFEval examples must contain at least one instruction")
    return {
        "strict_prompt_accuracy": sum(r.strict_prompt for r in results) / len(results),
        "strict_instruction_accuracy": sum(sum(r.strict) for r in results)
        / instruction_total,
        "loose_prompt_accuracy": sum(r.loose_prompt for r in results) / len(results),
        "loose_instruction_accuracy": sum(sum(r.loose) for r in results)
        / instruction_total,
    }


class VIFEval:
    """The verified 1,134-example Vietnamese V-IFEval benchmark."""

    eval_type = "instruction_following"
    split = EVAL_SPLIT

    def __init__(
        self,
        *,
        dataset_name: str = DATASET_NAME,
        split: str = EVAL_SPLIT,
        v_ifeval_path: str | os.PathLike[str] | None = None,
        dataset: Any | None = None,
        registry: Mapping[str, type] | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.ds = (
            dataset
            if dataset is not None
            else _load_eval_dataset(dataset_name, split)
        )
        if shuffle:
            if hasattr(self.ds, "shuffle"):
                self.ds = self.ds.shuffle(seed=seed)
            else:
                self.ds = list(self.ds)
                random.Random(seed).shuffle(self.ds)
        self.registry = (
            registry if registry is not None else load_instruction_registry(v_ifeval_path)
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = dict(self.ds[index])
        instruction_ids = row.get("instruction_id_list")
        kwargs = row.get("kwargs")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"V-IFEval example at index {index} has an invalid prompt")
        if not instruction_ids or len(instruction_ids) != len(kwargs or []):
            raise ValueError(
                f"V-IFEval example at index {index} has mismatched instructions and kwargs"
            )
        # render_for_completion expects a final assistant turn and removes it.
        row["messages"] = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": ""},
        ]
        return row

    def evaluate_details(
        self, conversation: Mapping[str, Any], assistant_response: str
    ) -> VIFEvalResult:
        return evaluate_response(conversation, assistant_response, self.registry)

    def evaluate(self, conversation: Mapping[str, Any], assistant_response: str) -> bool:
        return self.evaluate_details(conversation, assistant_response).strict_prompt

    def reward(self, conversation: Mapping[str, Any], assistant_response: str) -> float:
        return float(self.evaluate(conversation, assistant_response))
