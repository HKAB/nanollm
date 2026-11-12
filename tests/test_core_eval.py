import pytest
import torch
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from nanollm.core_eval import evaluate_example
from nanollm.tokenizer import Tokenizer


@pytest.mark.parametrize("eot_id", [None, 0, 7])
@pytest.mark.parametrize("task_type", ["multiple_choice", "schema", "language_modeling"])
def test_core_eval_padding_with_and_without_endoftext(eot_id, task_type):
    vocab = {"[UNK]": 0, "Q": 1, "A": 2, "B": 3, "C": 4, "D": 5, "E": 6}
    if eot_id is not None:
        if eot_id == 0:
            del vocab["[UNK]"]
        vocab["<|endoftext|>"] = eot_id
    backend = HFTokenizer(WordLevel(vocab))
    backend.pre_tokenizer = Whitespace()
    tokenizer = Tokenizer(backend)
    batches = []

    def model(input_ids):
        batches.append(input_ids.clone())
        # Always predict A; the A continuation should win each task.
        logits = torch.zeros(*input_ids.shape, 8)
        logits[..., 2] = 5
        return logits

    items = {
        "multiple_choice": {"query": "Q", "choices": ["A", "B C"], "gold": 0},
        "schema": {"context_options": ["Q", "Q B"], "continuation": "A", "gold": 0},
        "language_modeling": {"context": "Q", "continuation": "A"},
    }
    meta = {"task_type": task_type, "num_fewshot": 0, "continuation_delimiter": " "}
    assert evaluate_example(0, model, tokenizer, [items[task_type]], "cpu", meta)
    if task_type != "language_modeling":
        expected_pad = 0 if eot_id is None else eot_id
        assert batches[0][0, -1].item() == expected_pad
