import pytest
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import WordLevel

from nanollm.tokenizer import Tokenizer
from nanollm.tokenizers.gemma4_tokenizer import Gemma4Tokenizer
from nanollm.tokenizers.ling_tokenizer import LingTokenizer
from nanollm.tokenizers.qwen_tokenizer import QwenTokenizer


@pytest.mark.parametrize(
    "cls,vocab,expected",
    [
        (Tokenizer, {"<pad>": 0, "<|endoftext|>": 1}, 0),
        (Tokenizer, {"[PAD]": 0, "word": 1}, 0),
        (QwenTokenizer, {"word": 0, "<|endoftext|>": 1}, 1),
        (QwenTokenizer, {"word": 0, "<|im_end|>": 1}, 1),
        (Gemma4Tokenizer, {"<bos>": 0, "<eos>": 1, "<pad>": 2}, 2),
        (Gemma4Tokenizer, {"<bos>": 0, "<eos>": 1}, 1),
        (LingTokenizer, {"word": 0, "<|role_end|>": 1, "<|pad|>": 2}, 2),
        (LingTokenizer, {"word": 0, "<|startoftext|>": 1}, 1),
        (Tokenizer, {"word": 0}, 0),
    ],
)
def test_padding_token_selection(cls, vocab, expected):
    tokenizer = cls(HFTokenizer(WordLevel(vocab)))
    assert tokenizer.get_pad_token_id() == expected
    assert tokenizer.tokenizer.get_vocab() == vocab


def test_configured_padding_takes_precedence():
    backend = HFTokenizer(WordLevel({"<pad>": 0, "custom_padding": 1}))
    backend.enable_padding(pad_id=1, pad_token="custom_padding")
    assert Tokenizer(backend).get_pad_token_id() == 1


def test_empty_vocabulary_has_clear_error():
    with pytest.raises(ValueError, match="empty vocabulary"):
        Tokenizer(HFTokenizer(WordLevel({}))).get_pad_token_id()
