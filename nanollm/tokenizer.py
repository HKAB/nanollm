"""
Base Tokenizer wrapper.
Provides generic wrapper over HuggingFace tokenizers.
"""

import os
import copy
import json
from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path):
        # init from a HuggingFace pretrained tokenizer (e.g. "Qwen/Qwen1.5-0.5B")
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        # init from a local directory on disk
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self):
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id):
        return self.tokenizer.id_to_token(id)
    
    def token_to_id(self, token):
        return self.tokenizer.token_to_id(token)

    def _encode_one(self, text, prepend=None, append=None, num_threads=None):
        assert isinstance(text, str)
        ids = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            ids.append(append_id)
        return ids

    def encode_special(self, text):
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self):
        return self.tokenizer.token_to_id("<|endoftext|>")

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    def get_eos_token_ids(self):
        """Override in subclasses to provide valid EOS token IDs."""
        return set()

    def parse_tool_call(self, text):
        """Override in subclasses to parse model-specific tool calls."""
        return None

    def render_tool_response(self, result):
        """Override in subclasses to format a tool response."""
        return str(result)

    def render_conversation(self, conversation, max_tokens=2048, mask_history=False,
                            return_boundaries=False):
        """Override in subclasses to implement model-specific chat templates."""
        raise NotImplementedError("render_conversation must be implemented by Tokenizer subclasses.")

    def render_for_completion(self, conversation, enable_thinking=True):
        """Override in subclasses to implement model-specific chat templates."""
        raise NotImplementedError("render_for_completion must be implemented by Tokenizer subclasses.")

def get_tokenizer(model_id, architectures):
    from nanollm.models.registry import get_model_entry
    
    try:
        entry = get_model_entry(architectures)
        tokenizer_class = entry.get("tokenizer_class", Tokenizer)
    except ValueError:
        tokenizer_class = Tokenizer
        
    if os.path.isdir(model_id):
        return tokenizer_class.from_directory(model_id)
    return tokenizer_class.from_pretrained(model_id)

def get_token_bytes(device="cpu"):
    return None
