"""
Tests for the Engine class.

Run with:
    python -m pytest tests/test_engine.py -v
"""

from collections import deque
from dataclasses import dataclass
import re

import torch

from nanollm.engine import Engine, KVCache, dispatch_tool, parse_tool_call

# -----------------------------------------------------------------------------
# Shared mock infrastructure

IM_END_ID  = 151645   # used as sentinel in mocks
EOS_ID     = 151643   # <|endoftext|>
VOCAB_SIZE = 152064   # vocab size (doesn't matter for logic, just needs to be large enough)


@dataclass
class MockConfig:
    """Minimal config that Engine and KVCache need."""
    n_kv_groups: int = 4
    head_dim: int = 8
    n_layers: int = 2
    context_length: int = 512


class MockTokenizer:
    """
    UTF-8 byte-level tokenizer with special tokens.

    Token IDs:
      0-255  : raw UTF-8 bytes
      151643 : <|endoftext|>   (EOS / pad)
      151645 : <|im_end|>      (end of assistant turn)
    """
    _SPECIAL = {
        "<|endoftext|>": EOS_ID,
        "<|im_end|>":    IM_END_ID,
        "<|im_start|>":  151644,
    }
    _ID_TO_SPECIAL = {v: k for k, v in _SPECIAL.items()}

    def encode_special(self, s):
        return self._SPECIAL.get(s)

    def token_to_id(self, s):
        return self._SPECIAL.get(s)

    def decode(self, tokens, **kwargs):
        if not isinstance(tokens, list):
            tokens = tokens.tolist()
        res = bytearray()
        for t in tokens:
            if 0 <= t < 256:
                res.append(t)
            elif t == IM_END_ID:
                res.extend(b"<|im_end|>")
            elif t == EOS_ID:
                res.extend(b"<|endoftext|>")
        return res.decode("utf-8", errors="replace")

    def get_eos_token_ids(self):
        return {IM_END_ID, EOS_ID}

    def parse_tool_call(self, text):
        m = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        if not m:
            return None
        content = m.group(1)
        fn_m = re.search(r'<function=(\w+)', content)
        if not fn_m:
            return None
        func_name = fn_m.group(1)
        kwargs = {}
        for pm in re.finditer(r'<parameter=(\w+)>(.*?)</parameter>', content, re.DOTALL):
            kwargs[pm.group(1)] = pm.group(2).strip()
        return func_name, kwargs

    def render_tool_response(self, result):
        return f"\n<tool_response>\n{result}\n</tool_response>\n"

    def encode(self, text):
        return list(text.encode("utf-8"))


class UniformModel:
    """Returns uniform logits — every token equally likely."""
    def __init__(self):
        self.config = MockConfig()
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        B, T = ids.shape
        if kv_cache is not None:
            kv_cache.advance(T)
        return torch.zeros(B, T, VOCAB_SIZE)


class ScriptedModel:
    """
    Returns logits that force generation of a pre-scripted token sequence.
    After the script is exhausted, forces <|im_end|>.
    """
    def __init__(self, script_tokens):
        self.script = deque(script_tokens)
        self.config = MockConfig()
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        B, T = ids.shape
        if kv_cache is not None:
            kv_cache.advance(T)
        logits = torch.full((B, T, VOCAB_SIZE), -1e9)
        next_tok = self.script.popleft() if self.script else IM_END_ID
        logits[:, -1, next_tok] = 1e9
        return logits


def _token_seq(text, eos=True):
    """Encode text to byte tokens, optionally append <|im_end|>."""
    toks = list(text.encode("utf-8"))
    if eos:
        toks.append(IM_END_ID)
    return toks


# -----------------------------------------------------------------------------
# KVCache tests

def test_kv_cache_basic():
    kv = KVCache(batch_size=2, num_heads=3, seq_len=64, head_dim=5,
                 num_layers=6, device="cpu", dtype=torch.float32)
    assert kv.get_pos() == 0
    assert kv.k_cache.shape == (6, 2, 64, 3, 5)
    assert kv.v_cache.shape == (6, 2, 64, 3, 5)

    kv.advance(10)
    assert kv.get_pos() == 10
    kv.advance(5)
    assert kv.get_pos() == 15

    kv.reset()
    assert kv.get_pos() == 0

    k0, v0 = kv.get_layer_cache(0)
    assert k0.shape == (2, 64, 3, 5)
    assert v0.shape == (2, 64, 3, 5)


def test_kv_cache_prefill():
    src = KVCache(batch_size=1, num_heads=4, seq_len=32, head_dim=8,
                  num_layers=2, device="cpu", dtype=torch.float32)
    src.k_cache[0, 0, :16] = 1.0
    src.v_cache[0, 0, :16] = 2.0
    src.advance(16)

    dst = KVCache(batch_size=1, num_heads=4, seq_len=64, head_dim=8,
                  num_layers=2, device="cpu", dtype=torch.float32)
    dst.prefill(src)

    assert dst.get_pos() == 16
    assert (dst.k_cache[0, 0, :16] == 1.0).all()
    assert (dst.v_cache[0, 0, :16] == 2.0).all()


# -----------------------------------------------------------------------------
# parse_tool_call and dispatch_tool unit tests

def test_parse_tool_call_basic():
    text = (
        "<tool_call>\n"
        "<function=add>\n"
        "<parameter=a>3</parameter>\n"
        "<parameter=b>4</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    result = parse_tool_call(text)
    assert result is not None
    func_name, kwargs = result
    assert func_name == "add"
    assert kwargs == {"a": "3", "b": "4"}


def test_parse_tool_call_multiline_param():
    text = (
        "<tool_call>\n"
        "<function=greet>\n"
        "<parameter=message>\n"
        "Hello,\n"
        "world!\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    result = parse_tool_call(text)
    assert result is not None
    func_name, kwargs = result
    assert func_name == "greet"
    assert "Hello," in kwargs["message"]
    assert "world!" in kwargs["message"]


def test_parse_tool_call_missing_function():
    text = "<tool_call>\n<parameter=x>1</parameter>\n</tool_call>"
    assert parse_tool_call(text) is None


def test_parse_tool_call_no_block():
    assert parse_tool_call("just some plain text") is None


def test_dispatch_tool_basic():
    def add(a: int, b: int) -> int:
        return a + b

    result = dispatch_tool("add", {"a": "3", "b": "4"}, [add])
    assert result == "7"


def test_dispatch_tool_unknown():
    result = dispatch_tool("nonexistent", {}, [])
    assert "not found" in result


def test_dispatch_tool_error():
    def boom(x: int):
        raise ValueError("intentional error")

    result = dispatch_tool("boom", {"x": "1"}, [boom])
    assert "Error" in result


def test_dispatch_tool_type_coercion():
    def mul(x: float, y: float) -> float:
        return x * y

    result = dispatch_tool("mul", {"x": "2.5", "y": "4.0"}, [mul])
    assert result == "10.0"


# -----------------------------------------------------------------------------
# Engine generation tests

def test_seed_reproducibility():
    engine = Engine(UniformModel(), MockTokenizer())
    prompt = list("Hello".encode())

    for seed in [1, 42, 123]:
        r1, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r2, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        assert r1 == r2, f"seed={seed}: same seed must produce identical output"


def test_temperature_zero_determinism():
    engine = Engine(UniformModel(), MockTokenizer())
    prompt = list("Hi".encode())

    r1, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=1)
    r2, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=99)
    assert r1 == r2, "temperature=0 must give identical output regardless of seed"


def test_max_tokens_respected():
    engine = Engine(UniformModel(), MockTokenizer())
    prompt = list("Hi".encode())

    for max_tokens in [1, 4, 16]:
        results, _ = engine.generate_batch(prompt, max_tokens=max_tokens)
        n_generated = len(results[0]) - len(prompt)
        assert n_generated <= max_tokens


def test_num_samples_count():
    engine = Engine(UniformModel(), MockTokenizer())
    prompt = list("Hi".encode())

    for n in [1, 4, 8]:
        results, _ = engine.generate_batch(prompt, num_samples=n, max_tokens=3)
        assert len(results) == n


def test_multi_sample_diversity():
    """With uniform logits + temperature=1, 16 samples should not all be identical."""
    engine = Engine(UniformModel(), MockTokenizer())
    prompt = list("Hello".encode())

    first_toks = []
    for col, _ in engine.generate(prompt, num_samples=16, max_tokens=1, temperature=1.0, seed=42):
        first_toks = col

    assert len(set(first_toks)) > 1, (
        "All 16 samples produced the same first token — looks like broadcasting bug."
    )


def test_generate_prompts_batches_exact_prefill_lengths_and_ragged_decode():
    class RecordingModel:
        def __init__(self):
            self.config = MockConfig()
            self._device = torch.device("cpu")
            self.calls = []

        def get_device(self):
            return self._device

        def forward(self, ids, kv_cache=None, position_ids=None):
            before = kv_cache.cache_seqlens.clone()
            self.calls.append((tuple(ids.shape), before.tolist()))
            kv_cache.advance(ids.shape[1])
            logits = torch.full((*ids.shape, 16), -1e9)
            # Prefill emits token 5; the first decode step emits EOS.
            token = 5 if not torch.any(before) else EOS_ID
            vocab_token = 6 if token == EOS_ID else token
            logits[:, -1, vocab_token] = 1e9
            return logits

    class SmallTokenizer:
        def get_eos_token_ids(self):
            return {6}

    model = RecordingModel()
    engine = Engine(model, SmallTokenizer())
    prompts = [[1], [2, 3], [4], [5, 6]]
    results = engine.generate_prompts(
        prompts,
        batch_size=4,
        max_tokens=3,
        max_length_delta=4,
        use_cuda_graphs=False,
        completion_check_interval=1,
    )

    assert results == [prompt + [5] for prompt in prompts]
    assert ((2, 1), [0, 0]) in model.calls
    assert ((2, 2), [0, 0]) in model.calls
    assert ((4, 1), [1, 1, 2, 2]) in model.calls


def test_generate_prompts_uses_optional_packed_prefill_capability():
    class PackedRecordingModel:
        def __init__(self):
            self.config = MockConfig()
            self._device = torch.device("cpu")
            self.prefill = None

        def get_device(self):
            return self._device

        def supports_packed_prefill(self):
            return True

        def forward(self, ids, kv_cache=None, position_ids=None,
                    cu_seqlens=None, logit_positions=None):
            logits = torch.full((ids.shape[0], ids.shape[1], 16), -1e9)
            if cu_seqlens is not None:
                self.prefill = (
                    ids.tolist(), position_ids.tolist(), cu_seqlens.tolist(),
                    logit_positions.tolist(),
                )
                lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                kv_cache.cache_seqlens.copy_(lengths)
                kv_cache.has_previous_state = True
                selected = torch.full((lengths.numel(), 1, 16), -1e9)
                selected[:, :, 5] = 1e9
                return selected

            # First decode after packed prefill emits EOS.
            logits[:, -1, 6] = 1e9
            kv_cache.advance(ids.shape[1])
            return logits

    class SmallTokenizer:
        def get_eos_token_ids(self):
            return {6}

    model = PackedRecordingModel()
    engine = Engine(model, SmallTokenizer())
    prompts = [[1, 2, 3], [4], [5, 6]]
    results = engine.generate_prompts(
        prompts,
        batch_size=3,
        max_tokens=3,
        use_cuda_graphs=False,
        completion_check_interval=1,
    )

    assert results == [prompt + [5] for prompt in prompts]
    # Buckets are length-sorted to keep the following ragged decode compact.
    assert model.prefill == (
        [[4, 5, 6, 1, 2, 3]],
        [[0, 0, 1, 0, 1, 2]],
        [0, 1, 3, 6],
        [0, 2, 5],
    )


def test_eos_stops_generation():
    """<|im_end|> token must end generation for that row."""
    # Script: generate 3 real tokens then im_end
    tok = MockTokenizer()
    script = list("abc".encode()) + [IM_END_ID]
    engine = Engine(ScriptedModel(script), tok)

    results, _ = engine.generate_batch(list("X".encode()), max_tokens=20)
    generated = results[0][1:]  # strip prompt
    assert IM_END_ID not in generated, "<|im_end|> should be stripped from results"
    assert tok.decode(generated) == "abc"


def test_eos_id_stops_generation():
    """<|endoftext|> token must also end generation."""
    tok = MockTokenizer()
    script = list("hi".encode()) + [EOS_ID]
    engine = Engine(ScriptedModel(script), tok)

    results, _ = engine.generate_batch(list("X".encode()), max_tokens=20)
    generated = results[0][1:]
    assert EOS_ID not in generated
    assert tok.decode(generated) == "hi"


# -----------------------------------------------------------------------------
# Tool calling integration tests

def _make_tool_call_script(func_name, params: dict):
    """Build the byte-token script that spells out a tool_call block + <|im_end|>."""
    lines = ["<tool_call>\n", f"<function={func_name}>\n"]
    for k, v in params.items():
        lines.append(f"<parameter={k}>{v}</parameter>\n")
    lines.append("</function>\n</tool_call>")
    text = "".join(lines)
    return list(text.encode("utf-8")) + [IM_END_ID]


def test_tool_call_add():
    """Engine should detect <tool_call>, call add(a,b), inject <tool_response>."""
    def add(a: int, b: int) -> int:
        return a + b

    tok = MockTokenizer()
    script = _make_tool_call_script("add", {"a": "3", "b": "4"})
    engine = Engine(ScriptedModel(script), tok)

    results, masks = engine.generate_batch(
        list("Q".encode()), tools=[add], max_tokens=512
    )
    full_text = tok.decode(results[0])
    assert "<tool_response>" in full_text
    assert "7" in full_text
    assert "</tool_response>" in full_text


def test_tool_call_unknown_function():
    """Engine should inject an error message when the called function is not in the tools list."""
    tok = MockTokenizer()
    script = _make_tool_call_script("unknown_fn", {"x": "1"})
    engine = Engine(ScriptedModel(script), tok)

    # tools=[] means tool-calling is enabled but the list is empty — dispatch returns "not found"
    results, _ = engine.generate_batch(
        list("Q".encode()), tools=[], max_tokens=512
    )
    full_text = tok.decode(results[0])
    assert "<tool_response>" in full_text
    assert "not found" in full_text


def test_tool_call_no_tools_registered():
    """When tools=None, engine should NOT inject any tool response."""
    tok = MockTokenizer()
    script = _make_tool_call_script("add", {"a": "1", "b": "2"})
    engine = Engine(ScriptedModel(script), tok)

    results, _ = engine.generate_batch(
        list("Q".encode()), tools=None, max_tokens=512
    )
    full_text = tok.decode(results[0])
    assert "<tool_response>" not in full_text


def test_tool_call_mask_is_zero_for_forced_tokens():
    """Tokens injected as tool responses must have mask=0."""
    def add(a: int, b: int) -> int:
        return a + b

    tok = MockTokenizer()
    script = _make_tool_call_script("add", {"a": "10", "b": "20"})
    engine = Engine(ScriptedModel(script), tok)

    _, masks = engine.generate_batch(
        list("Q".encode()), tools=[add], max_tokens=512
    )
    combined_masks = masks[0]
    # prompt tokens are 0, sampled tokens are 1, forced (tool response) tokens are 0
    # The tool response should introduce some 0-mask tokens after 1-mask tokens
    # (prompt is all 0s, then model generates 1s, then forced response is 0s again)
    assert 0 in combined_masks[len(list("Q".encode())):], (
        "Tool response tokens should have mask=0"
    )


def test_tool_call_string_return():
    """Tool that returns a string should be injected correctly."""
    def get_weather(city: str) -> str:
        return f"Sunny in {city}, 25°C"

    tok = MockTokenizer()
    script = _make_tool_call_script("get_weather", {"city": "Paris"})
    engine = Engine(ScriptedModel(script), tok)

    results, _ = engine.generate_batch(
        list("Q".encode()), tools=[get_weather], max_tokens=512
    )
    full_text = tok.decode(results[0])
    assert "Sunny in Paris" in full_text


def test_tool_call_exception_is_caught():
    """A tool that raises should inject an error message, not crash the engine."""
    def broken_tool(x: int) -> int:
        raise RuntimeError("disk on fire")

    tok = MockTokenizer()
    script = _make_tool_call_script("broken_tool", {"x": "1"})
    engine = Engine(ScriptedModel(script), tok)

    results, _ = engine.generate_batch(
        list("Q".encode()), tools=[broken_tool], max_tokens=512
    )
    full_text = tok.decode(results[0])
    assert "<tool_response>" in full_text
    assert "Error" in full_text
