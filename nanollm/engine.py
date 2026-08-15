"""
Engine for efficient generation and inference of LLM models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization except for EOS/special tokens provided by the tokenizer.
- Tool calling formats are delegated to the tokenizer subclass via `tokenizer.parse_tool_call`.
"""

import time
import sys
import inspect
import re
from collections import deque
import torch
import torch.nn.functional as F

from nanollm.checkpoint_manager import load_pretrained_hf
from nanollm.cache import KVCache
from nanollm.common import autodetect_device_type, compute_init

# -----------------------------------------------------------------------------
# Tool call helpers




def parse_tool_call(text):
    """Parse the engine's compact XML-like tool-call representation."""
    block = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if block is None:
        return None
    function = re.search(r"<function=([^>\s]+)", block.group(1))
    if function is None:
        return None
    kwargs = {
        match.group(1): match.group(2).strip()
        for match in re.finditer(
            r"<parameter=([^>\s]+)>(.*?)</parameter>",
            block.group(1),
            re.DOTALL,
        )
    }
    return function.group(1), kwargs


def dispatch_tool(func_name, kwargs, tools):
    """
    Find a tool by name in `tools` and call it with the given kwargs.
    Attempts to coerce string args to annotated types.
    Returns a string result (or an error message).
    """
    func = next((f for f in tools if f.__name__ == func_name), None)
    if func is None:
        return f"Error: function '{func_name}' not found"
    try:
        hints = {}
        try:
            hints = {k: v.annotation for k, v in inspect.signature(func).parameters.items()
                     if v.annotation is not inspect.Parameter.empty}
        except Exception:
            pass
        coerced = {}
        for k, v in kwargs.items():
            if k in hints and hints[k] is not inspect.Parameter.empty:
                try:
                    coerced[k] = hints[k](v)
                except Exception:
                    coerced[k] = v
            else:
                coerced[k] = v
        return str(func(**coerced))
    except Exception as e:
        return f"Error: {e}"


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class _CudaGraphDecode:
    """Captured one-token decode for one stable batch shape."""

    def __init__(self, engine, source_cache, dtype):
        device = engine.model.get_device()
        self.engine = engine
        self.cache = engine._allocate_cache(
            source_cache.batch_size, source_cache.max_seq_len, dtype
        )
        self.cache.copy_from(source_cache)
        self.static_ids = torch.zeros(
            source_cache.batch_size, 1, dtype=torch.long, device=device
        )
        self.static_positions = self.cache.cache_seqlens.to(torch.long).unsqueeze(1)

        # Warm third-party kernels and allocator pools on a disposable cache.
        warm_cache = engine._allocate_cache(
            source_cache.batch_size, source_cache.max_seq_len, dtype
        )
        warm_cache.copy_from(source_cache)
        warm_stream = torch.cuda.Stream(device=device)
        warm_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(warm_stream):
            engine._forward(
                self.static_ids, warm_cache,
                position_ids=self.static_positions,
            )
        torch.cuda.current_stream(device).wait_stream(warm_stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_logits = engine._forward(
                self.static_ids, self.cache,
                position_ids=self.static_positions,
            )
        # Capture executes once and advances the state. Restore the true prefill.
        self.cache.copy_from(source_cache)

    def __call__(self, ids):
        self.static_ids.copy_(ids)
        self.static_positions.copy_(
            self.cache.cache_seqlens.to(torch.long).unsqueeze(1)
        )
        self.graph.replay()
        return self.static_logits


# -----------------------------------------------------------------------------

class RowState:
    """Per-row state tracking during generation."""
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()       # queue of tokens to force-inject
        self.text_buf = ""                 # decoded text of tokens generated so far (for tool call detection)
        self.in_tool_call = False          # currently inside a <tool_call> block
        self.tool_call_buf = ""            # text accumulated inside the tool call block
        self.completed = False


class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        try:
            signature = inspect.signature(model.forward)
            self._forward_parameters = set(signature.parameters)
            self._forward_accepts_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
        except (TypeError, ValueError):
            self._forward_parameters = set()
            self._forward_accepts_kwargs = False

    def _allocate_cache(self, batch_size, seq_len, dtype):
        """Use a model cache factory when extra inference state is required."""
        if hasattr(self.model, "create_kv_cache"):
            return self.model.create_kv_cache(batch_size, seq_len, dtype)
        config = self.model.config
        return KVCache(
            batch_size=batch_size,
            num_heads=config.n_kv_groups,
            seq_len=seq_len,
            head_dim=config.head_dim,
            num_layers=config.n_layers,
            device=self.model.get_device(),
            dtype=dtype,
        )

    def _forward(self, ids, cache, *, logit_positions=None, position_ids=None,
                 cu_seqlens=None):
        kwargs = {"kv_cache": cache}
        if logit_positions is not None and (
            "logit_positions" in self._forward_parameters
            or self._forward_accepts_kwargs
        ):
            kwargs["logit_positions"] = logit_positions
        if position_ids is not None and (
            "position_ids" in self._forward_parameters
            or self._forward_accepts_kwargs
        ):
            kwargs["position_ids"] = position_ids
        if cu_seqlens is not None and (
            "cu_seqlens" in self._forward_parameters
            or self._forward_accepts_kwargs
        ):
            kwargs["cu_seqlens"] = cu_seqlens
        return self.model.forward(ids, **kwargs)

    @torch.inference_mode()
    def generate(self, tokens, tools=None, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        Generate tokens from a prompt.

        Args:
            tokens: list[int] — prompt token ids
            tools: list[callable] | None — Python functions the model may call.
                   Each function's __name__ must match what the model uses in <function=NAME>.
                   Parameter types are inferred from annotations for automatic coercion.
            num_samples: number of independent samples to generate in parallel
            max_tokens: maximum number of tokens to generate (None = unlimited)
            temperature: sampling temperature (0.0 = greedy)
            top_k: top-k sampling (None = full softmax)
            seed: RNG seed

        Yields:
            (token_column, token_masks) where:
              - token_column: list[int] of length num_samples — the token chosen for each row
              - token_masks: list[int] of length num_samples — 1 if sampled, 0 if forced (tool output)
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        eos_ids = self.tokenizer.get_eos_token_ids()

        # 1) Batch-1 prefill of the prompt
        kv_cache_prefill = self._allocate_cache(1, len(tokens), dtype)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        # The full prompt still passes through every transformer layer. Only
        # its final hidden state needs the large vocabulary projection to seed
        # unrestricted autoregressive generation.
        logits = self._forward(ids, kv_cache_prefill, logit_positions=-1)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for all samples
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else getattr(self.model.config, 'context_length', 4096)
        kv_cache_decode = self._allocate_cache(num_samples, kv_length_hint, dtype)
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        # 3) Initialize per-row state
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)

                # EOS check
                if next_token in eos_ids:
                    state.completed = True
                    continue

                # Skip tool-call logic entirely when tools=None (disabled)
                if tools is None:
                    continue

                # Decode this token and append to the text buffer
                chunk = self.tokenizer.decode([next_token])
                state.text_buf += chunk

                # Detect entry into a <tool_call> block
                if not state.in_tool_call:
                    if "<tool_call>" in state.text_buf:
                        state.in_tool_call = True
                        # Reset tool_call_buf to everything after <tool_call>
                        state.tool_call_buf = state.text_buf.split("<tool_call>", 1)[1]
                        state.text_buf = ""  # reset to avoid repeated triggers
                elif state.in_tool_call:
                    state.tool_call_buf += chunk
                    if "</tool_call>" in state.tool_call_buf:
                        state.in_tool_call = False
                        full_block = "<tool_call>" + state.tool_call_buf
                        state.tool_call_buf = ""
                        parsed = self.tokenizer.parse_tool_call(full_block)
                        if parsed is not None:
                            func_name, kwargs = parsed
                            result = dispatch_tool(func_name, kwargs, tools)
                            response = self.tokenizer.render_tool_response(result)
                            response_tokens = self.tokenizer.encode(response)
                            state.forced_tokens.extend(response_tokens)

            yield token_column, token_masks
            num_generated += 1

            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            position_ids = kv_cache_decode.cache_seqlens.to(torch.long).unsqueeze(1)
            logits = self._forward(
                ids, kv_cache_decode, position_ids=position_ids
            )[:, -1, :]

    @torch.inference_mode()
    def generate_prompts(
        self,
        prompts,
        *,
        batch_size=32,
        max_tokens,
        temperature=0.0,
        top_k=None,
        seed=42,
        max_length_delta=64,
        use_cuda_graphs=True,
        completion_check_interval=16,
    ):
        """Generate one completion for each independent prompt.

        Models may opt into an exact packed prefill capability. Other models use
        equal-length prefill groups whose populated caches are collated into one
        ragged decode batch. Both paths prevent padding or neighboring prompts
        from entering model-specific recurrent state.
        """
        if not prompts:
            return []
        if batch_size <= 0 or max_tokens <= 0:
            raise ValueError("batch_size and max_tokens must be positive")
        if max_length_delta < 0:
            raise ValueError("max_length_delta must be non-negative")
        if any(not p or not isinstance(p[0], int) for p in prompts):
            raise ValueError("prompts must be non-empty lists of token ids")

        # Ragged decode needs an explicit per-row position interface. Models
        # without one remain fully supported through exact-length buckets.
        supports_ragged_positions = (
            "position_ids" in self._forward_parameters
            or self._forward_accepts_kwargs
        )
        effective_length_delta = (
            max_length_delta if supports_ragged_positions else 0
        )
        ordered = sorted(enumerate(prompts), key=lambda item: len(item[1]))
        buckets = []
        bucket = []
        bucket_min = None
        for item in ordered:
            length = len(item[1])
            if bucket and (
                len(bucket) >= batch_size
                or length - bucket_min > effective_length_delta
            ):
                buckets.append(bucket)
                bucket = []
                bucket_min = None
            if bucket_min is None:
                bucket_min = length
            bucket.append(item)
        if bucket:
            buckets.append(bucket)

        outputs = [None] * len(prompts)
        for bucket_index, items in enumerate(buckets):
            bucket_prompts = [prompt for _, prompt in items]
            generated = self._generate_prompt_bucket(
                bucket_prompts,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed + bucket_index,
                use_cuda_graphs=use_cuda_graphs,
                completion_check_interval=completion_check_interval,
            )
            for (original_index, _), result in zip(items, generated):
                outputs[original_index] = result
        return outputs

    def _generate_prompt_bucket(
        self, prompts, *, max_tokens, temperature, top_k, seed,
        use_cuda_graphs, completion_check_interval,
    ):
        device = self.model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        total_cache_len = max(map(len, prompts)) + max_tokens
        decode_cache = self._allocate_cache(len(prompts), total_cache_len, dtype)
        logits_by_row = [None] * len(prompts)

        supports_packed_prefill = getattr(
            self.model, "supports_packed_prefill", False
        )
        if callable(supports_packed_prefill):
            supports_packed_prefill = supports_packed_prefill()

        if supports_packed_prefill:
            lengths = torch.tensor(
                [len(prompt) for prompt in prompts],
                dtype=torch.int32,
                device=device,
            )
            cu_seqlens = torch.cat((
                lengths.new_zeros(1),
                lengths.cumsum(0, dtype=torch.int32),
            ))
            flat_tokens = [token for prompt in prompts for token in prompt]
            ids = torch.tensor([flat_tokens], dtype=torch.long, device=device)
            sequence_starts = torch.repeat_interleave(cu_seqlens[:-1], lengths)
            position_ids = (
                torch.arange(len(flat_tokens), device=device) - sequence_starts
            ).unsqueeze(0)
            # In packed mode these are absolute positions in the flattened row.
            last_token_positions = cu_seqlens[1:].to(torch.long) - 1
            logits = self._forward(
                ids,
                decode_cache,
                logit_positions=last_token_positions,
                position_ids=position_ids,
                cu_seqlens=cu_seqlens,
            )[:, -1, :]
        else:
            # Universal exact fallback. Only equal lengths are padded together,
            # so model-specific recurrent state can never cross prompt borders.
            rows_by_length = {}
            for row, prompt in enumerate(prompts):
                rows_by_length.setdefault(len(prompt), []).append(row)
            for length, rows in rows_by_length.items():
                prefill_cache = self._allocate_cache(len(rows), length, dtype)
                ids = torch.tensor(
                    [prompts[row] for row in rows], dtype=torch.long, device=device
                )
                group_logits = self._forward(
                    ids, prefill_cache, logit_positions=-1
                )[:, -1, :]
                for source_row, target_row in enumerate(rows):
                    decode_cache.copy_row_from(prefill_cache, source_row, target_row)
                    logits_by_row[target_row] = group_logits[source_row]

            logits = torch.stack(logits_by_row)
        graph_runner = None
        if use_cuda_graphs and device.type == "cuda":
            try:
                graph_runner = _CudaGraphDecode(self, decode_cache, dtype)
                decode_cache = graph_runner.cache
            except Exception as exc:
                # Some third-party recurrent kernels are not graph-capture safe.
                # Falling back preserves correctness and still keeps batching.
                if not getattr(self, "_reported_cuda_graph_failure", False):
                    print(f"CUDA graph decode unavailable; using eager decode: {exc}")
                    self._reported_cuda_graph_failure = True

        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        eos_ids = sorted(self.tokenizer.get_eos_token_ids())
        eos = torch.tensor(eos_ids, dtype=torch.long, device=device)
        fallback_eos = eos[0]
        completed = torch.zeros(len(prompts), dtype=torch.bool, device=device)
        generated_columns = []

        for step in range(max_tokens):
            next_ids = sample_next_token(logits, rng, temperature, top_k)[:, 0]
            next_ids = torch.where(completed, fallback_eos, next_ids)
            generated_columns.append(next_ids)
            completed |= (next_ids[:, None] == eos[None, :]).any(dim=1)

            should_check = (
                (step + 1) % max(1, completion_check_interval) == 0
                or step + 1 == max_tokens
            )
            if should_check and bool(completed.all().item()):
                break
            if step + 1 == max_tokens:
                break

            ids = next_ids.unsqueeze(1)
            if graph_runner is None:
                position_ids = decode_cache.cache_seqlens.to(torch.long).unsqueeze(1)
                logits = self._forward(
                    ids, decode_cache, position_ids=position_ids
                )[:, -1, :]
            else:
                logits = graph_runner(ids)[:, -1, :]

        generated = torch.stack(generated_columns, dim=1).cpu().tolist()
        results = []
        eos_set = set(eos_ids)
        for prompt, tokens in zip(prompts, generated):
            stop = next(
                (index for index, token in enumerate(tokens) if token in eos_set),
                len(tokens),
            )
            results.append(prompt + tokens[:stop])
        return results

    def generate_batch(self, tokens, tools=None, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that returns the final token sequences.

        Returns:
            results: list[list[int]] — generated token sequences (prompt + response, EOS excluded)
            masks:   list[list[int]] — per-token masks (1 = sampled, 0 = forced / prompt)
        """
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples

        eos_ids = self.tokenizer.get_eos_token_ids()

        for token_column, token_masks in self.generate(tokens, tools=tools, num_samples=num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token in eos_ids:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break

        return results, masks


if __name__ == "__main__":
    """
    Quick inline test: verify that the naive model.generate() and Engine.generate()
    produce identical token sequences.
    """
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/dummy-model-path-replace-me"
    model, tokenizer, meta = load_pretrained_hf(model_path, device, phase="eval")
    kwargs = dict(max_tokens=64, temperature=0.0)
    prompt_tokens = tokenizer.encode("The chemical formula of water is")

    # Reference: model.generate()
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    for token in model.generate(prompt_tokens, **kwargs):
        generated_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens

    # Engine.generate()
    generated_tokens = []
    engine = Engine(model, tokenizer)
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in engine.generate(prompt_tokens, num_samples=1, **kwargs):
        token = token_column[0]
        generated_tokens.append(token)
        print(tokenizer.decode([token]), end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")

    for i in range(min(len(reference_ids), len(generated_tokens))):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
