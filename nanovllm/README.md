# nanovllm

`nanovllm` is the offline inference runtime for models implemented in
`nanollm.models`. It does not load or reimplement model architectures and it
never falls back to `nanollm.engine.Engine`.

```python
from nanovllm import EngineConfig, LLMEngine, OutputSpec, SamplingParams

engine = LLMEngine(model, tokenizer, EngineConfig(num_kv_blocks=1024))
outputs = engine.generate(
    [[1, 2, 3], [4, 5]],
    sampling=SamplingParams(max_tokens=128, temperature=0.8, num_samples=4),
    output=OutputSpec(hidden_states=True),
)
```

The default payload contains generated token IDs only. Scalar token logprobs,
pre-head sampling hidden states, sampled masks, and full logits are independent
opt-ins. `sampling_hidden_states[t]` is aligned with the logits used to select
`output_token_ids[t]`.

## Cache behavior

Qwen3.5 full-attention layers use paged K/V storage. Gated DeltaNet recurrent
and causal-convolution states live in stable per-request slots and are gathered
and committed for each scheduled batch. Running requests are not preempted.
Identical prompts admitted in the same prefill batch are computed once, then
their KV, recurrent, and convolution state is cloned into independently writable
request storage. Ref-counted sharing and copy-on-write pages are future memory
optimizations.

Cross-request prefix caching is intentionally disabled. Correct hybrid prefix
reuse must snapshot both paged K/V and GDN/convolution state at the same token
boundary. Passing `enable_prefix_caching=True` therefore raises instead of
silently producing incomplete cache hits.

When `num_kv_blocks` is omitted, the runtime estimates a cache size from free
device memory. Server and benchmark runs should set it explicitly until warmup
peak-memory profiling is implemented.

## Server validation checklist

1. Compare greedy tokens and per-step logits with `nanollm.engine.Engine`.
2. Compare packed and one-request-at-a-time prefill results.
3. Test ragged decode batches crossing page boundaries.
4. Test Qwen3.5 GDN and convolution state across long decode runs.
5. Test optional hidden-state output followed by `model.project_logits()`.
6. Measure gather/scatter overhead for recurrent state.
7. Tune `num_kv_blocks` after recording warmup peak memory.
