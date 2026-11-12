from __future__ import annotations

import torch
from torch import nn
import pytest

from nanollm.inference import LayerCacheSpec, ModelCacheSpec
from nanovllm import EngineConfig, LLMEngine, OutputSpec, SamplingParams
from nanovllm.cache import PagedModelCache, _available_kv_cache_bytes


class TinyTokenizer:
    eos_token_id = 7


def test_auto_cache_budget_reclaims_allocator_memory_before_measuring(monkeypatch):
    from contextlib import nullcontext

    calls = []
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("reclaim"))

    def memory_info(device):
        assert calls == ["reclaim"]
        return 400, 1000

    monkeypatch.setattr(torch.cuda, "mem_get_info", memory_info)
    assert _available_kv_cache_bytes(torch.device("cuda:0"), 0.9) == 300
    calls.clear()
    assert _available_kv_cache_bytes(torch.device("cuda:0"), 0.5) == 0


class TinyModel(nn.Module):
    """Predict (last token + 1) and expose that token as its hidden state."""

    supports_packed_prefill = True

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.forward_calls = 0

    def get_inference_cache_spec(self):
        return ModelCacheSpec(())

    def forward(
        self,
        input_ids,
        *,
        kv_cache,
        position_ids=None,
        cu_seqlens=None,
        logit_positions=None,
        return_hidden_states=False,
    ):
        self.forward_calls += 1
        hidden = input_ids.float().unsqueeze(-1)
        if cu_seqlens is not None:
            selected = input_ids[0, logit_positions]
            hidden = selected.float().view(-1, 1, 1)
            kv_cache.cache_seqlens.copy_(cu_seqlens[1:] - cu_seqlens[:-1])
        elif isinstance(logit_positions, int):
            selected = input_ids[:, logit_positions]
            hidden = selected.float().view(-1, 1, 1)
            kv_cache.advance(input_ids.size(1))
        else:
            selected = input_ids[:, -1]
            kv_cache.advance(input_ids.size(1))
        logits = torch.full((selected.numel(), 1, 8), -20.0)
        logits[torch.arange(selected.numel()), 0, (selected + 1) % 8] = 20.0
        if return_hidden_states:
            return logits, hidden
        return logits


def test_generate_is_the_only_batched_api_and_returns_requested_payloads():
    engine = LLMEngine(
        TinyModel(),
        TinyTokenizer(),
        EngineConfig(
            max_num_seqs=2,
            max_num_batched_tokens=16,
            block_size=2,
            num_kv_blocks=8,
        ),
    )
    outputs = engine.generate(
        [[1, 2], [3]],
        sampling=SamplingParams(max_tokens=2, temperature=0),
        output=OutputSpec(logprobs=True, hidden_states=True),
    )

    assert [output.output_token_ids for output in outputs] == [[3, 4], [4, 5]]
    assert [output.finish_reason for output in outputs] == ["length", "length"]
    assert outputs[0].token_logprobs.shape == (2,)
    # The first vector came from the final prompt token; the second came from
    # the first generated token.
    assert outputs[0].sampling_hidden_states[:, 0].tolist() == [2.0, 3.0]
    assert outputs[0].logits is None


def test_generate_expands_num_samples_and_can_return_tokens_only():
    model = TinyModel()
    engine = LLMEngine(
        model, TinyTokenizer(),
        EngineConfig(max_num_seqs=3, block_size=2, num_kv_blocks=8),
    )
    outputs = engine.generate(
        [1],
        sampling=SamplingParams(max_tokens=1, temperature=0, num_samples=3),
    )
    assert len(outputs) == 3
    assert model.forward_calls == 1  # identical prompt is prefetched once
    assert [output.output_token_ids for output in outputs] == [[2], [2], [2]]
    assert all(output.token_logprobs is None for output in outputs)
    assert all(output.sampling_hidden_states is None for output in outputs)

    engine.release_cache()
    assert engine.cache is None
    rebuilt = engine.generate(
        [2], SamplingParams(max_tokens=1, temperature=0)
    )
    assert rebuilt[0].output_token_ids == [3]


def test_hybrid_prefix_cache_cannot_be_enabled_incompletely():
    with pytest.raises(NotImplementedError, match="recurrent snapshots"):
        LLMEngine(
            TinyModel(),
            TinyTokenizer(),
            EngineConfig(num_kv_blocks=4, enable_prefix_caching=True),
        )


@pytest.mark.parametrize("fail", [False, True])
def test_engine_context_releases_cache_after_generation(monkeypatch, fail):
    engine = LLMEngine(
        TinyModel(), TinyTokenizer(),
        EngineConfig(max_num_seqs=1, block_size=2, num_kv_blocks=2),
    )
    if fail:
        def broken_prefill(batch):
            raise RuntimeError("forward failed")
        monkeypatch.setattr(engine.runner, "prefill", broken_prefill)
        with pytest.raises(RuntimeError, match="forward failed"):
            with engine:
                engine.generate([1], SamplingParams(max_tokens=2))
    else:
        with engine:
            engine.generate([1], SamplingParams(max_tokens=2))
    assert engine.cache is None
    assert engine.runner is None
    assert engine.scheduler is None
    assert engine.block_manager is None


def test_kv_cache_size_options_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        EngineConfig(num_kv_blocks=4, kv_cache_memory_gib=1.0)


def test_kv_cache_memory_gib_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        EngineConfig(kv_cache_memory_gib=0)


class RecurrentSpecModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def get_inference_cache_spec(self):
        return ModelCacheSpec((LayerCacheSpec(
            layer_idx=0,
            kind="recurrent",
            recurrent_shape=(2, 3, 4),
            convolution_shapes=((5, 2),),
        ),))


class PagedSpecModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def get_inference_cache_spec(self):
        return ModelCacheSpec((LayerCacheSpec(
            layer_idx=0,
            kind="paged_kv",
            num_kv_heads=1,
            key_head_dim=1,
            value_head_dim=1,
        ),))


def test_kv_cache_memory_gib_is_converted_to_whole_blocks():
    # float32, two tokens per block, one K and one V element per token = 16 bytes.
    cache = PagedModelCache(
        PagedSpecModel(),
        max_num_seqs=1,
        block_size=2,
        num_kv_blocks=None,
        kv_cache_memory_gib=(3 * 16) / 1024**3,
        gpu_memory_utilization=0.9,
    )
    assert cache.num_kv_blocks == 3


def test_recurrent_rows_are_gathered_and_committed_by_stable_slot():
    cache = PagedModelCache(
        RecurrentSpecModel(),
        max_num_seqs=3,
        block_size=2,
        num_kv_blocks=2,
        kv_cache_memory_gib=None,
        gpu_memory_utilization=0.9,
    )
    first = cache.allocate_slot()
    second = cache.allocate_slot()
    cache.begin_forward(
        [second, first], torch.tensor([[0], [1]], dtype=torch.int32)
    )
    cache.linear_recurrent_states[0][0].fill_(4)
    cache.linear_recurrent_states[0][1].fill_(9)
    cache.linear_conv_states[0][0].fill_(5)
    cache.linear_conv_states[0][1].fill_(8)
    cache.cache_seqlens.copy_(torch.tensor([11, 12], dtype=torch.int32))
    cache.end_forward()

    cache.begin_forward(
        [first, second], torch.tensor([[1], [0]], dtype=torch.int32)
    )
    assert torch.all(cache.linear_recurrent_states[0][0] == 9)
    assert torch.all(cache.linear_recurrent_states[0][1] == 4)
    assert torch.all(cache.linear_conv_states[0][0] == 8)
    assert torch.all(cache.linear_conv_states[0][1] == 5)
    assert cache.cache_seqlens.tolist() == [12, 11]
    cache.end_forward()


def test_qwen_hybrid_rollout_with_fp32_weights_and_bf16_activations(monkeypatch):
    from nanollm.models import qwen

    monkeypatch.setattr(qwen, "COMPUTE_DTYPE", torch.bfloat16)
    config = qwen.Qwen3_5ModelConfig(
        vocab_size=32, context_length=16, emb_dim=16, n_heads=2, n_layers=2,
        hidden_dim=32, head_dim=8, qk_norm=True, n_kv_groups=1,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
    )
    model = qwen.Qwen3_5Model(config).float().eval()
    model.init_weights()
    engine = LLMEngine(
        model, TinyTokenizer(), EngineConfig(max_num_seqs=1, block_size=4, num_kv_blocks=2),
    )
    assert engine.cache.dtype == torch.bfloat16
    outputs = engine.generate(
        [1, 2], SamplingParams(max_tokens=3, temperature=0, ignore_eos=True),
        OutputSpec(logprobs=True),
    )
    assert outputs[0].token_logprobs.shape == (3,)
    assert torch.isfinite(outputs[0].token_logprobs).all()
    model(torch.tensor([[1, 2, 3]])).float().square().mean().backward()
    conv_grad = model.transformer.h[0].linear_attn.conv1d.weight.grad
    assert conv_grad.dtype == torch.float32
    assert torch.isfinite(conv_grad).all() and conv_grad.count_nonzero() > 0


def test_queued_prompt_groups_keep_output_order_across_decode_batches():
    engine = LLMEngine(
        TinyModel(), TinyTokenizer(),
        EngineConfig(max_num_seqs=3, block_size=2, num_kv_blocks=12),
    )
    outputs = engine.generate(
        [[1], [3], [5]],
        SamplingParams(num_samples=2, max_tokens=2, temperature=1, top_k=1, ignore_eos=True),
        OutputSpec(logprobs=True),
    )
    assert [output.prompt_token_ids for output in outputs] == [[1], [1], [3], [3], [5], [5]]
    assert [output.output_token_ids for output in outputs] == [
        [2, 3], [2, 3], [4, 5], [4, 5], [6, 7], [6, 7],
    ]
    assert all(torch.equal(output.token_logprobs, torch.zeros(2)) for output in outputs)
    assert engine.last_generation_stats["decode_steps"] == 2
    assert engine.last_generation_stats["decode_tokens"] == 6
