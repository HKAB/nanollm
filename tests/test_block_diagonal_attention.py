"""
Tests for block-diagonal (varlen) attention used by pretokenized SFT.

Run: python -m pytest tests/test_block_diagonal_attention.py -v

- The varlen SDPA-fallback math and the full-attention model equivalence run on CPU.
- The exact hybrid packed-prefill/cache test requires CUDA plus FLA convolution
  and gated-delta kernels, and is skipped otherwise.
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nanollm.models.qwen import (
    HAS_FLA,
    HAS_FLA_CONV,
    Qwen3_5Model,
    Qwen3_5ModelConfig,
)
import nanollm.flash_attention as fa
from nanollm.flash_attention import flash_attn_varlen_func


@pytest.fixture(autouse=True)
def force_sdpa():
    """Run the CPU tests through the SDPA fallback regardless of hardware."""
    saved_override, saved_use = fa._override_impl, fa.USE_FA3
    fa._override_impl, fa.USE_FA3 = "sdpa", False
    yield
    fa._override_impl, fa.USE_FA3 = saved_override, saved_use


def _full_attn_config():
    return Qwen3_5ModelConfig(
        vocab_size=128, context_length=64, emb_dim=32, n_heads=2, n_layers=2,
        hidden_dim=64, head_dim=16, qk_norm=True, n_kv_groups=1, rope_base=10000.0,
        partial_rotary_factor=0.5, layer_types=["full_attention", "full_attention"],
    )


def _pack(tokens, segs, device):
    idx = torch.tensor([tokens], device=device)
    cu = torch.tensor([0] + list(np.cumsum(segs)), dtype=torch.int32, device=device)
    pos = torch.tensor([sum([list(range(s)) for s in segs], [])], device=device)
    return idx, cu, pos


class TestVarlenSDPAFallback:
    def test_matches_per_segment_causal(self):
        torch.manual_seed(0)
        segs = [3, 2]
        total, H, D = sum(segs), 2, 4
        q, k, v = (torch.randn(total, H, D) for _ in range(3))
        cu = torch.tensor([0, 3, 5], dtype=torch.int32)
        out = flash_attn_varlen_func(q, k, v, cu, max_seqlen=3)

        ref = torch.empty_like(out)
        start = 0
        for s in segs:
            qs, ks, vs = (t[start:start + s].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
            r = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)
            ref[start:start + s] = r.squeeze(0).transpose(0, 1)
            start += s
        assert torch.allclose(out, ref, atol=1e-5)


class TestFullAttentionModel:
    def test_selected_logits_match_full_projection(self):
        torch.manual_seed(0)
        model = Qwen3_5Model(_full_attn_config()).eval()
        tokens = torch.tensor([[5, 6, 7], [8, 9, 10]])
        positions = torch.tensor([2, 1])

        with torch.no_grad():
            full = model(tokens)
            selected = model(tokens, logit_positions=positions)
            last = model(tokens, logit_positions=-1)

        expected = full[torch.arange(2), positions].unsqueeze(1)
        assert selected.shape == (2, 1, model.config.vocab_size)
        assert torch.allclose(selected, expected, atol=1e-6, rtol=1e-6)
        assert torch.allclose(last, full[:, -1:, :], atol=1e-6, rtol=1e-6)

    def test_packed_equals_standalone(self):
        torch.manual_seed(0)
        model = Qwen3_5Model(_full_attn_config()).eval()
        A, B = [5, 6, 7], [8, 9]

        def run(tokens, segs):
            idx, cu, pos = _pack(tokens, segs, "cpu")
            with torch.no_grad():
                return model(idx, cu_seqlens=cu, position_ids=pos)

        packed = run(A + B, [len(A), len(B)])
        la = run(A, [len(A)])
        lb = run(B, [len(B)])
        # packed conversations must produce identical logits to standalone (no leakage)
        assert torch.allclose(packed[0, :len(A)], la[0], atol=1e-5)
        assert torch.allclose(packed[0, len(A):], lb[0], atol=1e-5)

    def test_packing_order_invariance(self):
        torch.manual_seed(1)
        model = Qwen3_5Model(_full_attn_config()).eval()
        A, B = [5, 6, 7], [8, 9]
        with torch.no_grad():
            idx_ab, cu_ab, pos_ab = _pack(A + B, [len(A), len(B)], "cpu")
            idx_ba, cu_ba, pos_ba = _pack(B + A, [len(B), len(A)], "cpu")
            ab = model(idx_ab, cu_seqlens=cu_ab, position_ids=pos_ab)
            ba = model(idx_ba, cu_seqlens=cu_ba, position_ids=pos_ba)
        # A's logits identical whether it is packed first or second
        assert torch.allclose(ab[0, :len(A)], ba[0, len(B):], atol=1e-5)
        assert torch.allclose(ab[0, len(A):], ba[0, :len(B)], atol=1e-5)


@pytest.mark.skipif(not (torch.cuda.is_available() and HAS_FLA and HAS_FLA_CONV),
                    reason="exact hybrid packed prefill needs CUDA + FLA convolution")
class TestHybridFLAReset:
    def test_packed_prefill_matches_independent_prefill_and_caches(self):
        dev = "cuda"
        cfg = Qwen3_5ModelConfig(
            vocab_size=128, context_length=128, emb_dim=64, n_heads=4, n_layers=4,
            hidden_dim=128, head_dim=16, qk_norm=True, n_kv_groups=2, rope_base=10000.0,
            partial_rotary_factor=0.5,
            layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
            linear_num_value_heads=4, linear_num_key_heads=4,
            linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
        )
        torch.manual_seed(0)
        model = Qwen3_5Model(cfg).to(dev).eval()
        A, B = list(range(5, 25)), list(range(30, 47))

        def run(tokens, segs, cache_rows):
            idx, cu, pos = _pack(tokens, segs, dev)
            cache = model.create_kv_cache(cache_rows, 64, torch.bfloat16)
            with torch.no_grad():
                logits = model(
                    idx, kv_cache=cache, cu_seqlens=cu, position_ids=pos
                )
            return logits, cache

        packed, packed_cache = run(A + B, [len(A), len(B)], 2)
        standalone_a, cache_a = run(A, [len(A)], 1)
        standalone_b, cache_b = run(B, [len(B)], 1)

        torch.testing.assert_close(packed[0, :len(A)], standalone_a[0])
        torch.testing.assert_close(packed[0, len(A):], standalone_b[0])
        assert packed_cache.cache_seqlens.tolist() == [len(A), len(B)]

        for packed_state, a_state, b_state in zip(
            packed_cache.linear_conv_states,
            cache_a.linear_conv_states,
            cache_b.linear_conv_states,
        ):
            torch.testing.assert_close(packed_state[0], a_state[0])
            torch.testing.assert_close(packed_state[1], b_state[0])
        for packed_state, a_state, b_state in zip(
            packed_cache.linear_recurrent_states,
            cache_a.linear_recurrent_states,
            cache_b.linear_recurrent_states,
        ):
            torch.testing.assert_close(packed_state[0], a_state[0])
            torch.testing.assert_close(packed_state[1], b_state[0])

        full_layers = [i for i, kind in enumerate(cfg.layer_types) if kind == "full_attention"]
        for layer in full_layers:
            torch.testing.assert_close(
                packed_cache.k_cache[layer, 0, :len(A)],
                cache_a.k_cache[layer, 0, :len(A)],
            )
            torch.testing.assert_close(
                packed_cache.k_cache[layer, 1, :len(B)],
                cache_b.k_cache[layer, 0, :len(B)],
            )
            torch.testing.assert_close(
                packed_cache.v_cache[layer, 0, :len(A)],
                cache_a.v_cache[layer, 0, :len(A)],
            )
            torch.testing.assert_close(
                packed_cache.v_cache[layer, 1, :len(B)],
                cache_b.v_cache[layer, 0, :len(B)],
            )

        next_ids = torch.tensor([[70], [71]], device=dev)
        next_positions = torch.tensor([[len(A)], [len(B)]], device=dev)
        with torch.no_grad():
            packed_decode = model(
                next_ids, kv_cache=packed_cache, position_ids=next_positions
            )
            decode_a = model(
                next_ids[:1], kv_cache=cache_a, position_ids=next_positions[:1]
            )
            decode_b = model(
                next_ids[1:], kv_cache=cache_b, position_ids=next_positions[1:]
            )
        torch.testing.assert_close(packed_decode[:1], decode_a)
        torch.testing.assert_close(packed_decode[1:], decode_b)
