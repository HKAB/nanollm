"""Tests for the reusable model cache."""

import torch

from nanollm.cache import KVCache


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
