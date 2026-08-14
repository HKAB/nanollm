"""Reusable inference caches.

The base cache contains only the key/value state used by ordinary decoder-only
transformers. Models with additional recurrent state should subclass
``KVCache`` and override ``copy_row_from``/``reset`` as needed.
"""

from __future__ import annotations

import torch


class KVCache:
    """Pre-allocated, batch-aware key/value cache."""

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers,
                 device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        self.k_cache = torch.zeros(
            num_layers, batch_size, seq_len, num_heads, head_dim,
            device=device, dtype=dtype,
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.cache_seqlens = torch.zeros(
            batch_size, dtype=torch.int32, device=device
        )
        self.has_previous_state = False

    @property
    def device(self):
        return self.k_cache.device

    def reset(self):
        self.cache_seqlens.zero_()
        self.has_previous_state = False

    def get_pos(self):
        """Return the common position; reject ragged batches explicitly."""
        first = int(self.cache_seqlens[0].item())
        if not torch.all(self.cache_seqlens == first):
            raise ValueError("ragged cache has no single position")
        return first

    def get_layer_cache(self, layer_idx):
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        self.cache_seqlens += num_tokens
        self.has_previous_state = True

    def copy_row_from(self, other, src_row, dst_row):
        """Copy one populated sequence, including model-specific state."""
        length = int(other.cache_seqlens[src_row].item())
        if length > self.max_seq_len:
            raise ValueError(f"cache row of length {length} exceeds {self.max_seq_len}")
        self.k_cache[:, dst_row, :length].copy_(
            other.k_cache[:, src_row, :length]
        )
        self.v_cache[:, dst_row, :length].copy_(
            other.v_cache[:, src_row, :length]
        )
        self.cache_seqlens[dst_row] = length
        self.has_previous_state = self.has_previous_state or other.has_previous_state

    def copy_from(self, other):
        if self.batch_size != other.batch_size:
            raise ValueError("cache batch sizes must match")
        for row in range(self.batch_size):
            self.copy_row_from(other, row, row)

    def prefill(self, other):
        """Compatibility helper: broadcast a one-row cache to this batch."""
        if other.batch_size != 1:
            raise ValueError("prefill source must have batch size 1")
        if torch.any(self.cache_seqlens != 0):
            raise ValueError("cannot prefill a non-empty cache")
        for row in range(self.batch_size):
            self.copy_row_from(other, 0, row)

