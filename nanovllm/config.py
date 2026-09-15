from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int = 64
    temperature: float = 1.0
    top_k: int | None = None
    num_samples: int = 1
    ignore_eos: bool = False
    seed: int = 42

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")


@dataclass(frozen=True)
class OutputSpec:
    """Controls what crosses the model-runner boundary.

    ``sampling_hidden_states[t]`` is the pre-LM-head vector whose logits were
    used to choose ``output_token_ids[t]``.
    """

    token_ids: bool = True
    sampled_mask: bool = False
    logprobs: bool = False
    hidden_states: bool = False
    logits: bool = False
    hidden_state_dtype: torch.dtype | None = None
    hidden_state_device: Literal["cuda", "cpu"] | None = None


@dataclass(frozen=True)
class EngineConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 16_384
    max_model_len: int | None = None
    block_size: int = 256
    num_kv_blocks: int | None = None
    gpu_memory_utilization: float = 0.90
    enable_prefix_caching: bool = False
    kv_cache_memory_gib: float | None = None

    def __post_init__(self):
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.num_kv_blocks is not None and self.num_kv_blocks <= 0:
            raise ValueError("num_kv_blocks must be positive")
        if self.kv_cache_memory_gib is not None and self.kv_cache_memory_gib <= 0:
            raise ValueError("kv_cache_memory_gib must be positive")
        if self.num_kv_blocks is not None and self.kv_cache_memory_gib is not None:
            raise ValueError(
                "num_kv_blocks and kv_cache_memory_gib are mutually exclusive"
            )
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
