"""Small, engine-neutral contracts exposed by nanollm models.

The training package describes the state required by a model, while an
inference runtime decides how and where that state is allocated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


CacheKind = Literal["paged_kv", "recurrent"]


@dataclass(frozen=True)
class LayerCacheSpec:
    """Inference state owned by one token-mixing layer."""

    layer_idx: int
    kind: CacheKind
    num_kv_heads: int | None = None
    key_head_dim: int | None = None
    value_head_dim: int | None = None
    recurrent_shape: tuple[int, ...] | None = None
    convolution_shapes: tuple[tuple[int, ...], ...] | None = None
    recurrent_dtype: torch.dtype | None = None
    convolution_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class ModelCacheSpec:
    """Complete heterogeneous inference-state description for a model."""

    layers: tuple[LayerCacheSpec, ...]
    activation_dtype: torch.dtype | None = None
