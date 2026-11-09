from __future__ import annotations

from collections import deque

import torch

from nanollm.inference import ModelCacheSpec


class PagedModelCache:
    """Paged attention storage plus slot-indexed recurrent model state.

    The object presents the cache attributes currently consumed by nanollm's
    Qwen3.5 implementation. Active recurrent rows are gathered before a model
    call and committed back afterward, allowing the scheduler to batch any set
    of stable request slots.
    """

    is_paged = True

    def __init__(
        self,
        model,
        *,
        max_num_seqs: int,
        block_size: int,
        num_kv_blocks: int | None,
        kv_cache_memory_gib: float | None = None,
        gpu_memory_utilization: float,
    ):
        get_spec = getattr(model, "get_inference_cache_spec", None)
        if get_spec is None:
            raise TypeError(
                f"{type(model).__name__} must implement get_inference_cache_spec()"
            )
        spec = get_spec()
        if not isinstance(spec, ModelCacheSpec):
            raise TypeError("get_inference_cache_spec() must return ModelCacheSpec")

        self.spec = spec
        parameter = next(model.parameters())
        self.device = parameter.device
        self.dtype = spec.activation_dtype or parameter.dtype
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.n_layers = max((s.layer_idx for s in spec.layers), default=-1) + 1

        self._length_pool = torch.zeros(
            max_num_seqs, dtype=torch.int32, device=self.device
        )
        self._free_slots = deque(range(max_num_seqs))

        self._recurrent_pool: dict[int, torch.Tensor] = {}
        self._conv_pool: dict[int, tuple[torch.Tensor, ...]] = {}
        for layer in spec.layers:
            if layer.kind != "recurrent":
                continue
            recurrent_dtype = layer.recurrent_dtype or self.dtype
            convolution_dtype = layer.convolution_dtype or self.dtype
            self._recurrent_pool[layer.layer_idx] = torch.empty(
                max_num_seqs,
                *layer.recurrent_shape,
                device=self.device,
                dtype=recurrent_dtype,
            )
            self._conv_pool[layer.layer_idx] = tuple(
                torch.empty(
                    max_num_seqs,
                    *shape,
                    device=self.device,
                    dtype=convolution_dtype,
                )
                for shape in layer.convolution_shapes
            )

        paged_layers = [layer for layer in spec.layers if layer.kind == "paged_kv"]
        bytes_per_block = sum(
            block_size
            * layer.num_kv_heads
            * (layer.key_head_dim + layer.value_head_dim)
            * torch.empty((), dtype=self.dtype).element_size()
            for layer in paged_layers
        )
        if num_kv_blocks is None:
            if bytes_per_block == 0:
                num_kv_blocks = max_num_seqs
            elif kv_cache_memory_gib is not None:
                cache_bytes = int(kv_cache_memory_gib * 1024**3)
                num_kv_blocks = cache_bytes // bytes_per_block
            elif self.device.type != "cuda":
                num_kv_blocks = max_num_seqs
            else:
                free, total = torch.cuda.mem_get_info(self.device)
                budget = max(0, int(total * gpu_memory_utilization) - (total - free))
                num_kv_blocks = budget // bytes_per_block
        if num_kv_blocks <= 0:
            raise RuntimeError("no memory remains for paged KV blocks")
        self.num_kv_blocks = num_kv_blocks

        self._k_cache: dict[int, torch.Tensor] = {}
        self._v_cache: dict[int, torch.Tensor] = {}
        for layer in paged_layers:
            self._k_cache[layer.layer_idx] = torch.empty(
                num_kv_blocks,
                block_size,
                layer.num_kv_heads,
                layer.key_head_dim,
                device=self.device,
                dtype=self.dtype,
            )
            self._v_cache[layer.layer_idx] = torch.empty(
                num_kv_blocks,
                block_size,
                layer.num_kv_heads,
                layer.value_head_dim,
                device=self.device,
                dtype=self.dtype,
            )

        self.active_slots: torch.Tensor | None = None
        self.block_tables: torch.Tensor | None = None
        self.cache_seqlens = torch.empty(0, dtype=torch.int32, device=self.device)
        self.batch_size = 0
        self.has_previous_state = False
        self.linear_recurrent_states = [None] * self.n_layers
        self.linear_conv_states = [None] * self.n_layers

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def allocate_slot(self) -> int:
        if not self._free_slots:
            raise RuntimeError("no free recurrent-state slots")
        slot = self._free_slots.popleft()
        self._length_pool[slot] = 0
        for pool in self._recurrent_pool.values():
            pool[slot].zero_()
        for pools in self._conv_pool.values():
            for pool in pools:
                pool[slot].zero_()
        return slot

    def free_slot(self, slot: int):
        self._length_pool[slot] = 0
        self._free_slots.append(slot)

    def clone_sequence_state(
        self,
        *,
        source_slot: int,
        target_slot: int,
        source_blocks: list[int],
        target_blocks: list[int],
    ):
        """Clone one completed prefill into an independently writable request."""
        self._length_pool[target_slot].copy_(self._length_pool[source_slot])
        for pool in self._recurrent_pool.values():
            pool[target_slot].copy_(pool[source_slot])
        for pools in self._conv_pool.values():
            for pool in pools:
                pool[target_slot].copy_(pool[source_slot])
        for source_block, target_block in zip(source_blocks, target_blocks):
            for pool in self._k_cache.values():
                pool[target_block].copy_(pool[source_block])
            for pool in self._v_cache.values():
                pool[target_block].copy_(pool[source_block])

    def begin_forward(self, slots: list[int], block_tables: torch.Tensor):
        self.active_slots = torch.tensor(slots, dtype=torch.long, device=self.device)
        self.block_tables = block_tables.to(device=self.device, dtype=torch.int32)
        self.cache_seqlens = self._length_pool.index_select(0, self.active_slots)
        self.batch_size = len(slots)
        self.has_previous_state = bool(torch.all(self.cache_seqlens > 0).item())
        for layer_idx, state in self._recurrent_pool.items():
            self.linear_recurrent_states[layer_idx] = state.index_select(
                0, self.active_slots
            )
            conv_states = tuple(
                pool.index_select(0, self.active_slots)
                for pool in self._conv_pool[layer_idx]
            )
            self.linear_conv_states[layer_idx] = (
                conv_states[0] if len(conv_states) == 1 else conv_states
            )

    def end_forward(self):
        if self.active_slots is None:
            return
        self._length_pool.index_copy_(0, self.active_slots, self.cache_seqlens)
        for layer_idx, state in self._recurrent_pool.items():
            state.index_copy_(
                0, self.active_slots, self.linear_recurrent_states[layer_idx]
            )
            active_conv = self.linear_conv_states[layer_idx]
            if not isinstance(active_conv, tuple):
                active_conv = (active_conv,)
            for pool, rows in zip(self._conv_pool[layer_idx], active_conv):
                pool.index_copy_(0, self.active_slots, rows)
        self.active_slots = None
        self.block_tables = None

    def get_pos(self) -> int:
        if self.cache_seqlens.numel() == 0:
            return 0
        first = int(self.cache_seqlens[0].item())
        if not torch.all(self.cache_seqlens == first):
            raise ValueError("ragged cache has no single position")
        return first

    def get_layer_cache(self, layer_idx: int):
        return self._k_cache[layer_idx], self._v_cache[layer_idx]

    def advance(self, num_tokens: int):
        self.cache_seqlens += num_tokens
        self.has_previous_state = True

    def store_packed_kv(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ):
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        rows = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=self.device), lengths
        )
        flat = torch.arange(keys.shape[0], device=self.device)
        positions = flat - torch.repeat_interleave(cu_seqlens[:-1], lengths)
        logical_blocks = positions.div(self.block_size, rounding_mode="floor")
        offsets = positions.remainder(self.block_size)
        physical_blocks = self.block_tables[rows, logical_blocks]
        self._k_cache[layer_idx][physical_blocks, offsets] = keys
        self._v_cache[layer_idx][physical_blocks, offsets] = values
