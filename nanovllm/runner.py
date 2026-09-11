from __future__ import annotations

import inspect

import torch

from nanovllm.cache import PagedModelCache
from nanovllm.request import Request


class ModelRunner:
    def __init__(self, model, cache: PagedModelCache, block_size: int):
        self.model = model
        self.cache = cache
        self.block_size = block_size
        try:
            self._forward_parameters = set(inspect.signature(model.forward).parameters)
        except (TypeError, ValueError):
            self._forward_parameters = set()

    def _block_tables(self, requests: list[Request]) -> torch.Tensor:
        width = max(len(request.block_table) for request in requests)
        tables = [
            request.block_table + [-1] * (width - len(request.block_table))
            for request in requests
        ]
        return torch.tensor(tables, dtype=torch.int32, device=self.cache.device)

    def _call_model(self, input_ids, **kwargs):
        wants_hidden = kwargs.pop("wants_hidden")
        if wants_hidden:
            if "return_hidden_states" not in self._forward_parameters:
                raise TypeError(
                    f"{type(self.model).__name__} does not support hidden-state output"
                )
            kwargs["return_hidden_states"] = True
        result = self.model(input_ids, **kwargs)
        if wants_hidden:
            logits, hidden = result
        else:
            logits, hidden = result, None
        return logits.squeeze(1), None if hidden is None else hidden.squeeze(1)

    @torch.inference_mode()
    def prefill(self, requests: list[Request]):
        roots = []
        root_by_prompt = {}
        root_index_by_request = []
        for request in requests:
            key = tuple(request.prompt_token_ids)
            root_index = root_by_prompt.get(key)
            if root_index is None:
                root_index = len(roots)
                root_by_prompt[key] = root_index
                roots.append(request)
            root_index_by_request.append(root_index)

        logits, hidden = self._prefill_unique(roots)
        for request, root_index in zip(requests, root_index_by_request):
            root = roots[root_index]
            if request is root:
                continue
            self.cache.clone_sequence_state(
                source_slot=root.state_slot,
                target_slot=request.state_slot,
                source_blocks=root.block_table,
                target_blocks=request.block_table,
            )
        indices = torch.tensor(
            root_index_by_request, dtype=torch.long, device=logits.device
        )
        expanded_logits = logits.index_select(0, indices)
        expanded_hidden = (
            hidden.index_select(0, indices) if hidden is not None else None
        )
        return expanded_logits, expanded_hidden

    @torch.inference_mode()
    def _prefill_unique(self, requests: list[Request]):
        wants_hidden = any(request.output.hidden_states for request in requests)
        packed = len(requests) > 1
        supports_packed = getattr(self.model, "supports_packed_prefill", False)
        supports_packed = supports_packed() if callable(supports_packed) else supports_packed
        if packed and supports_packed:
            lengths = torch.tensor(
                [len(request.prompt_token_ids) for request in requests],
                dtype=torch.int32,
                device=self.cache.device,
            )
            cu_seqlens = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
            flat_ids = [token for request in requests for token in request.prompt_token_ids]
            input_ids = torch.tensor([flat_ids], dtype=torch.long, device=self.cache.device)
            starts = torch.repeat_interleave(cu_seqlens[:-1], lengths)
            position_ids = (
                torch.arange(len(flat_ids), device=self.cache.device) - starts
            ).unsqueeze(0)
            last_positions = cu_seqlens[1:].long() - 1
            self.cache.begin_forward(
                [request.state_slot for request in requests],
                self._block_tables(requests),
            )
            try:
                return self._call_model(
                    input_ids,
                    kv_cache=self.cache,
                    cu_seqlens=cu_seqlens,
                    position_ids=position_ids,
                    logit_positions=last_positions,
                    wants_hidden=wants_hidden,
                )
            finally:
                self.cache.end_forward()

        logits = []
        hidden = []
        for request in requests:
            input_ids = torch.tensor(
                [request.prompt_token_ids], dtype=torch.long, device=self.cache.device
            )
            position_ids = torch.arange(
                len(request.prompt_token_ids), device=self.cache.device
            ).unsqueeze(0)
            self.cache.begin_forward([request.state_slot], self._block_tables([request]))
            try:
                row_logits, row_hidden = self._call_model(
                    input_ids,
                    kv_cache=self.cache,
                    position_ids=position_ids,
                    logit_positions=-1,
                    wants_hidden=wants_hidden,
                )
            finally:
                self.cache.end_forward()
            logits.append(row_logits[0])
            if wants_hidden:
                hidden.append(row_hidden[0])
        return torch.stack(logits), torch.stack(hidden) if wants_hidden else None

    @torch.inference_mode()
    def decode(self, requests: list[Request]):
        wants_hidden = any(request.output.hidden_states for request in requests)
        input_ids = torch.tensor(
            [[request.last_token] for request in requests],
            dtype=torch.long,
            device=self.cache.device,
        )
        position_ids = torch.tensor(
            [[request.length - 1] for request in requests],
            dtype=torch.long,
            device=self.cache.device,
        )
        self.cache.begin_forward(
            [request.state_slot for request in requests], self._block_tables(requests)
        )
        try:
            return self._call_model(
                input_ids,
                kv_cache=self.cache,
                position_ids=position_ids,
                logit_positions=-1,
                wants_hidden=wants_hidden,
            )
        finally:
            self.cache.end_forward()
