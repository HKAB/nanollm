from __future__ import annotations

from collections import deque

from nanovllm.block_manager import BlockManager
from nanovllm.cache import PagedModelCache
from nanovllm.config import EngineConfig
from nanovllm.request import Request, RequestStatus


class Scheduler:
    """Synchronous continuous-batching scheduler without preemption."""

    def __init__(
        self,
        config: EngineConfig,
        block_manager: BlockManager,
        cache: PagedModelCache,
    ):
        self.config = config
        self.block_manager = block_manager
        self.cache = cache
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add(self, request: Request):
        prompt_len = len(request.prompt_token_ids)
        if self.config.max_model_len is not None and prompt_len >= self.config.max_model_len:
            raise ValueError("prompt leaves no room for a generated token")
        if self.block_manager.blocks_for(prompt_len) > self.block_manager.num_blocks:
            raise ValueError("prompt is larger than the complete paged KV cache")
        self.waiting.append(request)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def schedule_prefill(self) -> list[Request]:
        scheduled = []
        tokens = 0
        available_seq_slots = self.config.max_num_seqs - len(self.running)
        while self.waiting and len(scheduled) < available_seq_slots:
            request = self.waiting[0]
            prompt_len = len(request.prompt_token_ids)
            if prompt_len > self.config.max_num_batched_tokens:
                raise ValueError(
                    f"prompt of {prompt_len} tokens exceeds max_num_batched_tokens"
                )
            if tokens + prompt_len > self.config.max_num_batched_tokens:
                break
            if not self.block_manager.can_allocate(prompt_len):
                break
            if self.cache.num_free_slots == 0:
                break
            self.waiting.popleft()
            request.state_slot = self.cache.allocate_slot()
            self.block_manager.allocate(request)
            request.status = RequestStatus.RUNNING
            self.running.append(request)
            scheduled.append(request)
            tokens += prompt_len
        return scheduled

    def schedule_decode(self) -> list[Request]:
        scheduled = []
        for request in self.running:
            if len(scheduled) >= self.config.max_num_seqs:
                break
            if not self.block_manager.ensure_append(request):
                continue
            scheduled.append(request)
        if self.running and not scheduled:
            raise RuntimeError(
                "paged KV cache is exhausted and preemption is disabled"
            )
        return scheduled

    def finish(self, request: Request, reason: str):
        request.status = RequestStatus.FINISHED
        request.finish_reason = reason
        self.block_manager.free(request)
        self.cache.free_slot(request.state_slot)
        request.state_slot = None
        self.running.remove(request)

    def abort_all(self):
        """Release resident resources after a failed generation call."""
        for request in list(self.running):
            self.block_manager.free(request)
            if request.state_slot is not None:
                self.cache.free_slot(request.state_slot)
                request.state_slot = None
        self.running.clear()
        self.waiting.clear()
