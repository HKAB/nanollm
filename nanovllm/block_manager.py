from __future__ import annotations

from collections import deque

from nanovllm.request import Request


class BlockManager:
    """Physical paged-KV allocation without cross-request prefix reuse."""

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free = deque(range(num_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        return self.num_free_blocks >= self.blocks_for(num_tokens)

    def allocate(self, request: Request):
        if request.block_table:
            raise ValueError("request already has KV blocks")
        count = self.blocks_for(request.length)
        if self.num_free_blocks < count:
            raise RuntimeError("insufficient paged KV blocks")
        request.block_table.extend(self._free.popleft() for _ in range(count))

    def ensure_append(self, request: Request) -> bool:
        # The last sampled token is appended to the cache by the upcoming
        # decode call. It is already included in request.length.
        required = self.blocks_for(request.length)
        if required <= len(request.block_table):
            return True
        if not self._free:
            return False
        request.block_table.append(self._free.popleft())
        return True

    def free(self, request: Request):
        self._free.extend(request.block_table)
        request.block_table.clear()
