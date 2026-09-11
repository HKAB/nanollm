from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count

import torch

from nanovllm.config import OutputSpec, SamplingParams


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


_REQUEST_IDS = count()


@dataclass
class Request:
    prompt_token_ids: list[int]
    sampling: SamplingParams
    output: OutputSpec
    request_id: int = field(default_factory=lambda: next(_REQUEST_IDS))
    status: RequestStatus = RequestStatus.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    block_table: list[int] = field(default_factory=list)
    state_slot: int | None = None
    token_logprobs: list[torch.Tensor] = field(default_factory=list)
    sampling_hidden_states: list[torch.Tensor] = field(default_factory=list)
    sampling_logits: list[torch.Tensor] = field(default_factory=list)
    finish_reason: str | None = None

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def length(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

