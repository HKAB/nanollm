from __future__ import annotations

import torch
import torch.nn.functional as F

from nanovllm.request import Request


@torch.inference_mode()
def sample(logits: torch.Tensor, requests: list[Request], generators):
    """Sample one token per row and optionally retain its scalar logprob."""

    token_ids = []
    token_logprobs = []
    for row, request in enumerate(requests):
        row_logits = logits[row].float()
        params = request.sampling
        if params.temperature == 0:
            token = torch.argmax(row_logits)
            if request.output.logprobs:
                logprob = F.log_softmax(row_logits, dim=-1)[token]
        elif params.top_k is not None and params.top_k > 0:
            k = min(params.top_k, row_logits.numel())
            values, indices = torch.topk(row_logits, k)
            log_probs = F.log_softmax(values / params.temperature, dim=-1)
            local = torch.multinomial(
                log_probs.exp(), 1, generator=generators[request.request_id]
            )[0]
            token = indices[local]
            if request.output.logprobs:
                logprob = log_probs[local]
        else:
            log_probs = F.log_softmax(row_logits / params.temperature, dim=-1)
            token = torch.multinomial(
                log_probs.exp(), 1, generator=generators[request.request_id]
            )[0]
            if request.output.logprobs:
                logprob = log_probs[token]
        token_ids.append(int(token.item()))
        token_logprobs.append(
            logprob.detach() if request.output.logprobs else None
        )
    return token_ids, token_logprobs

