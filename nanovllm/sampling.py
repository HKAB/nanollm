from __future__ import annotations

import torch
import torch.nn.functional as F

from nanovllm.request import Request


@torch.inference_mode()
def sample(logits: torch.Tensor, requests: list[Request], generators):
    """Sample one token per row and optionally retain its scalar logprob."""

    sampled_tokens = []
    token_logprobs = []
    for row, request in enumerate(requests):
        row_logits = logits[row].float()
        params = request.sampling
        if params.temperature == 0:
            token = torch.argmax(row_logits)
            if request.output.logprobs:
                logprob = F.log_softmax(row_logits, dim=-1).gather(0, token.view(1)).squeeze(0)
        elif params.top_k is not None and params.top_k > 0:
            k = min(params.top_k, row_logits.numel())
            values, indices = torch.topk(row_logits, k)
            log_probs = F.log_softmax(values / params.temperature, dim=-1)
            local = torch.multinomial(
                log_probs.exp(), 1, generator=generators[request.request_id]
            )[0]
            token = indices.gather(0, local.view(1)).squeeze(0)
            if request.output.logprobs:
                logprob = log_probs.gather(0, local.view(1)).squeeze(0)
        else:
            log_probs = F.log_softmax(row_logits / params.temperature, dim=-1)
            token = torch.multinomial(
                log_probs.exp(), 1, generator=generators[request.request_id]
            )[0]
            if request.output.logprobs:
                logprob = log_probs.gather(0, token.view(1)).squeeze(0)
        sampled_tokens.append(token)
        token_logprobs.append(
            logprob.detach() if request.output.logprobs else None
        )
    # One device-to-host transfer for the whole scheduler tick.
    return torch.stack(sampled_tokens).tolist(), token_logprobs

