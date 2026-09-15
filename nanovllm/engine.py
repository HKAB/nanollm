from __future__ import annotations

from dataclasses import dataclass

import torch

from nanovllm.block_manager import BlockManager
from nanovllm.cache import PagedModelCache
from nanovllm.config import EngineConfig, OutputSpec, SamplingParams
from nanovllm.request import Request
from nanovllm.runner import ModelRunner
from nanovllm.sampling import sample
from nanovllm.scheduler import Scheduler


@dataclass
class GenerationOutput:
    request_id: int
    prompt_token_ids: list[int]
    output_token_ids: list[int] | None
    sampled_mask: torch.Tensor | None
    token_logprobs: torch.Tensor | None
    sampling_hidden_states: torch.Tensor | None
    logits: torch.Tensor | None
    finish_reason: str


class LLMEngine:
    """Offline, synchronous continuous-batching engine."""

    def __init__(self, model, tokenizer, config: EngineConfig | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EngineConfig()
        if self.config.enable_prefix_caching:
            raise NotImplementedError(
                "hybrid prefix caching requires synchronized KV and recurrent snapshots"
            )
        self.max_model_len = self.config.max_model_len or getattr(
            getattr(model, "config", None), "context_length", None
        )
        self.cache = None
        self.block_manager = None
        self.scheduler = None
        self.runner = None
        self._build_runtime()

    def _build_runtime(self):
        self.cache = PagedModelCache(
            self.model,
            max_num_seqs=self.config.max_num_seqs,
            block_size=self.config.block_size,
            num_kv_blocks=self.config.num_kv_blocks,
            kv_cache_memory_gib=self.config.kv_cache_memory_gib,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
        )
        self.block_manager = BlockManager(
            self.cache.num_kv_blocks, self.config.block_size
        )
        self.scheduler = Scheduler(self.config, self.block_manager, self.cache)
        self.runner = ModelRunner(self.model, self.cache, self.config.block_size)

    def release_cache(self):
        """Release inference state so training can reclaim the device memory."""
        if self.scheduler is not None and not self.scheduler.is_finished():
            raise RuntimeError("cannot release cache during active generation")
        self.runner = None
        self.scheduler = None
        self.block_manager = None
        self.cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _eos_ids(self) -> set[int]:
        if hasattr(self.tokenizer, "get_eos_token_ids"):
            return set(self.tokenizer.get_eos_token_ids())
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            return set()
        return set(eos if isinstance(eos, (list, tuple, set)) else [eos])

    @staticmethod
    def _normalize_prompts(prompts) -> list[list[int]]:
        if not prompts:
            return []
        if isinstance(prompts[0], int):
            prompts = [prompts]
        normalized = [list(prompt) for prompt in prompts]
        if any(not prompt or not isinstance(prompt[0], int) for prompt in normalized):
            raise ValueError("prompts must be non-empty token-id sequences")
        return normalized

    def generate(
        self,
        prompts,
        sampling: SamplingParams | None = None,
        output: OutputSpec | None = None,
    ) -> list[GenerationOutput]:
        if self.scheduler is None:
            self._build_runtime()
        if not self.scheduler.is_finished():
            raise RuntimeError("this engine already has active requests")
        sampling = sampling or SamplingParams()
        output = output or OutputSpec()
        prompts = self._normalize_prompts(prompts)
        if not prompts:
            return []
        if self.max_model_len is not None:
            for prompt in prompts:
                if len(prompt) + sampling.max_tokens > self.max_model_len:
                    raise ValueError(
                        "prompt plus max_tokens exceeds the model context length"
                    )

        requests = []
        generators = {}
        for prompt_index, prompt in enumerate(prompts):
            for sample_index in range(sampling.num_samples):
                request = Request(prompt.copy(), sampling, output)
                requests.append(request)
                self.scheduler.add(request)
                generator = torch.Generator(device=self.cache.device)
                generator.manual_seed(
                    sampling.seed + prompt_index * sampling.num_samples + sample_index
                )
                generators[request.request_id] = generator

        eos_ids = self._eos_ids()
        try:
            while not self.scheduler.is_finished():
                batch = self.scheduler.schedule_prefill()
                if batch:
                    logits, hidden = self.runner.prefill(batch)
                else:
                    batch = self.scheduler.schedule_decode()
                    if not batch:
                        raise RuntimeError("no request can make progress")
                    logits, hidden = self.runner.decode(batch)

                token_ids, logprobs = sample(logits, batch, generators)
                for row, (request, token_id, logprob) in enumerate(
                    zip(batch, token_ids, logprobs)
                ):
                    request.output_token_ids.append(token_id)
                    if request.output.logprobs:
                        request.token_logprobs.append(logprob)
                    if request.output.hidden_states:
                        state = hidden[row].detach()
                        if output.hidden_state_dtype is not None:
                            state = state.to(output.hidden_state_dtype)
                        if output.hidden_state_device is not None:
                            state = state.to(output.hidden_state_device)
                        request.sampling_hidden_states.append(state)
                    if request.output.logits:
                        request.sampling_logits.append(logits[row].detach())

                    if not sampling.ignore_eos and token_id in eos_ids:
                        self.scheduler.finish(request, "eos")
                    elif len(request.output_token_ids) >= sampling.max_tokens:
                        self.scheduler.finish(request, "length")
        except Exception:
            self.scheduler.abort_all()
            raise

        results = []
        for request in requests:
            results.append(GenerationOutput(
                request_id=request.request_id,
                prompt_token_ids=request.prompt_token_ids,
                output_token_ids=(
                    request.output_token_ids if output.token_ids else None
                ),
                sampled_mask=(
                    torch.ones(len(request.output_token_ids), dtype=torch.bool)
                    if output.sampled_mask else None
                ),
                token_logprobs=(
                    torch.stack(request.token_logprobs)
                    if output.logprobs else None
                ),
                sampling_hidden_states=(
                    torch.stack(request.sampling_hidden_states)
                    if output.hidden_states else None
                ),
                logits=(
                    torch.stack(request.sampling_logits)
                    if output.logits else None
                ),
                finish_reason=request.finish_reason,
            ))
        generated_lengths = [len(request.output_token_ids) for request in requests]
        self.last_generation_stats = {
            "num_prompts": len(prompts),
            "num_sequences": len(requests),
            "prompt_tokens": sum(len(request.prompt_token_ids) for request in requests),
            "generated_tokens": sum(generated_lengths),
            "average_prompt_tokens": (
                sum(len(request.prompt_token_ids) for request in requests)
                / len(requests)
            ),
            "average_generated_tokens": sum(generated_lengths) / len(requests),
            "max_generated_tokens": max(generated_lengths),
        }
        return results

    def notify_weights_updated(self):
        """Invalidate reusable state after an optimizer update.

        Version one has no persistent prefix cache, so an idle engine has no
        state to invalidate. Reject updates during active generation to prevent
        a sequence from spanning two model versions.
        """
        if self.scheduler is not None and not self.scheduler.is_finished():
            raise RuntimeError("cannot update weights during active generation")
