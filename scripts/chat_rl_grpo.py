"""
Clipped group-relative policy optimization.

1 GPU:
python -m scripts.chat_rl_grpo

8 GPUs:
torchrun --standalone --nproc_per_node=8 -m scripts.chat_rl_grpo --run=default
"""

import argparse
import copy
import itertools
import math
import os
import re
from pathlib import Path
import sys
from contextlib import contextmanager
from time import perf_counter
from types import SimpleNamespace

import torch
import torch.distributed as dist
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def policy_terms(logits, targets, advantages, old_logprobs, *, ratio_level="sequence", clip_eps=0.2):
    """Clipped surrogate per completion, with entropy over sampled positions."""
    mask = targets >= 0
    lengths = mask.sum(-1).clamp_min(1)
    log_probs = logits.float().log_softmax(dim=-1)
    token_logprobs = log_probs.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    log_ratio = (token_logprobs - old_logprobs.detach()).masked_fill(~mask, 0)
    if ratio_level != "token":
        log_ratio = log_ratio.sum(-1)
        if ratio_level == "sequence":
            log_ratio = log_ratio / lengths
    else:
        advantages = advantages[:, None]

    # Select min(ratio * A, clipped_ratio * A) before exponentiating.
    # A negative advantage reverses the ordering of the two ratios.
    clipped_log_ratio = log_ratio.clamp(math.log1p(-clip_eps), math.log1p(clip_eps))
    sign = advantages.sign()
    signed_log_ratio = log_ratio.masked_fill(advantages == 0, 0) * sign
    effective_log_ratio = sign * torch.minimum(signed_log_ratio, sign * clipped_log_ratio)
    objective = effective_log_ratio.exp() * advantages
    if ratio_level == "token":
        objective = (objective * mask).sum(-1) / lengths

    with torch.no_grad():
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        entropy = (entropy * mask).sum(-1) / lengths
    return -objective.masked_fill(~mask.any(-1), 0), entropy

def main():
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/nanollm/inductor"))
    from torch._inductor import config as inductor_config

    inductor_config.fx_graph_cache = True

    from nanollm.checkpoint_manager import load_model, load_pretrained_hf, save_checkpoint
    from nanollm.common import (
        DummyWandb,
        autodetect_device_type,
        compute_cleanup,
        compute_init,
        get_base_dir,
        print0,
    )
    from nanovllm import EngineConfig, LLMEngine, OutputSpec, SamplingParams
    from tasks.sft.gsm8k import GSM8K

    # -----------------------------------------------------------------------------
    # CLI arguments
    parser = argparse.ArgumentParser(description="Clipped group-relative policy optimization")
    # Logging
    parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
    parser.add_argument("--sample-every", type=int, default=5, help="log one training rollout on rounds 1, N, 2N, ... (0 disables)")
    parser.add_argument("--sample-max-chars", type=int, default=800, help="maximum displayed characters per sample field")
    # Runtime
    parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
    # Model loading
    parser.add_argument("--model-source", choices=("base", "sft", "rl"),
                        help="native checkpoint stage (default: sft)")
    parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
    parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
    parser.add_argument("--hf-model", help="Hugging Face repo ID or local safetensors directory")
    parser.add_argument("--hf-backend", choices=("native", "transformers"), default="native",
                        help="load a registered native architecture by default; transformers is for comparison")
    parser.add_argument("--hf-lr", type=float, default=1e-6, help="AdamW learning rate for --hf-model")
    # Training horizon
    parser.add_argument("--max-steps", type=int, default=None, help="cap rollout rounds for smoke tests")
    parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K")
    # Batch sizes / sampling
    parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
    parser.add_argument("--rollout-prompts", "--examples-per-step", type=int, default=16,
                        help="total prompts per rollout round across ranks")
    parser.add_argument("--num-samples", type=int, default=8, help="completions per prompt")
    parser.add_argument("--minibatch-size", type=int, default=16,
                        help="completions per rank per optimizer update")
    parser.add_argument("--update-epochs", type=int, default=1, help="passes over each fixed rollout batch")
    parser.add_argument("--ratio-level", choices=("sequence", "token", "sequence-sum"), default="sequence",
                        help="GSPO geometric mean, token GRPO, or raw sequence likelihood ratio")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="ratio clipping around 1")
    # Generation
    parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
    parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
    parser.add_argument("--top-k", type=int, default=0, help="top-k sampling (0 = disabled)")
    # Optimization
    parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
    parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
    parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
    parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
    # Evaluation / checkpointing
    parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
    parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
    parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
    args = parser.parse_args()
    if args.num_samples < 2 or args.device_batch_size < 1 or args.rollout_prompts < 1:
        parser.error("num-samples must be >= 2 and batch sizes must be positive")
    if args.minibatch_size < 1 or args.update_epochs < 1 or not 0 < args.clip_eps < 1:
        parser.error("minibatch-size and update-epochs must be positive; clip-eps must be in (0, 1)")
    if args.temperature != 1.0 or args.top_k != 0:
        parser.error("matching rollout and training distributions requires --temperature=1 and --top-k=0")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("max-steps must be positive")
    if args.hf_model and any(value is not None for value in (args.model_source, args.model_tag, args.model_step)):
        parser.error("--hf-model cannot be combined with --model-source, --model-tag, or --model-step")
    if not args.hf_model and args.hf_backend != "native":
        parser.error("--hf-backend transformers requires --hf-model")
    use_transformers = bool(args.hf_model and args.hf_backend == "transformers")
    if args.sample_every < 0 or args.sample_max_chars < 1:
        parser.error("sample-every must be nonnegative and sample-max-chars must be positive")
    user_config = vars(args).copy()
    # -----------------------------------------------------------------------------

    # Init compute/precision
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.

    # wandb logging init
    use_dummy_wandb = args.run == "dummy" or not master_process
    wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanollm-rl", name=args.run, config=user_config)

    # Init model and tokenizer
    if use_transformers:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
        model = AutoModelForCausalLM.from_pretrained(args.hf_model, dtype=torch.float32).to(device).eval()
    elif args.hf_model:
        model_dir = Path(args.hf_model).expanduser()
        if not model_dir.is_dir():
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(
                args.hf_model, allow_patterns=["config.json", "tokenizer.json", "*.safetensors"],
            )
        model, tokenizer, meta = load_pretrained_hf(model_dir, device, phase="eval")
        # Match the Transformers path's FP32 trainable weights and AdamW.
        model.float()
    else:
        model, tokenizer, meta = load_model(
            args.model_source or "sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step,
        )

    old_policy = copy.deepcopy(model).requires_grad_(False).eval()
    engine = None if use_transformers else LLMEngine(
        old_policy, tokenizer, EngineConfig(max_num_seqs=args.device_batch_size),
    )


    rollout_backend = "transformers" if engine is None else "nanovllm"
    print0(f"Rollout backend: {rollout_backend} | max sequences per generation call: {args.device_batch_size}")

    def render_prompt(conversation):
        if not use_transformers:
            return tokenizer.render_for_completion(conversation, enable_thinking=None)
        if tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                conversation["messages"][:-1], tokenize=False, add_generation_prompt=True,
            )
            # Qwen templates can prefill an opening tag or an empty reasoning block.
            text = re.sub(r"<think>\s*(?:</think>\s*)?\Z", "", text)
        else:
            question = conversation["messages"][-2]["content"]
            text = f"Question: {question}\nAnswer: "
        return tokenizer.encode(text, add_special_tokens=False)


    def policy_logits(inputs):
        if not use_transformers:
            return model(inputs)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            return model(inputs, use_cache=False).logits


    def release_rollout_cache():
        if engine is not None:
            engine.release_cache()


    def generate_rollouts(prompt, *, num_samples, max_tokens, temperature=1.0, top_k=0,
                            seed=42, logprobs=False):
        if use_transformers:
            ids = torch.tensor([prompt], device=device)
            eos_id = tokenizer.eos_token_id
            # Use an explicit config so checkpoint defaults cannot change the sampling distribution.
            from transformers import GenerationConfig

            config = GenerationConfig(
                do_sample=True, temperature=temperature, top_k=top_k, top_p=1.0,
                max_new_tokens=max_tokens, num_return_sequences=num_samples,
                eos_token_id=eos_id, pad_token_id=eos_id,
                return_dict_in_generate=True, output_scores=logprobs,
            )
            cuda_devices = [device.index] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=cuda_devices), torch.autocast(
                device.type, dtype=torch.bfloat16, enabled=device.type == "cuda",
            ):
                torch.random.default_generator.manual_seed(seed)
                if cuda_devices:
                    torch.cuda.manual_seed(seed)
                generated = old_policy.generate(ids, attention_mask=torch.ones_like(ids), generation_config=config)
            completions = generated.sequences[:, len(prompt):]
            sampled = None
            if logprobs:
                sampled = torch.stack([
                    score.float().log_softmax(-1).gather(1, completions[:, i:i + 1]).squeeze(1)
                    for i, score in enumerate(generated.scores)
                ], dim=1)
            outputs = []
            for i, tokens in enumerate(completions.tolist()):
                if eos_id in tokens:
                    tokens = tokens[:tokens.index(eos_id) + 1]
                outputs.append(SimpleNamespace(
                    output_token_ids=tokens,
                    token_logprobs=None if sampled is None else sampled[i, :len(tokens)],
                ))
            return outputs
        return engine.generate(
            prompt,
            sampling=SamplingParams(
                num_samples=num_samples, max_tokens=max_tokens,
                temperature=temperature, top_k=top_k, seed=seed,
            ),
            output=OutputSpec(logprobs=logprobs),
        )

    train_task = GSM8K(subset="main", split="train", format_prompt=True)
    val_task = GSM8K(subset="main", split="test", format_prompt=True) if args.eval_every > 0 and args.eval_examples > 0 else None
    num_steps = (len(train_task) // args.rollout_prompts) * args.num_epochs
    if args.max_steps is not None:
        num_steps = min(num_steps, args.max_steps)

    @contextmanager
    def measure_phase(timings, name, *, cuda=False):
        if cuda:
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            yield
            end.record()
            timings[name].append((start, end))
        else:
            start = perf_counter()
            yield
            timings[name].append(perf_counter() - start)


    def summarize_timings(timings):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        totals = torch.tensor([
            sum(start.elapsed_time(end) / 1000 for start, end in samples)
            if samples and isinstance(samples[0], tuple) else sum(samples)
            for samples in timings.values()
        ], device=device, dtype=torch.float32)
        maxima = totals.clone()
        if ddp:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
            totals /= ddp_world_size
        return {
            f"timing/{name}_{stat}_s": value
            for stat, values in (("mean", totals.tolist()), ("max", maxima.tolist()))
            for name, value in zip(timings, values)
        }


    @torch.no_grad()
    def collect_rollouts(round_idx, rank_indices, timings, *, log_sample=False):
        sequences, old_logprobs, rewards, advantages = [], [], [], []
        sample = None
        for prompt_idx in range(prompts_per_rank):
            example_idx = next(rank_indices)
            conversation = train_task[example_idx]
            prompt = render_prompt(conversation)
            conversation = {**conversation, "completion_prompt": tokenizer.decode(prompt)}
            group_rewards = []
            for offset in range(0, args.num_samples, args.device_batch_size):
                with measure_phase(timings, "rollout"):
                    outputs = generate_rollouts(
                        prompt, num_samples=min(args.device_batch_size, args.num_samples - offset),
                        max_tokens=args.max_new_tokens, logprobs=True,
                        seed=hash((round_idx, example_idx, offset)) & 0x7FFFFFFF,
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                with measure_phase(timings, "reward"):
                    completions = [tokenizer.decode(output.output_token_ids) for output in outputs]
                    group_rewards.extend(train_task.reward(conversation, text) for text in completions)
                if log_sample and prompt_idx == 0 and offset == 0:
                    sample = {
                        "example": example_idx,
                        "prompt": conversation["completion_prompt"],
                        "completion": completions[0],
                        "expected_answer": conversation.get("answer", ""),
                        "completion_tokens": len(outputs[0].output_token_ids),
                    }
                for output in outputs:
                    sequences.append(torch.tensor(prompt + output.output_token_ids, device=device))
                    old_logprobs.append(output.token_logprobs)

            group_rewards = torch.tensor(group_rewards, dtype=torch.float32, device=device)
            rewards.append(group_rewards)
            advantages.append((group_rewards - group_rewards.mean()) / (group_rewards.std(unbiased=False) + 1e-4))

        release_rollout_cache()
        return sequences, old_logprobs, torch.cat(rewards), torch.cat(advantages), sample

    def collate_rollouts(sequences, old_logprobs, indices):
        inputs = [sequences[i][:-1] for i in indices]
        targets, behavior_logprobs = [], []
        for i in indices:
            sampled_logprobs = old_logprobs[i]
            target = sequences[i][1:].clone()
            prompt_length = target.numel() - sampled_logprobs.numel()
            target[:prompt_length] = -1
            targets.append(target)
            behavior_logprobs.append(torch.nn.functional.pad(sampled_logprobs, (prompt_length, 0)))
        pad = torch.nn.utils.rnn.pad_sequence
        return (
            pad(inputs, batch_first=True, padding_value=0),
            pad(targets, batch_first=True, padding_value=-1),
            pad(behavior_logprobs, batch_first=True),
        )

    # -----------------------------------------------------------------------------
    # Simple evaluation loop for GSM8K pass@k
    def run_eval(task, max_examples, num_samples):
        for idx in range(ddp_rank, min(max_examples, len(task)), ddp_world_size):
            conversation = task[idx]
            prompt = render_prompt(conversation)
            outputs = generate_rollouts(prompt, num_samples=num_samples, max_tokens=256, top_k=50)
            yield [task.evaluate(conversation, tokenizer.decode(output.output_token_ids)) for output in outputs]

    # -----------------------------------------------------------------------------
    # Training loop

    # Init the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.hf_lr, weight_decay=args.weight_decay) if args.hf_model else model.setup_optimizer(
        unembedding_lr=args.unembedding_lr,
        embedding_lr=args.embedding_lr,
        matrix_lr=args.matrix_lr,
        weight_decay=args.weight_decay,
    )

    # Set the initial learning rate as a fraction of the base learning rate
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * (1.0 if args.hf_model else args.init_lr_frac)
        group["initial_lr"] = group["lr"]

    # Learning rate scheduler: simple rampdown to zero over num_steps
    def get_lr_multiplier(it):
        lrm = 1.0 - it / num_steps
        return lrm

    if args.rollout_prompts % ddp_world_size:
        parser.error("rollout-prompts must be divisible by world size")
    prompts_per_rank = args.rollout_prompts // ddp_world_size
    rank_indices = itertools.cycle(range(ddp_rank, len(train_task), ddp_world_size))
    print0(f"{num_steps} rounds | {args.rollout_prompts * args.num_samples} completions per round")

    for step in range(num_steps):
        old_policy.load_state_dict(model.state_dict())
        if engine is not None:
            engine.notify_weights_updated()

        if val_task is not None and step % args.eval_every == 0:
            passk = torch.zeros(args.device_batch_size, device=device) # pass@k for k=1..device_batch_size
            records = list(run_eval(val_task, args.eval_examples, args.device_batch_size))
            release_rollout_cache()
            for k in range(1, args.device_batch_size + 1):
                passk[k - 1] = sum(any(outcomes[:k]) for outcomes in records)
            num_records = torch.tensor(len(records), dtype=torch.long, device=device)
            if ddp:
                dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
                dist.all_reduce(passk, op=dist.ReduceOp.SUM)
            passk = passk / num_records.item() # normalize by the total number of records
            print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
            print0(f"Step {step} | {', '.join(print_passk)}")
            log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
            wandb_run.log({
                "step": step,
                **log_passk,
            })

        timings = {name: [] for name in ("rollout", "reward", "forward", "backward")}
        round_number = step + 1
        log_sample = master_process and args.sample_every > 0 and (
            round_number == 1 or round_number % args.sample_every == 0
        )
        sequences, old_logprobs, rewards, advantages, sample = collect_rollouts(
            step, rank_indices, timings, log_sample=log_sample,
        )
        num_rollouts = len(sequences)
        metrics = torch.zeros(2, device=device)
        lrm = get_lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm

        for _ in range(args.update_epochs):
            order = torch.randperm(num_rollouts).tolist()
            for start in range(0, num_rollouts, args.minibatch_size):
                minibatch = order[start:start + args.minibatch_size]
                for offset in range(0, len(minibatch), args.device_batch_size):
                    indices = minibatch[offset:offset + args.device_batch_size]
                    inputs, targets, behavior_logprobs = collate_rollouts(sequences, old_logprobs, indices)
                    with measure_phase(timings, "forward", cuda=device.type == "cuda"):
                        losses, entropy = policy_terms(
                            policy_logits(inputs), targets, advantages[indices], behavior_logprobs,
                            ratio_level=args.ratio_level, clip_eps=args.clip_eps,
                        )
                        loss = losses.sum() / len(minibatch)
                    with measure_phase(timings, "backward", cuda=device.type == "cuda"):
                        loss.backward()
                    metrics[0] += losses.detach().sum()
                    metrics[1] += entropy.sum()

                if ddp:
                    # Average before clipping so each rank clips the same gradient.
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                model.zero_grad(set_to_none=True)

        stats = torch.stack((
            metrics[0] / args.update_epochs, metrics[1] / args.update_epochs,
            rewards.sum(), advantages.sum(), advantages.square().sum(),
            rewards.new_tensor(sum(lp.numel() for lp in old_logprobs)),
        ))
        if ddp:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss, entropy, reward, advantage_mean, advantage_sq_mean, length = (
            stats / (num_rollouts * ddp_world_size)
        ).tolist()
        logs = {
            "step": step,
            "loss": loss,
            "reward": reward,
            "advantage_mean": advantage_mean,
            "advantage_std": max(0.0, advantage_sq_mean - advantage_mean ** 2) ** 0.5,
            "entropy": entropy,
            "completion_length": length,
            "lrm": lrm,
        }
        timing_logs = summarize_timings(timings)
        rollout_tokens = int(stats[-1].item())
        rollout_seconds = timing_logs["timing/rollout_max_s"]
        rollout_tok_per_sec = rollout_tokens / rollout_seconds if rollout_seconds > 0 else 0.0
        logs.update(timing_logs)
        logs.update({
            "rollout/generated_tokens": rollout_tokens,
            "rollout/tokens_per_second": rollout_tok_per_sec,
        })
        timing_text = " | ".join(f"{name} time: {timing_logs[f'timing/{name}_max_s']:.3f}s" for name in timings)
        print0(f"Round {step}/{num_steps} | loss: {loss:.6f} | reward: {reward:.4f} | entropy: {entropy:.4f} | {rollout_tok_per_sec:.1f} rollout tok/s | {timing_text}")
        if sample is not None:
            for field in ("prompt", "completion", "expected_answer"):
                value = str(sample[field])
                if len(value) > args.sample_max_chars:
                    budget = args.sample_max_chars - 1
                    tail = budget // 2 if field == "completion" else 0
                    value = value[:budget - tail] + "…" + value[len(value) - tail:]
                sample[field] = value
            sample["reward"] = rewards[0].item()
            sample["advantage"] = advantages[0].item()
            print0(
                f"Rollout sample | round {step} | example {sample['example']} "
                f"| tokens: {sample['completion_tokens']} | expected: {sample['expected_answer']!r} "
                f"| reward: {sample['reward']:.4f} | advantage: {sample['advantage']:.4f}\n"
                f"  Prompt: {sample['prompt']!r}\n"
                f"  Completion: {sample['completion']!r}"
            )
            if not use_dummy_wandb:
                logs["rollout/sample"] = wandb.Table(columns=list(sample), data=[list(sample.values())])
        wandb_run.log(logs)

        # Master process saves the model once in a while. Skip first step. Save last step.
        if master_process and args.save_every > 0 and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
            base_dir = get_base_dir()
            checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", f"step_{step:05d}")
            if use_transformers:
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
            else:
                save_checkpoint(
                    checkpoint_dir, step, model.state_dict(), None,
                    {"model_config": meta.get("model_config", {})}, tokenizer=tokenizer,
                )
            print(f"✅ Saved model checkpoint to {checkpoint_dir}")

    # Log to report
    from nanollm.report import get_report

    get_report().log(section="Chat RL", data=[
        user_config, # CLI args
    ])

    wandb_run.finish() # wandb run finish
    compute_cleanup()


if __name__ == "__main__":
    main()
