"""On-policy distillation with token-level importance sampling."""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.distributed as dist

from nanollm.checkpoint_manager import load_model, load_pretrained_hf, save_checkpoint
from nanollm.common import (
    COMPUTE_DTYPE,
    DummyWandb,
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    get_dist_info,
    print0,
)
from nanovllm import EngineConfig, LLMEngine, OutputSpec, SamplingParams


def parse_model_kwargs(value):
    try:
        kwargs = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(f"Expected a JSON object: {error.msg}") from error
    if not isinstance(kwargs, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object")
    return kwargs


def add_model_args(parser, prefix=""):
    parser.add_argument(f"--{prefix}model-source", choices=("base", "sft", "rl", "opd"))
    parser.add_argument(f"--{prefix}model-tag")
    parser.add_argument(f"--{prefix}model-step", type=int)
    parser.add_argument(f"--{prefix}hf-model", help="Hugging Face repo ID or local snapshot")
    parser.add_argument(f"--{prefix}model-kwargs", type=parse_model_kwargs, default={},
                        help="JSON object of keyword arguments for the model loader")


def model_args(args, prefix=""):
    return [getattr(args, prefix + name) for name in ("model_source", "model_tag", "model_step", "hf_model")]


def load_native_model(source, tag, step, hf_model, device, **kwargs):
    if not hf_model:
        return load_model(source or "sft", device, phase="eval", model_tag=tag, step=step, **kwargs)

    model_dir = Path(hf_model).expanduser()
    if not model_dir.is_dir():
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(
            hf_model, allow_patterns=["config.json", "tokenizer.json", "*.safetensors"],
        )
    return load_pretrained_hf(model_dir, device, phase="eval", **kwargs)


def load_teacher(args, device):
    if args.teacher_hf_backend == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.teacher_hf_model).backend_tokenizer
        kwargs = args.teacher_model_kwargs.copy()
        kwargs.setdefault("dtype", COMPUTE_DTYPE if device.type == "cuda" else torch.float32)
        teacher = AutoModelForCausalLM.from_pretrained(args.teacher_hf_model, **kwargs).to(device)
        return teacher, tokenizer

    teacher, tokenizer, _ = load_native_model(*model_args(args, "teacher_"), device, **args.teacher_model_kwargs)
    return teacher, tokenizer.tokenizer


def render_prompt(tokenizer, prompt, prompt_format, max_seq_len):
    if prompt_format == "raw":
        return tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    conversation = {"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""},
    ]}
    return tokenizer.render_for_completion(conversation, enable_thinking=None, max_tokens=max_seq_len)


def collate_rollouts(outputs, pad_id, device):
    sequences, targets, sampled_logprobs = [], [], []
    for output in outputs:
        prompt = output.prompt_token_ids
        completion = output.output_token_ids
        sequences.append(torch.tensor((prompt + completion)[:-1], device=device))
        # The last prompt position predicts the first completion token.
        targets.append(torch.tensor([-1] * (len(prompt) - 1) + completion, device=device))
        sampled_logprobs.append(torch.nn.functional.pad(output.token_logprobs, (len(prompt) - 1, 0)))

    pad = torch.nn.utils.rnn.pad_sequence
    return (
        pad(sequences, batch_first=True, padding_value=pad_id),
        pad(targets, batch_first=True, padding_value=-1),
        pad(sampled_logprobs, batch_first=True),
    )


def completion_logprobs(logits, targets, completion_mask):
    selected_logits = logits[completion_mask].float()
    selected_ids = targets[completion_mask].unsqueeze(-1)
    return selected_logits.log_softmax(dim=-1).gather(-1, selected_ids).squeeze(-1)


def importance_sampling(current_logprobs, sampled_logprobs, advantages):
    """Return per-token losses and ratios for sampled completion tokens."""
    ratio = (current_logprobs - sampled_logprobs.detach()).exp()
    return -ratio * advantages.detach(), ratio


def train(args, ddp, rank, world_size, device):
    from datasets import load_dataset

    if args.prompt_file:
        file_format = "parquet" if Path(args.prompt_file).suffix == ".parquet" else "json"
        dataset = load_dataset(file_format, data_files=args.prompt_file, split="train")
    else:
        dataset = load_dataset(args.dataset, split=args.dataset_split)
    dataset = dataset.select_columns(["prompt"]).shuffle(seed=args.seed)
    rounds_per_epoch = len(dataset) // args.rollout_prompts
    if rounds_per_epoch == 0:
        raise ValueError("The dataset must contain at least --rollout-prompts rows")
    num_steps = min(args.max_steps, rounds_per_epoch * args.num_epochs)
    prompts_per_rank = args.rollout_prompts // world_size

    student, tokenizer, meta = load_native_model(*model_args(args), device, **args.model_kwargs)
    student.float()
    teacher, teacher_tokenizer = load_teacher(args, device)
    teacher.eval().requires_grad_(False)
    if tokenizer.tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
        raise ValueError("OPD requires identical student and teacher token-to-ID mappings")

    teacher_config = teacher.config
    if args.teacher_hf_backend == "transformers":
        teacher_config = teacher_config.get_text_config()
    if teacher_config.vocab_size < student.config.vocab_size:
        raise ValueError("The teacher must score every token ID the student can sample")
    max_seq_len = min(
        args.max_seq_len,
        student.config.context_length,
        getattr(teacher_config, "context_length", None)
        or getattr(teacher_config, "max_position_embeddings", None)
        or args.max_seq_len,
    )
    if args.max_new_tokens >= max_seq_len:
        raise ValueError("--max-new-tokens must leave room for the prompt in both models")

    pad_id = tokenizer.get_pad_token_id()
    concurrency = min(args.rollout_batch_size, prompts_per_rank * args.num_samples)
    block_size = EngineConfig.block_size
    engine_config = EngineConfig(
        max_num_seqs=concurrency,
        max_model_len=max_seq_len,
        num_kv_blocks=concurrency * math.ceil(max_seq_len / block_size),
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    wandb_run = DummyWandb()
    if rank == 0 and args.run != "dummy":
        import wandb

        run_config = {key: value for key, value in vars(args).items() if not key.endswith("model_kwargs")}
        wandb_run = wandb.init(project="nanollm-opd", name=args.run, config=run_config)

    print0(f"{num_steps} OPD steps | {args.rollout_prompts * args.num_samples} completions per step")
    engine = LLMEngine(student, tokenizer, engine_config)
    for step in range(1, num_steps + 1):
        start = ((step - 1) % rounds_per_epoch) * args.rollout_prompts + rank * prompts_per_rank
        prompt_texts = dataset[start:start + prompts_per_rank]["prompt"]
        prompts = [render_prompt(tokenizer, prompt, args.prompt_format, max_seq_len) for prompt in prompt_texts]
        # Synchronous inference uses the current weights. Free KV state before backpropagation.
        with torch.no_grad():
            outputs = engine.generate(
                prompts,
                sampling=SamplingParams(
                    max_tokens=args.max_new_tokens, num_samples=args.num_samples,
                    seed=args.seed + ((step - 1) * world_size + rank) * prompts_per_rank * args.num_samples,
                ),
                output=OutputSpec(logprobs=True),
            )
        engine.release_cache()

        token_count = torch.tensor(sum(len(out.output_token_ids) for out in outputs), device=device)
        if ddp:
            dist.all_reduce(token_count)
        totals = torch.zeros(3, device=device)
        for offset in range(0, len(outputs), args.device_batch_size):
            inputs, targets, sampled = collate_rollouts(outputs[offset:offset + args.device_batch_size], pad_id, device)
            mask = targets >= 0
            sampled = sampled[mask]
            with torch.no_grad():
                if args.teacher_hf_backend == "transformers":
                    logits = teacher(input_ids=inputs, use_cache=False).logits
                else:
                    logits = teacher(inputs)
                advantages = completion_logprobs(logits, targets, mask) - sampled
                del logits

            current = completion_logprobs(student(inputs), targets, mask)
            # Prepare for asynchronous training, not useful if synchronous training.
            losses, ratios = importance_sampling(current, sampled, advantages)
            loss_sum = losses.sum()
            # Gradient averaging across ranks gives a global completion-token mean.
            (loss_sum * world_size / token_count).backward()
            totals += torch.stack((loss_sum.detach(), -advantages.sum(), ratios.detach().sum()))

        if ddp:
            for parameter in student.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
            dist.all_reduce(totals)
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss, reverse_kl, ratio = (totals / token_count).tolist()
        logs = dict(step=step, loss=loss, teacher_kl=reverse_kl, importance_ratio=ratio,
                    completion_tokens=token_count.item(), grad_norm=grad_norm.item())
        print0(f"Step {step}/{num_steps} | loss {loss:.6f} | sampled reverse KL {reverse_kl:.6f} "
               f"| mean ratio {ratio:.6f} | grad norm {logs['grad_norm']:.6f} | tokens {logs['completion_tokens']}")
        wandb_run.log(logs)
        if rank == 0 and args.save_every and (step % args.save_every == 0 or step == num_steps):
            checkpoint_dir = Path(get_base_dir()) / "chatopd_checkpoints" / args.save_tag
            save_checkpoint(checkpoint_dir, step, student.state_dict(), None, meta, tokenizer=tokenizer)

    wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_args(parser)
    add_model_args(parser, "teacher-")
    parser.add_argument("--teacher-hf-backend", choices=("native", "transformers"), default="native")
    parser.add_argument("--dataset", default="hkab/vi-ifeval")
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--prompt-file", help="Local JSON, JSONL, or parquet with a prompt column")
    parser.add_argument("--prompt-format", choices=("chat", "raw"), default="chat")
    parser.add_argument("--device-type", choices=("cuda", "cpu", "mps"))
    parser.add_argument("--rollout-prompts", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--device-batch-size", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", default="dummy")
    parser.add_argument("--save-tag", default="opd")
    parser.add_argument("--save-every", type=int, default=0, help="Checkpoint interval, 0 disables saving")
    args = parser.parse_args()

    for prefix in ("", "teacher_"):
        source, tag, step, hf_model = model_args(args, prefix)
        if hf_model and any(value is not None for value in (source, tag, step)):
            parser.error(f"Choose either a model source/tag/step or --{prefix.replace('_', '-')}hf-model")
    if not (args.teacher_hf_model or args.teacher_model_tag):
        parser.error("Specify --teacher-hf-model or --teacher-model-tag")
    if args.teacher_hf_backend == "transformers" and not args.teacher_hf_model:
        parser.error("--teacher-hf-backend transformers requires --teacher-hf-model")
    for name in ("rollout_prompts", "num_samples", "rollout_batch_size", "device_batch_size",
                 "max_seq_len", "max_new_tokens", "num_epochs", "max_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("--lr must be positive and --weight-decay must be nonnegative")
    if args.save_every < 0:
        parser.error("--save-every must be nonnegative")

    _, _, _, world_size = get_dist_info()
    device_type = args.device_type or autodetect_device_type()
    if world_size > 1 and device_type != "cuda":
        parser.error("Multi-process OPD requires CUDA")
    if args.rollout_prompts % world_size:
        parser.error("--rollout-prompts must be divisible by world size")
    _, rank, _, world_size, device = compute_init(device_type)
    try:
        train(args, dist.is_initialized(), rank, world_size, device)
    finally:
        compute_cleanup()


if __name__ == "__main__":
    main()
