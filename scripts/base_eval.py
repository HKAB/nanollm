"""
Unified evaluation script for base models.

Supports three evaluation modes (comma-separated):
  --eval core    : CORE metric (accuracy on ICL tasks)
  --eval loss     : Bits per byte on train/val splits
  --eval sample  : Generate samples from the model

Default is all three: --eval core,loss,sample

Examples:

    # Evaluate a nanollm model (e.g. d24) using 8 GPUs
    torchrun --nproc_per_node=8 -m scripts.base_eval --model-tag d24 --device-batch-size=16

    # Quick/approximate evaluation using a single GPU
    python -m scripts.base_eval --model-tag d24 --device-batch-size=16 --max-per-task=100 --split-tokens=524288
"""
import argparse
import os
import random
import time

from nanollm.checkpoint_manager import load_pretrained_hf
from nanollm.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_base_dir,
    print0,
)
from nanollm.core_eval import evaluate_task
from nanollm.dataloader import pretrain_loader
from nanollm.engine import Engine
from nanollm.loss_eval import evaluate_loss
from nanollm.report import get_report       
from tasks.pretrain.core import CORE_TASKS, load_core_data

# -----------------------------------------------------------------------------
# CORE evaluation

def evaluate_core(model, tokenizer, device, max_per_task=-1):
    """
    Evaluate a base model on the CORE benchmark.
    Returns dict with results, centered_results, and core_metric.
    """
    data_by_task = load_core_data()

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in CORE_TASKS:
        start_time = time.time()
        label = task.label
        task_meta = {
            'task_type': task.task_type,
            'num_fewshot': task.num_fewshot,
            'continuation_delimiter': task.continuation_delimiter,
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        data = list(data_by_task[label])

        # Shuffle for consistent subsampling when using max_per_task
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        accuracy = evaluate_task(model, tokenizer, data, device, task_meta)
        results[label] = accuracy
        centered_result = (accuracy - task.random_baseline) / (1.0 - task.random_baseline)
        centered_results[label] = centered_result
        elapsed = time.time() - start_time
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {elapsed:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser(description="Base model evaluation")
    parser.add_argument('--eval', type=str, default='core,loss,sample', help='Comma-separated evaluations to run: core,loss,sample (default: all)')
    parser.add_argument('--pretrained-model-path', type=str, required=True, help='path or HF repo id of pretrained model/tokenizer')
    parser.add_argument('--model-tag', type=str, default=None, help='nanollm model tag to identify the checkpoint directory')
    parser.add_argument('--step', type=int, default=None, help='Model step to load (default = last)')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per CORE task (-1 = all)')
    parser.add_argument('--device-batch-size', type=int, default=32, help='Per-device batch size for BPB evaluation')
    parser.add_argument('--dataset-root', type=str, default=None, help='Root dir of pre-tokenized parquet files (required for loss eval)')
    parser.add_argument('--split-tokens', type=int, default=40*524288, help='Number of tokens to evaluate per split for BPB')
    parser.add_argument('--device-type', type=str, default='', help='cuda|cpu|mps (empty = autodetect)')
    args = parser.parse_args()

    # Parse evaluation modes
    eval_modes = set(mode.strip() for mode in args.eval.split(','))
    valid_modes = {'core', 'loss', 'sample'}
    invalid = eval_modes - valid_modes
    if invalid:
        parser.error(f"Invalid eval modes: {invalid}. Valid: {valid_modes}")

    # Distributed / precision setup
    device_type = autodetect_device_type() if args.device_type == '' else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    # Load model and tokenizer
    model, tokenizer, meta = load_pretrained_hf(args.pretrained_model_path, device, phase="eval", model_tag=args.model_tag, step=args.step)
    sequence_len = meta["model_config"].get("context_length", meta["model_config"].get("sequence_len", 1024))
    model_name = f"base_model (step {meta['step']})"
    model_slug = f"base_model_{meta['step']:06d}"

    print0(f"Evaluating model: {model_name}")
    print0(f"Eval modes: {', '.join(sorted(eval_modes))}")

    # Results to log
    core_results = None
    loss_results = {}
    samples = []
    unconditioned_samples = []

    # --- Sampling ---
    if 'sample' in eval_modes:
        print0("\n" + "="*80)
        print0("Model Samples")
        print0("="*80)
        if ddp_rank == 0:
            prompts = [
                "Một cây làm chẳng nên non,",
                "Thấy Tấm bắt được một giỏ đầy, Cám bảo chị:",
                "Trăm năm trong cõi người ta,"
                "Cacbon có 2 hóa trị là"
                "Vừa gà vừa chó bó lại cho tròn 36 con 100 chân chẵn. Hỏi có bao nhiêu con gà, bao nhiêu con chó?"
            ]
            engine = Engine(model, tokenizer)
            print0("\nConditioned samples:")
            for prompt in prompts:
                tokens = tokenizer(prompt)
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                sample_str = tokenizer.decode(sample[0])
                print0("-" * 80)
                print0(sample_str)
                samples.append(sample_str)

            print0("\nUnconditioned samples:")
            tokens = tokenizer("")
            uncond, _ = engine.generate_batch(tokens, num_samples=8, max_tokens=128, temperature=1.0)
            for sample in uncond:
                sample_str = tokenizer.decode(sample)
                print0("-" * 80)
                print0(sample_str)
                unconditioned_samples.append(sample_str)

    # --- BPB evaluation ---
    if 'loss' in eval_modes:
        print0("\n" + "="*80)
        print0("BPB Evaluation")
        print0("="*80)
        assert args.dataset_root is not None, "loss eval requires --dataset-root (pre-tokenized parquet files)"
        tokens_per_step = args.device_batch_size * sequence_len * ddp_world_size
        if args.split_tokens % tokens_per_step != 0:
            args.split_tokens = (args.split_tokens // tokens_per_step) * tokens_per_step
            print0(f"Adjusted split_tokens to {args.split_tokens} (must be divisible by {tokens_per_step})")
        steps = args.split_tokens // tokens_per_step

        # Pre-tokenized data: read the held-out val files directly, no online tokenization.
        loader = pretrain_loader(
            args.device_batch_size, sequence_len, split="val",
            dataset_root=args.dataset_root, device=device,
        )
        loss = evaluate_loss(model, loader, steps)
        loss_results["val"] = loss
        print0(f"Val loss: {loss:.6f}")

    # --- CORE evaluation ---
    if 'core' in eval_modes:
        print0("\n" + "="*80)
        print0("CORE Evaluation")
        print0("="*80)
        core_results = evaluate_core(model, tokenizer, device, max_per_task=args.max_per_task)

        # Write CSV output
        if ddp_rank == 0:
            base_dir = get_base_dir()
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in core_results["results"]:
                    acc = core_results["results"][label]
                    centered = core_results["centered_results"][label]
                    f.write(f"{label:<35}, {acc:<10.6f}, {centered:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_results['core_metric']:<10.6f}\n")
            print0(f"\nResults written to: {output_csv_path}")
            print0(f"CORE metric: {core_results['core_metric']:.4f}")

    # --- Log to report ---
    report_data = [{"model": model_name}]

    if core_results:
        report_data[0]["CORE metric"] = core_results["core_metric"]
        report_data.append(core_results["centered_results"])

    if loss_results:
        report_data[0]["train loss"] = loss_results.get("train")
        report_data[0]["val loss"] = loss_results.get("val")

    if samples:
        report_data.append({f"sample {i}": s for i, s in enumerate(samples)})
    if unconditioned_samples:
        report_data.append({f"unconditioned {i}": s for i, s in enumerate(unconditioned_samples)})

    get_report().log(section="Base model evaluation", data=report_data)

    compute_cleanup()


if __name__ == "__main__":
    main()
