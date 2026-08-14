"""
Evaluate the Chat model.
All the generic code lives here, and all the evaluation-specific
code lives in nanollm directory and is imported from here.

Example runs:
python -m scripts.chat_eval -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.chat_eval -- -a ARC-Easy
"""

import argparse
import time

import torch
import torch.distributed as dist

from nanollm.checkpoint_manager import load_model
from nanollm.report import get_report
from nanollm.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    get_dist_info,
    print0,
)
from nanollm.engine import Engine
from tasks.abmusu import AbMusu
from tasks.global_mmlu import GlobalMMLU
from tasks.nlr_causal_reasoning import NLRCausalReasoningVI
from tasks.uit_viquad import UITViQuADHallucination, UITViQuADQA
from tasks.uit_vsfc import UITVSFCSentiment
from tasks.uit_vsmec import UITVSMEC
from tasks.vianli import ViANLI
from tasks.v_ifeval import VIFEval

# -----------------------------------------------------------------------------
# Generative evaluation loop (we go one problem at a time, sample, evaluate)

def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    # Run the evaluation
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]

        # Tokenize the prompt
        encoded_prompt = tokenizer.render_for_completion(conversation)
        # Get the completions
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # Decode the completions as text
        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        # Evaluate success criteria
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        # Keep stats
        total += 1
        num_passed += int(passed)

        # Logging (overwrite the same line in the console)
        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    # Finish the in-place progress line with a newline before final summary
    print()

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    print0("=" * 50)
    print0(f"Final: {num_passed}/{total} ({100*num_passed/total:.2f}%)")

    # Return the accuracy
    return num_passed/total


def run_generative_classification_eval(
    task_object,
    tokenizer,
    model,
    engine,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    max_problems=None,
):
    """Report accuracy and macro-F1 for generated label predictions."""

    if num_samples != 1:
        raise ValueError("Classification metrics require --num-samples=1")
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    device = model.get_device()
    labels = tuple(task_object.labels)
    label_to_index = {label: index for index, label in enumerate(labels)}
    num_problems = len(task_object) if max_problems is None else min(
        len(task_object), max_problems
    )
    if num_problems <= 0:
        raise ValueError("Classification evaluation requires at least one problem")

    # The last prediction column collects invalid/non-label generations.
    confusion = torch.zeros(
        (len(labels), len(labels) + 1), dtype=torch.long, device=device
    )
    local_total = local_correct = 0
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(
            conversation, enable_thinking=False
        )
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        prediction = tokenizer.decode(results[0][len(encoded_prompt):]).strip()
        gold_index = label_to_index[conversation["answer"]]
        prediction_index = label_to_index.get(prediction, len(labels))
        confusion[gold_index, prediction_index] += 1
        local_total += 1
        local_correct += int(gold_index == prediction_index)
        print(
            f"\r\033[KRank {ddp_rank} | {local_correct}/{local_total} "
            f"({100 * local_correct / local_total:.2f}%)",
            end="",
            flush=True,
        )
    print()

    if ddp:
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
    total = confusion.sum().item()
    correct = sum(confusion[index, index].item() for index in range(len(labels)))
    class_f1 = []
    for index in range(len(labels)):
        true_positive = confusion[index, index].item()
        false_positive = confusion[:, index].sum().item() - true_positive
        false_negative = confusion[index, :].sum().item() - true_positive
        support = true_positive + false_negative
        if support > 0 or false_positive > 0:
            denominator = 2 * true_positive + false_positive + false_negative
            class_f1.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    metrics = {
        "accuracy": correct / total,
        "macro_f1": sum(class_f1) / len(class_f1),
    }
    task_object.metrics = metrics
    print0("=" * 50)
    print0(f"{type(task_object).__name__} accuracy: {100 * metrics['accuracy']:.2f}%")
    print0(f"{type(task_object).__name__} macro-F1: {100 * metrics['macro_f1']:.2f}%")
    return metrics[task_object.primary_metric]


def run_extractive_qa_eval(
    task_object,
    tokenizer,
    model,
    engine,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    max_problems=None,
):
    """Report standard extractive-QA exact match and token F1."""

    if num_samples != 1:
        raise ValueError("Extractive QA metrics require --num-samples=1")
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    device = model.get_device()
    num_problems = len(task_object) if max_problems is None else min(
        len(task_object), max_problems
    )
    if num_problems <= 0:
        raise ValueError("Extractive QA evaluation requires at least one problem")

    totals = [0.0, 0.0, 0.0]  # count, exact matches, token-F1 sum
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(
            conversation, enable_thinking=False
        )
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        completion = tokenizer.decode(results[0][len(encoded_prompt):])
        score = task_object.evaluate_details(conversation, completion)
        totals[0] += 1
        totals[1] += int(score.exact_match)
        totals[2] += score.f1
        print(
            f"\r\033[KRank {ddp_rank} | QA F1 "
            f"{100 * totals[2] / totals[0]:.2f}% ({int(totals[0])} questions)",
            end="",
            flush=True,
        )
    print()

    totals_tensor = torch.tensor(totals, dtype=torch.float64, device=device)
    if ddp:
        dist.all_reduce(totals_tensor, op=dist.ReduceOp.SUM)
    total, exact_matches, f1_sum = totals_tensor.tolist()
    metrics = {"exact_match": exact_matches / total, "f1": f1_sum / total}
    task_object.metrics = metrics
    print0("=" * 50)
    print0(f"UIT-ViQuAD QA exact match: {100 * metrics['exact_match']:.2f}%")
    print0(f"UIT-ViQuAD QA token F1:    {100 * metrics['f1']:.2f}%")
    return metrics["f1"]


def run_hallucination_eval(
    task_object,
    tokenizer,
    model,
    engine,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    max_problems=None,
):
    """Evaluate grounded QA and exact refusal behavior on UIT-ViQuAD2.0."""

    if num_samples != 1:
        raise ValueError(
            "UIT-ViQuAD hallucination evaluation requires --num-samples=1"
        )
    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    device = model.get_device()
    num_problems = len(task_object) if max_problems is None else min(
        len(task_object), max_problems
    )
    if num_problems <= 0:
        raise ValueError("UIT-ViQuAD hallucination evaluation needs at least one problem")

    # overall total/correct, answerable total/QA-correct/non-refusal,
    # impossible total/refused.
    counts = [0, 0, 0, 0, 0, 0, 0]
    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(
            conversation, enable_thinking=False
        )
        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        completion = tokenizer.decode(results[0][len(encoded_prompt):])
        outcome = task_object.evaluate_details(conversation, completion)
        counts[0] += 1
        counts[1] += int(outcome.correct)
        if outcome.is_impossible:
            counts[5] += 1
            counts[6] += int(outcome.refused)
        else:
            counts[2] += 1
            counts[3] += int(outcome.correct)
            counts[4] += int(not outcome.refused)
        print(
            f"\r\033[KRank {ddp_rank} | grounded accuracy "
            f"{100 * counts[1] / counts[0]:.2f}% ({counts[0]} questions)",
            end="",
            flush=True,
        )
    print()

    if ddp:
        count_tensor = torch.tensor(counts, dtype=torch.long, device=device)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        counts = count_tensor.tolist()
    total, correct, answerable, answerable_correct, answerable_nonrefusal, impossible, refused = counts
    refusal_accuracy = refused / impossible if impossible else 0.0

    def binary_f1(true_positive, false_positive, false_negative):
        denominator = 2 * true_positive + false_positive + false_negative
        return 0.0 if denominator == 0 else 2 * true_positive / denominator

    answerable_f1 = binary_f1(
        answerable_nonrefusal,
        impossible - refused,
        answerable - answerable_nonrefusal,
    )
    impossible_f1 = binary_f1(
        refused,
        answerable - answerable_nonrefusal,
        impossible - refused,
    )
    metrics = {
        "overall_accuracy": correct / total,
        "answerable_accuracy": answerable_correct / answerable if answerable else 0.0,
        "answerability_accuracy": (answerable_nonrefusal + refused) / total,
        "answerability_macro_f1": (answerable_f1 + impossible_f1) / 2,
        "refusal_accuracy": refusal_accuracy,
        "hallucination_rate": 1.0 - refusal_accuracy if impossible else 0.0,
    }
    task_object.metrics = metrics
    print0("=" * 50)
    print0(f"UIT-ViQuAD overall accuracy:    {100 * metrics['overall_accuracy']:.2f}%")
    print0(f"UIT-ViQuAD answerable accuracy: {100 * metrics['answerable_accuracy']:.2f}%")
    print0(f"UIT-ViQuAD answerability acc:   {100 * metrics['answerability_accuracy']:.2f}%")
    print0(f"UIT-ViQuAD answerability F1:    {100 * metrics['answerability_macro_f1']:.2f}%")
    print0(f"UIT-ViQuAD refusal accuracy:    {100 * metrics['refusal_accuracy']:.2f}%")
    print0(f"UIT-ViQuAD hallucination rate:  {100 * metrics['hallucination_rate']:.2f}%")
    return metrics["overall_accuracy"]


def run_instruction_following_eval(
    task_object,
    tokenizer,
    model,
    engine,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    batch_size=32,
    use_cuda_graphs=True,
    max_problems=None,
):
    """Run V-IFEval and aggregate its official strict/loose metrics."""

    if num_samples != 1:
        raise ValueError("V-IFEval requires --num-samples=1 for comparable accuracy")

    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    device = model.get_device()
    num_problems = len(task_object) if max_problems is None else min(
        len(task_object), max_problems
    )
    if num_problems <= 0:
        raise ValueError("V-IFEval requires at least one problem")

    local_conversations = [
        task_object[i] for i in range(ddp_rank, num_problems, ddp_world_size)
    ]
    encoded_prompts = [
        tokenizer.render_for_completion(conversation, enable_thinking=False)
        for conversation in local_conversations
    ]
    generated = engine.generate_prompts(
        encoded_prompts,
        batch_size=batch_size,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        use_cuda_graphs=use_cuda_graphs,
    )

    # prompt total/correct, instruction total/correct, first strict then loose.
    counts = [0, 0, 0, 0, 0, 0]
    for conversation, encoded_prompt, result in zip(
        local_conversations, encoded_prompts, generated
    ):
        completion = tokenizer.decode(result[len(encoded_prompt):])
        outcome = task_object.evaluate_details(conversation, completion)

        counts[0] += 1
        counts[1] += int(outcome.strict_prompt)
        counts[2] += len(outcome.strict)
        counts[3] += sum(outcome.strict)
        counts[4] += int(outcome.loose_prompt)
        counts[5] += sum(outcome.loose)
        print(
            f"\r\033[KRank {ddp_rank} | strict prompts "
            f"{counts[1]}/{counts[0]} ({100 * counts[1] / counts[0]:.2f}%)",
            end="",
            flush=True,
        )
    print()

    if ddp:
        count_tensor = torch.tensor(counts, dtype=torch.long, device=device)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        counts = count_tensor.tolist()

    prompt_total, strict_prompts, instruction_total, strict_instructions, loose_prompts, loose_instructions = counts
    metrics = {
        "strict_prompt_accuracy": strict_prompts / prompt_total,
        "strict_instruction_accuracy": strict_instructions / instruction_total,
        "loose_prompt_accuracy": loose_prompts / prompt_total,
        "loose_instruction_accuracy": loose_instructions / instruction_total,
    }
    task_object.metrics = metrics
    print0("=" * 50)
    print0(f"V-IFEval strict prompt:      {100 * metrics['strict_prompt_accuracy']:.2f}%")
    print0(f"V-IFEval strict instruction: {100 * metrics['strict_instruction_accuracy']:.2f}%")
    print0(f"V-IFEval loose prompt:       {100 * metrics['loose_prompt_accuracy']:.2f}%")
    print0(f"V-IFEval loose instruction:  {100 * metrics['loose_instruction_accuracy']:.2f}%")
    return metrics["strict_prompt_accuracy"]


def run_summarization_eval(
    task_object,
    tokenizer,
    model,
    engine,
    num_samples,
    max_new_tokens,
    temperature,
    top_k,
    batch_size=32,
    use_cuda_graphs=True,
    max_problems=None,
):
    """Generate one summary per cluster and macro-average ROUGE-2."""

    if num_samples != 1:
        raise ValueError("AbMusu requires --num-samples=1 for comparable ROUGE")
    if max_new_tokens <= 0:
        raise ValueError("AbMusu max new tokens must be positive")

    ddp, ddp_rank, _, ddp_world_size = get_dist_info()
    device = model.get_device()
    num_problems = len(task_object) if max_problems is None else min(
        len(task_object), max_problems
    )
    if num_problems <= 0:
        raise ValueError("AbMusu requires at least one problem")

    context_length = getattr(model.config, "context_length", 4096)
    # Leave room for the assistant header inserted after conversation rendering.
    max_prompt_tokens = context_length - max_new_tokens - 32
    if max_prompt_tokens <= 0:
        raise ValueError(
            f"AbMusu needs max_new_tokens smaller than context length {context_length}"
        )

    local_conversations = [
        task_object[i] for i in range(ddp_rank, num_problems, ddp_world_size)
    ]
    encoded_prompts = [
        tokenizer.render_for_completion(
            conversation,
            enable_thinking=False,
            max_tokens=max_prompt_tokens,
        )
        for conversation in local_conversations
    ]
    generated = engine.generate_prompts(
        encoded_prompts,
        batch_size=batch_size,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        use_cuda_graphs=use_cuda_graphs,
    )

    total = 0
    precision_sum = recall_sum = f1_sum = 0.0
    for conversation, encoded_prompt, result in zip(
        local_conversations, encoded_prompts, generated
    ):
        completion = tokenizer.decode(result[len(encoded_prompt):])
        score = task_object.evaluate_details(conversation, completion)
        total += 1
        precision_sum += score.precision
        recall_sum += score.recall
        f1_sum += score.f1
        print(
            f"\r\033[KRank {ddp_rank} | ROUGE-2 F1 "
            f"{100 * f1_sum / total:.2f}% ({total} clusters)",
            end="",
            flush=True,
        )
    print()

    totals = [total, precision_sum, recall_sum, f1_sum]
    if ddp:
        totals_tensor = torch.tensor(totals, dtype=torch.float64, device=device)
        dist.all_reduce(totals_tensor, op=dist.ReduceOp.SUM)
        totals = totals_tensor.tolist()
    total, precision_sum, recall_sum, f1_sum = totals
    metrics = {
        "rouge2_precision": precision_sum / total,
        "rouge2_recall": recall_sum / total,
        "rouge2_f1": f1_sum / total,
    }
    task_object.metrics = metrics
    print0("=" * 50)
    print0(f"AbMusu ROUGE-2 precision: {100 * metrics['rouge2_precision']:.2f}%")
    print0(f"AbMusu ROUGE-2 recall:    {100 * metrics['rouge2_recall']:.2f}%")
    print0(f"AbMusu ROUGE-2 F1:        {100 * metrics['rouge2_f1']:.2f}%")
    return metrics["rouge2_f1"]

# -----------------------------------------------------------------------------
# Categorical evaluation loop
# A lot easier because we don't have to sample. Therefore, we can actually go
# batches at a time and just check the logits for correct answer choices.

@torch.inference_mode()
def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None):

    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    pad_token_id = tokenizer.token_to_id("<|endoftext|>")

    # We'll process batches of independent problems at a time because there is no sampling needed
    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    # Run the evaluation
    letter_to_id_cache = {} # many letters will repeat often, let's save the tokenizer some work
    num_passed, total = 0, 0
    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        # Prepare the batch of problems. They might all be of different length, so we pad/collate them.
        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids] # where the last token is (and the predicted answer)
        padded_prompt_ids = [ids + [pad_token_id] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        # Run the complete prompts, but project only their answer positions to
        # the vocabulary. This returns (B, 1, V), not the wasteful (B, T, V).
        logit_positions = torch.tensor(
            answer_time_positions, dtype=torch.long, device=device
        )
        logits = model(prompt_ids, logit_positions=logit_positions)

        # Focus on the available answer on just the letters corresponding to choices
        # Note that this helps the evaluation a lot because it specifically narrows the focus to only the available letters
        # The much harder alternative would be to just generate from the Assistant and check if it responded with the correct
        # letter (e.g. A, B, C, D), but evaluations typically make the task easier in this way.
        for idx, conversation in enumerate(conversations):
            # get the token ids of all the available letters of this problem
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if not letter in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])
            # focus logits just down to the answer position and the available letters of the answer
            focus_logits = logits[idx, 0, letter_ids]
            # get the argmax letter (the predicted answer)
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            # evaluate the outcome
            outcome = task_object.evaluate(conversation, predicted_letter)
            num_passed += int(outcome)
            total += 1

    # Aggregate results across all ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    average = num_passed/total
    print0(f"Final: {num_passed}/{total} ({100*average:.2f}%)")
    return average

# -----------------------------------------------------------------------------

def run_chat_eval(task_name, model, tokenizer, engine,
                   batch_size=1, num_samples=1, temperature=0.0, top_k=50,
                   max_problems=None,
                   shuffle=False, seed=42, use_cuda_graphs=True):
    device = model.get_device()
    task_factories = {
        'GlobalMMLU': lambda: GlobalMMLU('./.cache/nanollm/eval_bundle/eval_data/global_mmlu.jsonl', shuffle=shuffle, seed=seed),
        'NLR-Causal-Reasoning-vi': lambda: NLRCausalReasoningVI(shuffle=shuffle, seed=seed),
        'UIT-ViQuAD-Hallucination': lambda: UITViQuADHallucination(split='validation', shuffle=shuffle, seed=seed),
        'UIT-ViQuAD-QA': lambda: UITViQuADQA(split='validation', shuffle=shuffle, seed=seed),
        'UIT-VSFC-Sentiment': lambda: UITVSFCSentiment(shuffle=shuffle, seed=seed),
        'UIT-VSMEC': lambda: UITVSMEC(shuffle=shuffle, seed=seed),
        'ViANLI': lambda: ViANLI(shuffle=shuffle, seed=seed),
        'V-IFEval': lambda: VIFEval(shuffle=shuffle, seed=seed),
        'AbMusu': lambda: AbMusu(shuffle=shuffle, seed=seed),
    }
    if task_name not in task_factories:
        raise ValueError(f"Unknown task: {task_name!r}. Available: {list(task_factories)}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started_at = time.perf_counter()
    task_object = task_factories[task_name]()
    max_new_tokens = task_object.max_new_tokens
    if task_object.eval_type == 'generative':
        acc = run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens, temperature, top_k, max_problems=max_problems)
    elif task_object.eval_type == 'categorical':
        acc = run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=max_problems)
    elif task_object.eval_type == 'generative_classification':
        acc = run_generative_classification_eval(
            task_object, tokenizer, model, engine, num_samples, max_new_tokens,
            temperature, top_k, max_problems=max_problems,
        )
    elif task_object.eval_type == 'extractive_qa':
        acc = run_extractive_qa_eval(
            task_object, tokenizer, model, engine, num_samples, max_new_tokens,
            temperature, top_k, max_problems=max_problems,
        )
    elif task_object.eval_type == 'hallucination':
        acc = run_hallucination_eval(
            task_object, tokenizer, model, engine, num_samples, max_new_tokens,
            temperature, top_k, max_problems=max_problems,
        )
    elif task_object.eval_type == 'instruction_following':
        acc = run_instruction_following_eval(
            task_object, tokenizer, model, engine, num_samples, max_new_tokens,
            temperature, top_k, batch_size=batch_size,
            use_cuda_graphs=use_cuda_graphs,
            max_problems=max_problems,
        )
    elif task_object.eval_type == 'summarization':
        acc = run_summarization_eval(
            task_object, tokenizer, model, engine, num_samples,
            max_new_tokens, temperature, top_k, batch_size=batch_size,
            use_cuda_graphs=use_cuda_graphs,
            max_problems=max_problems,
        )
    else:
        raise ValueError(f"Unsupported task evaluation type: {task_object.eval_type}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started_at
    evaluated = len(task_object) if max_problems is None else min(
        len(task_object), max_problems
    )
    rate = evaluated / elapsed if elapsed > 0 else float("inf")
    timing = f"{elapsed:.1f}s, {rate:.2f} examples/s"
    if device.type == "cuda":
        allocated_gib = torch.cuda.memory_allocated(device) / (1024**3)
        process_peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        timing += (
            f", allocated {allocated_gib:.2f} GiB, "
            f"process peak {process_peak_gib:.2f} GiB"
        )
    print0(f"Timing {task_name}: {timing}")
    return acc

# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--source', type=str, required=True, help="Source of the model: sft|rl")
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Task name. Default = all tasks. Use | to split multiple tasks.")
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Evaluation batch size')
    parser.add_argument('--no-cuda-graphs', action='store_true', help='Disable CUDA graphs for batched generation')
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='Max problems to evaluate')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # Get the tasks to evaluate on
    all_tasks = [
        'GlobalMMLU',
        'NLR-Causal-Reasoning-vi',
        'UIT-ViQuAD-Hallucination',
        'UIT-ViQuAD-QA',
        'UIT-VSMEC',
        'UIT-VSFC-Sentiment',
        'ViANLI',
        'V-IFEval',
        'AbMusu',
    ]
    baseline_accuracies = {
        'GlobalMMLU': 0.25, # multiple choice 1 of 4 => 25%
        'NLR-Causal-Reasoning-vi': 0.5, # random choice between A and B
        'UIT-ViQuAD-Hallucination': 0.0,
        'UIT-ViQuAD-QA': 0.0,
        'UIT-VSMEC': 1 / 7, # random choice among seven emotion labels
        'UIT-VSFC-Sentiment': 1 / 3, # random choice among three sentiments
        'ViANLI': 1 / 3, # random choice among three NLI relations
        'V-IFEval': 0.0, # no meaningful random instruction-following baseline
        'AbMusu': 0.0, # no meaningful random summarization baseline
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')

    # Run all the task evaluations sequentially
    results = {}
    for task_name in task_names:
        acc = run_chat_eval(
            task_name,
            model, tokenizer, engine,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_k=args.top_k,
            max_problems=args.max_problems,
            use_cuda_graphs=not args.no_cuda_graphs,
        )
        results[task_name] = acc
        metric_name = {
            "AbMusu": "ROUGE-2 F1",
            "UIT-VSMEC": "macro-F1",
            "UIT-VSFC-Sentiment": "macro-F1",
            "UIT-ViQuAD-QA": "token F1",
        }.get(task_name, "accuracy")
        print0(f"{task_name} {metric_name}: {100 * acc:.2f}%")

    # Log to report
    report = get_report()
    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    # calculate the ChatCORE metric if we can (similar to CORE, it's the mean centered accuracy)
    # this way, ChatCORE ranges from 0 (at random baseline) to 1 (peak performance)
    chatcore_metric_dict = {}
    if all_tasks_were_evaluated:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        chatcore_metric_dict = {"ChatCORE metric": chatcore_metric}
    get_report().log(section="Chat evaluation " + args.source, data=[
        vars(args), # CLI args
        results,
        chatcore_metric_dict,
    ])

    compute_cleanup()
