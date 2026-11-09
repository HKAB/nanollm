import pytest
import torch

from scripts.chat_rl_grpo import policy_terms
from tasks.sft.gsm8k import GSM8K, think_reward


@pytest.mark.parametrize('text, expected', [
    ('<think>reasoning</think>#### 42', 1),
    ('<think></think>#### 42', 0),
    ('<think>   </think>#### 42', 0),
    ('<think>reasoning</think>', 0),
    ('answer', 0),
    ('</think>answer<think>', 0),
    ('<think>a</think><think>b</think>42', 0),
])
def test_think_reward(text, expected):
    assert think_reward(text) == expected


def sampled_logprobs(logits, targets):
    return logits.detach().log_softmax(-1).gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)


def test_task_reward_accounts_for_prefill():
    task = GSM8K.__new__(GSM8K)
    task.think_bonus = 0.1
    conversation = {"answer": "42", "completion_prompt": "assistant\n<think>\n"}
    assert task.reward(conversation, "reasoning</think>\n#### 42") == 1.1
    assert task.reward(conversation, "</think>\n#### 42") == 1.0


@pytest.mark.parametrize("ratio_level", ["sequence", "sequence-sum", "token"])
def test_clipped_objective_uses_frozen_logprobs(ratio_level):
    logits = torch.zeros(4, 3, 2, requires_grad=True)
    targets = torch.tensor([[0, 1, -1]] * 4)
    advantages = torch.tensor([1., -1., 1., -1.])
    token_ratios = torch.tensor([[2., 2.], [0.5, 0.5], [2., 0.5], [2., 0.5]])
    old_logprobs = sampled_logprobs(logits, targets)
    old_logprobs[:, :2] -= token_ratios.log()
    ratio = token_ratios
    if ratio_level != "token":
        ratio = ratio.prod(-1)
        if ratio_level == "sequence":
            ratio = ratio.sqrt()
    else:
        advantages = advantages[:, None]
    expected = -torch.minimum(ratio * advantages, ratio.clamp(0.8, 1.2) * advantages)
    if ratio_level == "token":
        expected = expected.mean(-1)
    losses, _ = policy_terms(logits, targets, advantages.flatten(), old_logprobs, ratio_level=ratio_level)
    torch.testing.assert_close(losses, expected)
    losses.sum().backward()
    assert logits.grad[:2].count_nonzero() == 0
    assert logits.grad[:, 2].count_nonzero() == 0


def test_entropy_masking_and_policy_gradient():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 5, requires_grad=True)
    targets = torch.tensor([[-1, 1, 2, -1], [-1, 3, -1, -1]])
    advantages = torch.tensor([1., -1.])
    losses, entropy = policy_terms(logits, targets, advantages, sampled_logprobs(logits, targets))
    distribution = torch.distributions.Categorical(logits=logits)
    expected_entropy = distribution.entropy()
    torch.testing.assert_close(entropy, torch.stack([
        expected_entropy[0, 1:3].mean(), expected_entropy[1, 1],
    ]))
    losses.mean().backward()
    actual_grad = logits.grad.clone()
    logits.grad = None
    mask = targets >= 0
    logp = distribution.log_prob(targets.clamp_min(0))
    reference_loss = -((logp * mask).sum(-1) / mask.sum(-1) * advantages).mean()
    reference_loss.backward()
    torch.testing.assert_close(actual_grad, logits.grad)
    assert actual_grad[~mask].count_nonzero() == 0
    assert actual_grad.abs().sum() > 0


def test_uneven_microbatches_match_full_batch():
    torch.manual_seed(1)
    logits = torch.randn(3, 4, 5, requires_grad=True)
    targets = torch.tensor([[1, 2, -1, -1], [2, 3, 4, 0], [4, -1, -1, -1]])
    advantages = torch.tensor([-1., 0., 1.])
    policy_terms(logits, targets, advantages, sampled_logprobs(logits, targets))[0].mean().backward()
    full_gradient = logits.grad.clone()
    logits.grad = None
    for start in (0, 2):
        policy_terms(logits[start:start + 2], targets[start:start + 2], advantages[start:start + 2], sampled_logprobs(logits, targets)[start:start + 2])[0].sum().div(3).backward()
    torch.testing.assert_close(logits.grad, full_gradient)


def test_qwen3_pretrained_smoke():
    """Opt in with GRPO_TEST_HF=1; downloads ~1.2 GB on first use."""
    import os
    if os.environ.get('GRPO_TEST_HF') != '1':
        pytest.skip('set GRPO_TEST_HF=1 to test pretrained Qwen3')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    name = 'Qwen/Qwen3-0.6B-Base'
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).eval()
    prompt = tokenizer('Question: What is 1 + 1?\nAnswer: <think>\n', return_tensors='pt').input_ids
    with torch.no_grad():
        sequences = model.generate(
            prompt, attention_mask=torch.ones_like(prompt), do_sample=True, temperature=1., top_k=0, top_p=1.,
            num_return_sequences=2, max_new_tokens=16,
            pad_token_id=tokenizer.eos_token_id,
        )
    targets = sequences[:, 1:].clone()
    targets[:, :prompt.shape[1] - 1] = -1
    # Synthetic mixed rewards guarantee a nonzero gradient even if both
    # short untrained rollouts fail the sparse math/format rewards.
    rewards = torch.tensor([1., 0.])
    advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-4)
    logits = model(sequences[:, :-1], use_cache=False).logits
    losses, entropy = policy_terms(logits, targets, advantages, sampled_logprobs(logits, targets))
    losses.mean().backward()
    assert torch.isfinite(entropy).all() and (entropy > 0).all()
    assert model.lm_head.weight.grad.norm() > 0
    before = model.lm_head.weight.detach().clone()
    optimizer = torch.optim.SGD([model.lm_head.weight], lr=1e-4)
    optimizer.step()
    assert not torch.equal(before, model.lm_head.weight)
    print(f'Qwen3 CPU: entropy={entropy.tolist()}, head_grad_norm={model.lm_head.weight.grad.norm().item():.6f}')
