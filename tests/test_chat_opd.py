from types import SimpleNamespace

import torch

from scripts.chat_opd import collate_rollouts, completion_logprobs, importance_sampling


def test_importance_sampling_differentiates_only_current_policy():
    current = torch.tensor([-2.0, -2.0], requires_grad=True)
    sampled = torch.tensor([-2.0, -3.0], requires_grad=True)
    teacher = torch.tensor([-1.0, -4.0], requires_grad=True)
    losses, ratios = importance_sampling(current, sampled, teacher - sampled)

    torch.testing.assert_close(ratios, torch.tensor([0.0, 1.0]).exp())
    torch.testing.assert_close(losses, torch.tensor([-1.0, 1.0]) * ratios.detach())
    losses.sum().backward()
    torch.testing.assert_close(current.grad, losses.detach())
    assert sampled.grad is None
    assert teacher.grad is None


def test_opd_scores_generated_tokens_at_shifted_positions():
    outputs = [
        SimpleNamespace(prompt_token_ids=[1, 2], output_token_ids=[3, 4],
                        token_logprobs=torch.tensor([-0.2, -0.3])),
        SimpleNamespace(prompt_token_ids=[1], output_token_ids=[4],
                        token_logprobs=torch.tensor([-0.4])),
    ]
    inputs, targets, sampled = collate_rollouts(outputs, pad_id=0, device="cpu")
    assert inputs.tolist() == [[1, 2, 3], [1, 0, 0]]
    assert targets.tolist() == [[-1, 3, 4], [4, -1, -1]]
    mask = targets >= 0
    torch.testing.assert_close(sampled[mask], torch.tensor([-0.2, -0.3, -0.4]))

    logits = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5)
    full_logprobs = logits.log_softmax(-1)
    expected = full_logprobs[[0, 0, 1], [1, 2, 0], [3, 4, 4]]
    torch.testing.assert_close(completion_logprobs(logits, targets, mask), expected)
