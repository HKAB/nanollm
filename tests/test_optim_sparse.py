import torch

import nanollm.optim as optim_module
from nanollm.optim import MuonAdamW


def test_muon_skips_parameters_with_none_grad(monkeypatch):
    def fake_muon_step(
        grads, params, momentum, second_momentum,
        momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim,
    ):
        # Deliberately slot-local update: distinct gradients reveal any
        # misalignment between the gradient and parameter stacks.
        params.add_(grads)
        momentum.add_(grads)
        second_momentum.add_(grads.square().mean(dim=red_dim, keepdim=True))

    monkeypatch.setattr(optim_module, "muon_step_fused", fake_muon_step)
    active = torch.nn.Parameter(torch.zeros(2, 2))
    inactive = torch.nn.Parameter(torch.full((2, 2), 3.0))
    second_active = torch.nn.Parameter(torch.full((2, 2), 5.0))
    active.grad = torch.ones_like(active)
    inactive.grad = None
    second_active.grad = torch.full_like(second_active, 2.0)
    optimizer = MuonAdamW([{
        "kind": "muon",
        "params": [active, inactive, second_active],
        "lr": 1e-3,
        "momentum": 0.9,
        "ns_steps": 1,
        "beta2": 0.9,
        "weight_decay": 0.1,
    }])

    optimizer.step()

    assert torch.equal(active, torch.ones_like(active))
    assert torch.equal(inactive, torch.full_like(inactive, 3.0))
    assert torch.equal(second_active, torch.full_like(second_active, 7.0))
    state = optimizer.state[active]
    assert torch.equal(state["momentum_buffer"][0], torch.ones(2, 2))
    assert torch.count_nonzero(state["momentum_buffer"][1]) == 0
    assert torch.equal(state["momentum_buffer"][2], torch.full((2, 2), 2.0))
    assert torch.equal(state["second_momentum_buffer"][0], torch.ones(2, 1))
    assert torch.count_nonzero(state["second_momentum_buffer"][1]) == 0
    assert torch.equal(state["second_momentum_buffer"][2], torch.full((2, 1), 4.0))
