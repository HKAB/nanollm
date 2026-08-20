import pytest
import torch

import nanollm.optim as optim_module
from nanollm.optim import DistMuonAdamW, MuonAdamW


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


def test_eager_optimizer_state_offload_layout_is_stable():
    adam_param = torch.nn.Parameter(torch.zeros(2, 2))
    muon_params = [
        torch.nn.Parameter(torch.zeros(2, 2)),
        torch.nn.Parameter(torch.ones(2, 2)),
    ]
    optimizer = MuonAdamW([
        {
            "kind": "adamw",
            "params": [adam_param],
            "lr": 1e-3,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
        {
            "kind": "muon",
            "params": muon_params,
            "lr": 1e-3,
            "momentum": 0.9,
            "ns_steps": 1,
            "beta2": 0.9,
            "weight_decay": 0.0,
        },
    ], state_offload=True)

    first_bytes = optimizer.initialize_state()
    first_tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    first_ptrs = [tensor.data_ptr() for tensor in first_tensors]
    second_bytes = optimizer.initialize_state()
    second_tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]

    assert first_bytes == second_bytes == 80
    assert all(tensor.device.type == "cpu" for tensor in second_tensors)
    assert [tensor.data_ptr() for tensor in second_tensors] == first_ptrs
    assert optimizer.state[adam_param]["step"] == 0
    assert optimizer.state[muon_params[0]]["momentum_buffer"].shape == (2, 2, 2)
    assert optimizer.state[muon_params[0]]["second_momentum_buffer"].shape == (2, 2, 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA pinned memory")
def test_cuda_optimizer_state_offload_matches_resident(monkeypatch):
    def fake_adam_step(p, grad, exp_avg, exp_avg_sq, *args):
        exp_avg.add_(grad)
        exp_avg_sq.add_(grad.square())
        p.add_(exp_avg, alpha=-0.1)

    def fake_muon_step(grads, params, momentum, second_momentum, *args):
        momentum.add_(grads)
        second_momentum.add_(grads.square().mean(dim=-1, keepdim=True))
        params.add_(momentum)

    monkeypatch.setattr(optim_module, "adamw_step_fused", fake_adam_step)
    monkeypatch.setattr(optim_module, "muon_step_fused", fake_muon_step)

    def make_optimizer(offload):
        adam = torch.nn.Parameter(torch.zeros(2, 2, device="cuda"))
        muon = torch.nn.Parameter(torch.ones(2, 2, device="cuda"))
        optimizer = MuonAdamW([
            {
                "kind": "adamw", "params": [adam], "lr": 1e-3,
                "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0,
            },
            {
                "kind": "muon", "params": [muon], "lr": 1e-3,
                "momentum": 0.9, "ns_steps": 1, "beta2": 0.9,
                "weight_decay": 0.0,
            },
        ], state_offload=offload)
        optimizer.initialize_state()
        adam.grad = torch.ones_like(adam)
        muon.grad = torch.full_like(muon, 2.0)
        optimizer.step()
        return adam, muon, optimizer

    adam_resident, muon_resident, _ = make_optimizer(False)
    adam_offloaded, muon_offloaded, offloaded_optimizer = make_optimizer(True)

    assert torch.equal(adam_offloaded, adam_resident)
    assert torch.equal(muon_offloaded, muon_resident)
    state_tensors = [
        value
        for state in offloaded_optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ]
    assert all(tensor.device.type == "cpu" and tensor.is_pinned() for tensor in state_tensors)


def test_distributed_eager_offload_initializes_rank_local_shards(monkeypatch):
    monkeypatch.setattr(optim_module.dist, "get_world_size", lambda: 2)
    adam_param = torch.nn.Parameter(torch.zeros(1024, 2))
    muon_params = [
        torch.nn.Parameter(torch.zeros(2, 2)) for _ in range(3)
    ]
    optimizer = DistMuonAdamW([
        {
            "kind": "adamw", "params": [adam_param], "lr": 1e-3,
            "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0,
        },
        {
            "kind": "muon", "params": muon_params, "lr": 1e-3,
            "momentum": 0.9, "ns_steps": 1, "beta2": 0.9,
            "weight_decay": 0.0,
        },
    ], state_offload=True)

    state_bytes = optimizer.initialize_state()

    adam_state = optimizer.state[adam_param]
    muon_state = optimizer.state[muon_params[0]]
    assert adam_state["exp_avg"].shape == (512, 2)
    assert adam_state["exp_avg_sq"].shape == (512, 2)
    assert muon_state["momentum_buffer"].shape == (2, 2, 2)
    assert muon_state["second_momentum_buffer"].shape == (2, 2, 1)
    assert state_bytes == 8240
    assert all(
        value.device.type == "cpu"
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
