import pytest
import torch

import nanollm.profiler as profiler_module
from nanollm.profiler import CudaModuleProfiler


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU()),
            torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU()),
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class FakeEvent:
    clock = 0

    def __init__(self, **kwargs):
        self.timestamp = None

    def record(self):
        self.timestamp = FakeEvent.clock
        FakeEvent.clock += 1

    def elapsed_time(self, other):
        return float(other.timestamp - self.timestamp)


def test_cuda_module_profiler_matches_names_and_types(monkeypatch):
    monkeypatch.setattr(profiler_module.torch.cuda, "Event", FakeEvent)
    model = TinyModel()
    profiler = CudaModuleProfiler(
        model,
        ["linear=blocks.*.0", "activation=type:ReLU"],
    )

    model(torch.ones(1, 2))
    profiler.close()

    assert profiler.elapsed_ms() == {"linear": 2.0, "activation": 2.0}
    assert not profiler.handles


@pytest.mark.parametrize("selector", ["missing", "=blocks.*", "label="])
def test_cuda_module_profiler_rejects_invalid_selectors(selector):
    with pytest.raises(ValueError):
        CudaModuleProfiler(TinyModel(), [selector])


def test_cuda_module_profiler_rejects_unmatched_selector():
    with pytest.raises(ValueError, match="matched no module names"):
        CudaModuleProfiler(TinyModel(), ["missing=transformer.*"])


def test_cuda_module_profiler_rejects_overlapping_modules():
    with pytest.raises(ValueError, match="overlap"):
        CudaModuleProfiler(
            TinyModel(),
            ["block=blocks.0", "linear=blocks.0.0"],
        )
