"""Lightweight, selector-based CUDA module timing utilities."""

from collections import defaultdict
from fnmatch import fnmatchcase

import torch


class CudaModuleProfiler:
    """Aggregate CUDA forward time for selected ``nn.Module`` instances.

    Selectors use ``label=qualified.name.glob`` or ``label=type:ClassNameGlob``.
    Multiple selectors may share a label, but selected modules may not overlap.
    """

    def __init__(self, model, selectors):
        self.events = defaultdict(list)
        self.pending = defaultdict(list)
        self.handles = []
        matches = self._resolve(model, selectors)
        self._validate_non_overlapping(matches)
        for name, module, label in matches:
            self.events[label]  # Keep selected-but-not-executed categories visible as 0 ms.
            self.handles.append(module.register_forward_pre_hook(
                self._make_pre_hook(label)
            ))
            self.handles.append(module.register_forward_hook(self._post_hook))

    @staticmethod
    def _parse(selector):
        if "=" not in selector:
            raise ValueError(
                f"Invalid profile selector {selector!r}; expected LABEL=GLOB"
            )
        label, pattern = (part.strip() for part in selector.split("=", 1))
        if not label or not pattern:
            raise ValueError(
                f"Invalid profile selector {selector!r}; label and pattern are required"
            )
        return label, pattern

    @classmethod
    def _resolve(cls, model, selectors):
        named_modules = list(model.named_modules())
        matches = []
        selected_ids = {}
        for selector in selectors:
            label, pattern = cls._parse(selector)
            by_type = pattern.startswith("type:")
            pattern = pattern.removeprefix("type:") if by_type else pattern
            selector_matches = []
            for name, module in named_modules:
                candidate = type(module).__name__ if by_type else name
                if fnmatchcase(candidate, pattern):
                    previous = selected_ids.get(id(module))
                    if previous is not None:
                        raise ValueError(
                            f"Profile selectors {previous!r} and {selector!r} both "
                            f"match module {name!r}"
                        )
                    selected_ids[id(module)] = selector
                    selector_matches.append((name, module, label))
            if not selector_matches:
                target = "module type" if by_type else "module name"
                raise ValueError(
                    f"Profile selector {selector!r} matched no {target}s"
                )
            matches.extend(selector_matches)
        return matches

    @staticmethod
    def _validate_non_overlapping(matches):
        names = sorted(name for name, _, _ in matches)
        for parent, child in zip(names, names[1:]):
            if parent == "" or child.startswith(parent + "."):
                raise ValueError(
                    f"Profiled modules {parent or '<root>'!r} and {child!r} overlap; "
                    "select only the parent or its children to avoid double-counting"
                )

    def _make_pre_hook(self, label):
        def pre_hook(module, inputs):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self.pending[id(module)].append((label, start))

        return pre_hook

    def _post_hook(self, module, inputs, output):
        label, start = self.pending[id(module)].pop()
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self.events[label].append((start, end))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def elapsed_ms(self):
        return {
            label: sum(start.elapsed_time(end) for start, end in pairs)
            for label, pairs in self.events.items()
        }
