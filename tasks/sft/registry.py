"""Registry for the tasks that make up ChatCORE."""

from dataclasses import dataclass
from typing import Callable

from tasks.sft.abmusu import AbMusu
from tasks.sft.global_mmlu import GlobalMMLU
from tasks.sft.nlr_causal_reasoning import NLRCausalReasoningVI
from tasks.sft.uit_viquad import UITViQuADHallucination, UITViQuADQA
from tasks.sft.uit_vsfc import UITVSFCSentiment
from tasks.sft.uit_vsmec import UITVSMEC
from tasks.sft.vianli import ViANLI
from tasks.sft.v_ifeval import VIFEval


@dataclass(frozen=True)
class ChatTask:
    name: str
    factory: Callable
    random_baseline: float
    periodic_limit: int


CHAT_TASKS = (
    ChatTask("GlobalMMLU", GlobalMMLU, 0.25, 500),
    ChatTask("NLR-Causal-Reasoning-vi", NLRCausalReasoningVI, 0.5, 500),
    ChatTask("ViANLI", ViANLI, 1 / 3, 300),
    ChatTask("UIT-VSMEC", UITVSMEC, 1 / 7, 350),
    ChatTask("UIT-VSFC-Sentiment", UITVSFCSentiment, 1 / 3, 300),
    ChatTask(
        "UIT-ViQuAD-QA",
        lambda **kwargs: UITViQuADQA(split="validation", **kwargs),
        0.0,
        250,
    ),
    ChatTask(
        "UIT-ViQuAD-Hallucination",
        lambda **kwargs: UITViQuADHallucination(split="validation", **kwargs),
        0.0,
        250,
    ),
    ChatTask("V-IFEval", VIFEval, 0.0, 500),
    # Validation has 100 labeled clusters; the 300-example test is unlabeled.
    ChatTask("AbMusu", AbMusu, 0.0, 100),
)
CHAT_TASKS_BY_NAME = {task.name: task for task in CHAT_TASKS}


def create_chat_task(name, *, shuffle=False, seed=42):
    try:
        task = CHAT_TASKS_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"Unknown task: {name!r}. Available: {list(CHAT_TASKS_BY_NAME)}"
        ) from None
    return task.factory(shuffle=shuffle, seed=seed)
