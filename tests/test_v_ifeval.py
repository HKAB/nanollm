from tasks.v_ifeval import VIFEval, aggregate_results, evaluate_response


class ContainsChecker:
    def __init__(self, instruction_id):
        self.instruction_id = instruction_id

    def build_description(self, *, keyword=None, prompt=None):
        if keyword is not None:
            self.keyword = keyword

    def get_instruction_args(self):
        return {}

    def check_following(self, response):
        return self.keyword in response


class WrappedChecker(ContainsChecker):
    def check_following(self, response):
        return response == "answer"


REGISTRY = {"contains": ContainsChecker, "wrapped": WrappedChecker}
ROWS = [
    {
        "prompt": "Include alpha and beta.",
        "instruction_id_list": ["contains", "wrapped"],
        "kwargs": [{"keyword": "alpha"}, {"keyword": "unused"}],
    }
]


def test_task_builds_completion_conversation():
    task = VIFEval(dataset=ROWS, registry=REGISTRY)
    example = task[0]

    assert example["messages"] == [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": ""},
    ]


def test_strict_and_loose_follow_official_semantics():
    example = ROWS[0]
    result = evaluate_response(example, "header\nanswer", REGISTRY)

    assert result.strict == (False, False)
    assert result.loose == (False, True)
    assert not result.strict_prompt
    assert not result.loose_prompt


def test_aggregate_uses_prompt_and_instruction_micro_accuracy():
    first = evaluate_response(ROWS[0], "alpha", REGISTRY)
    second = evaluate_response(ROWS[0], "answer", REGISTRY)
    metrics = aggregate_results([first, second])

    assert metrics["strict_prompt_accuracy"] == 0.0
    assert metrics["strict_instruction_accuracy"] == 0.5
    assert metrics["loose_prompt_accuracy"] == 0.0
    assert metrics["loose_instruction_accuracy"] == 0.5


def test_non_string_response_fails_every_instruction():
    result = evaluate_response(ROWS[0], None, REGISTRY)
    assert result.strict == (False, False)
    assert result.loose == (False, False)
