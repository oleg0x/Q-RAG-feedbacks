"""Unit tests for the PQN fixes except masking (that is in test_search_line_a).

The mask before `topk` is tested together with the rest of the line A geometry;
here: the α schedule, vLLM retries and excluding invalid transitions from the loss.
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from rl.agents.pqn import AlphaSchedule
from rl.feedback.llm_answer import LlmAnswer, VllmUnavailableError


# --------------------------------------------------------------------------
# Alpha schedule
# --------------------------------------------------------------------------


def test_constant_schedule_ignores_the_step() -> None:
    schedule = AlphaSchedule(start=0.005)

    assert schedule.value(0) == 0.005
    assert schedule.value(10_000) == 0.005


def test_linear_schedule_falls_from_start_to_final() -> None:
    schedule = AlphaSchedule(start=1.0, kind="linear", final=0.2, warmup=100, total=1100)

    assert schedule.value(0) == 1.0
    assert schedule.value(100) == 1.0
    assert schedule.value(600) == pytest.approx(0.6)
    assert schedule.value(1100) == pytest.approx(0.2)
    assert schedule.value(9999) == pytest.approx(0.2)


def test_linear_schedule_requires_its_horizon() -> None:
    with pytest.raises(ValueError, match="total"):
        AlphaSchedule(start=1.0, kind="linear", final=0.2)


def test_schedule_is_read_from_the_config_and_defaults_to_constant() -> None:
    """Without a config section α stays constant and does not depend on lr."""
    empty = OmegaConf.create({"pqn": {"hyperparams": {"alpha": 0.005}}})
    assert AlphaSchedule.from_config(empty, 0.005).kind == "constant"

    configured = OmegaConf.create({
        "pqn": {
            "hyperparams": {
                "alpha": 0.005,
                "alpha_schedule": {
                    "kind": "linear",
                    "start": 0.01,
                    "final": 0.001,
                    "total": 200,
                },
            }
        }
    })
    schedule = AlphaSchedule.from_config(configured, 0.005)
    assert schedule.kind == "linear"
    assert schedule.value(0) == 0.01
    assert schedule.value(200) == pytest.approx(0.001)


# --------------------------------------------------------------------------
# A vLLM error is not reward 0
# --------------------------------------------------------------------------


class FlakyClient:
    """Client that fails the first `failures` times, then responds."""

    def __init__(self, failures: int, response: str = "Final answer: paris") -> None:
        self.failures = failures
        self.calls = 0
        self.response = response

    def chat_completion(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("vLLM is down")
        return self.response, None


def make_answer(retries: int = 3, stall_timeout=900.0) -> LlmAnswer:
    """`LlmAnswer` without network: real clients are swapped out right after construction."""
    answer = LlmAnswer.__new__(LlmAnswer)
    answer.completed = False
    answer.never_terminate = True
    answer.model = "toy"
    answer.base_url = "http://127.0.0.1:0/v1"
    answer.max_tokens = 16
    answer.thinking = False
    answer.task = "HotPotQA+2WikiMultihopQA"
    answer.api_key = None
    answer.retries = retries
    answer.retry_backoff = 0.0
    answer.retry_backoff_max = 0.0
    answer.stall_timeout = stall_timeout
    answer.error_count = 0
    answer.request_count = 0
    answer.last_transition_valid = True
    answer._first_failure_at = None
    answer.rouge = None
    answer.last_metrics = {}
    return answer


OBS = {"question": "capital of France", "pred_chunks": ["France ..."]}
INFO = {"answer": "Paris"}


def test_a_recovered_request_still_yields_the_real_reward() -> None:
    answer = make_answer(retries=3)
    answer.vllm_client = FlakyClient(failures=2)
    answer.vllm_client_judge = FlakyClient(failures=0)

    reward = answer.reward(OBS, INFO, is_final=True)

    assert reward == 1.0
    assert answer.last_transition_valid is True
    assert answer.error_stats()["vllm_errors"] == 2


def test_exhausted_retries_mark_the_transition_invalid_not_zero_reward() -> None:
    answer = make_answer(retries=2)
    answer.vllm_client = FlakyClient(failures=99)
    answer.vllm_client_judge = FlakyClient(failures=0)

    reward = answer.reward(OBS, INFO, is_final=True)

    # The reward value is still 0.0, but the flag says it is not a measurement.
    assert reward == 0.0
    assert answer.last_transition_valid is False
    assert answer.error_stats()["vllm_errors"] == 2
    assert answer.get_feedback(OBS, INFO, truncated=True)["valid"] is False


def test_a_silent_server_stops_training_instead_of_poisoning_it() -> None:
    """The stall threshold turns endless retries into a stop with a checkpoint."""
    answer = make_answer(retries=5, stall_timeout=0.0)
    answer.vllm_client = FlakyClient(failures=99)
    answer.vllm_client_judge = FlakyClient(failures=0)

    with pytest.raises(VllmUnavailableError, match="failing"):
        answer.reward(OBS, INFO, is_final=True)


def test_normalize_answer_saves_a_judge_call_on_punctuation() -> None:
    """`normalize_answer`, not `strip().lower()`: no judge call on punctuation alone."""
    answer = make_answer()
    answer.vllm_client = FlakyClient(failures=0, response="Final answer: The Paris.")
    judge = FlakyClient(failures=0, response="Final answer: INCORRECT")
    answer.vllm_client_judge = judge

    reward = answer.reward(OBS, INFO, is_final=True)

    assert reward == 1.0
    assert judge.calls == 0
    assert answer.get_metrics()["EM"] == 1


def test_judge_fallback_is_kept_for_genuinely_different_answers() -> None:
    """The judge fallback stays: reward equals the judge metric by design."""
    answer = make_answer()
    answer.vllm_client = FlakyClient(failures=0, response="Final answer: capital of France")
    judge = FlakyClient(failures=0, response="Final answer: CORRECT")
    answer.vllm_client_judge = judge

    reward = answer.reward(OBS, INFO, is_final=True)

    assert judge.calls == 1
    assert reward == 1.0
    assert answer.get_metrics()["EM"] == 0


# --------------------------------------------------------------------------
# Invalid transitions and the loss
# --------------------------------------------------------------------------


class ConstantCritic(torch.nn.Module):
    """Critic whose logits do not depend on the input: only the loss is tested."""

    # TextQNet contract: train_step compares the calibrated heads with the
    # target. An uncalibrated critic must return the legacy 2·q_i.
    calibrated = False

    def __init__(self, values: torch.Tensor) -> None:
        super().__init__()
        self.values = torch.nn.Parameter(values.clone())

    def forward(self, state, action):
        return self.values / 2, self.values / 2

    def head_values(self, logits_1, logits_2):
        return 2 * logits_1, 2 * logits_2


def run_train_step(values, targets, valid):
    from rl.agents.pqn import PQN

    agent = PQN.__new__(PQN)
    agent.accumulate_grads = 1
    train_step = agent.make_train_step()
    critic = ConstantCritic(values)
    loss = train_step(critic, None, None, targets, valid)
    return float(loss)


def test_invalid_transitions_do_not_reach_the_loss() -> None:
    """An invalid transition must vanish from the loss, not arrive as a zero reward."""
    values = torch.tensor([0.2, 0.8, 0.5])
    targets = torch.tensor([1.0, 0.0, 0.5])

    masked = run_train_step(values, targets, torch.tensor([True, False, True]))
    # The same batch without the second transition at all.
    without = run_train_step(
        torch.tensor([0.2, 0.5]), torch.tensor([1.0, 0.5]), torch.tensor([True, True])
    )
    all_valid = run_train_step(values, targets, torch.tensor([True, True, True]))

    assert masked == pytest.approx(without, rel=1e-6)
    assert masked != pytest.approx(all_valid, rel=1e-6)


def test_loss_without_a_mask_is_byte_identical_to_the_previous_behaviour() -> None:
    """Line B must not notice that the mask was introduced."""
    values = torch.tensor([0.2, 0.8, 0.5])
    targets = torch.tensor([1.0, 0.0, 0.5])

    with_ones = run_train_step(values, targets, torch.tensor([True, True, True]))
    without_mask = run_train_step(values, targets, None)

    assert with_ones == pytest.approx(without_mask, rel=1e-6)
