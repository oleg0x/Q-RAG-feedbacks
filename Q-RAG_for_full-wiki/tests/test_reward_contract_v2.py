"""Reward contract v2: answer aliases and request parity with evaluation.

The training reward and the evaluation metric must be the same quantity. The
evaluation script (`answer_judge_llms.py`) is the reference, so this file checks
exactly where training used to differ from it: the chunk separator, what is
sent to the judge, the judge's `max_tokens` and the reasoning reader.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prompts_and_metrics import prompts
from rl.feedback.llm_answer import LlmAnswer, answer_variants
from vLLM_clients.vllm_client import BasicVllmClient


EVAL_SYS_QA = prompts.sys_qa


class FakeClient:
    """Stand-in for vLLM: records requests and returns canned replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        if not self.replies:
            raise AssertionError("Unexpected extra request to vLLM")
        return self.replies.pop(0), None


def make_feedback(reader_replies, judge_replies=()):
    feedback = LlmAnswer.__new__(LlmAnswer)
    feedback.task = "NQ+HotPotQA"
    feedback.rouge = None
    feedback.retries = 1
    feedback.retry_backoff = 0.0
    feedback.retry_backoff_max = 0.0
    feedback.stall_timeout = None
    feedback.error_count = 0
    feedback.request_count = 0
    feedback.last_transition_valid = True
    feedback._first_failure_at = None
    feedback.last_metrics = {}
    feedback.base_url = "http://fake/v1"
    feedback.vllm_client = FakeClient(reader_replies)
    feedback.vllm_client_judge = FakeClient(judge_replies)
    return feedback


def reward_of(feedback, variants, prediction_chunks=("chunk one", "chunk two")):
    obs = {
        "question": "who wrote it?",
        "sample_id": "x",
        "pred_idx": [0, 1],
        "pred_chunks": list(prediction_chunks),
    }
    info = {"answer": variants[0], "answer_variants": list(variants)}
    return feedback.reward(obs, info, is_final=True)


# --------------------------------------------------------------------------
# answer aliases


def test_single_variant_reward_and_metrics_are_unchanged() -> None:
    """Single-answer example: EM fires and the judge is not called."""
    feedback = make_feedback(["Final Answer: The Beatles"])
    assert reward_of(feedback, ["the beatles"]) == 1.0
    metrics = feedback.get_metrics()
    assert metrics["EM"] == 1
    assert metrics["em_alias"] == 1
    assert metrics["judge"] is None
    assert feedback.vllm_client_judge.calls == []


def test_second_variant_wins_where_the_first_one_misses() -> None:
    """Multi-alias example: EM takes the max, no need to call the judge.

    Joining aliases with ', ' broke exactly this: the prediction was compared
    with the string "Dai Xiuli, Dai Yongge, Yongge Dai", matched nothing and
    went to the judge.
    """
    feedback = make_feedback(["Final Answer: Dai Yongge"])
    variants = ["Xiu Li Dai", "Dai Xiuli", "Dai Yongge", "Yongge Dai"]
    assert reward_of(feedback, variants) == 1.0
    metrics = feedback.get_metrics()
    # EM against the primary answer is zero and is logged separately: the
    # final metric must not be silently replaced by the alias version.
    assert metrics["EM"] == 0
    assert metrics["em_alias"] == 1
    assert metrics["answer_variants"] == 4
    assert feedback.vllm_client_judge.calls == []


def test_joined_variants_would_have_missed() -> None:
    """Old behaviour: the joined string matches none of the aliases.

    The same answer on the same example with aliases joined by ', ' gives
    EM = 0 and goes to the judge, which affected about half of NQ examples.
    """
    feedback = make_feedback(
        ["Final Answer: Dai Yongge"], ["Final Answer: INCORRECT"]
    )
    joined = ", ".join(["Xiu Li Dai", "Dai Xiuli", "Dai Yongge"])
    assert reward_of(feedback, [joined]) == 0.0
    assert len(feedback.vllm_client_judge.calls) == 1


def test_judge_is_called_with_all_variants_and_raw_strings() -> None:
    feedback = make_feedback(
        ["Final Answer: a thousand"], ["Final Answer: CORRECT"]
    )
    assert reward_of(feedback, ["1,000", "one thousand"]) == 1.0
    call = feedback.vllm_client_judge.calls[0]
    prediction, reference = call["two_answers"]
    # Raw strings, not normalize_answer: only they let the judge tell "1,000"
    # from "one thousand", since normalization strips punctuation.
    assert prediction == "a thousand"
    assert reference == "1,000 | one thousand"
    metrics = feedback.get_metrics()
    assert metrics["em_alias"] == 0
    assert metrics["judge"] == 1.0


def test_answer_variants_falls_back_to_the_single_answer() -> None:
    assert answer_variants({"answer": "yes"}) == ["yes"]
    assert answer_variants({"answer_variants": ["a", "b"]}) == ["a", "b"]
    assert answer_variants({"answer_variants": []}) == [""]


# --------------------------------------------------------------------------
# four former discrepancies with evaluation


def eval_reader_request(chunks, question):
    """Reader request exactly as answer_judge_llms.py builds it."""
    context = "\n\n".join(chunks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nFinal Answer:"


def test_reader_request_is_byte_identical_to_the_eval_one() -> None:
    client = BasicVllmClient.__new__(BasicVllmClient)
    client._system_message = {"role": "system", "content": EVAL_SYS_QA}
    chunks = ['"Alpha"\nfirst chunk text', '"Beta"\nsecond chunk text']
    question = "Were Scott Derrickson and Ed Wood of the same nationality?"

    messages = client._prepare_messages(query=question, passages=chunks)

    assert messages[0]["content"] == EVAL_SYS_QA
    assert messages[1]["content"] == eval_reader_request(chunks, question)


def test_judge_prompt_differs_from_v1_only_in_the_reference_description() -> None:
    v1 = prompts.sys_judge
    v2 = prompts.sys_judge_v2
    assert v1 != v2
    assert prompts.JUDGE_REFERENCE_V1 in v1
    assert prompts.JUDGE_REFERENCE_V2 in v2
    # Judge strictness is untouched: only the reference description changes.
    assert v2 == v1.replace(
        prompts.JUDGE_REFERENCE_V1, prompts.JUDGE_REFERENCE_V2
    )


def test_thinking_reader_is_refused() -> None:
    with pytest.raises(ValueError, match="thinking"):
        LlmAnswer(
            model="Qwen3-4B",
            base_url="http://fake/v1",
            max_tokens=1000,
            thinking=True,
            task="NQ+HotPotQA",
        )


def test_judge_max_tokens_defaults_to_the_eval_value() -> None:
    import inspect

    signature = inspect.signature(LlmAnswer.__init__)
    assert signature.parameters["judge_max_tokens"].default == 100
