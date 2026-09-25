"""
Exact-Match Feedback for Q-RAG training.

Reward = EM(generated_answer, gold_answer)  ∈ {0.0, 1.0}

Uses a vLLM-backed LLMGenerator to produce a short answer from the retrieved
context, then scores it with ``compute_exact_match`` from
``prompts_and_metrics.general_qa`` (SQuAD-style normalization).

Qwen3 <think>…</think> blocks are stripped before scoring so that thinking
tokens never pollute the prediction string.

General hyperparameters (api_base_url, model, reward_scaling, …) are taken
from configs/feedback/base_rewards.yaml (same names as defaults.yaml).
"""

from __future__ import annotations

import re
import logging

from prompts_and_metrics.general_qa import compute_exact_match
from rl.feedback.feedback import AFeedbackModel
from rl.feedback.llm_feedback import LLMGenerator

logger = logging.getLogger(__name__)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3-style outputs."""
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    result = re.sub(r"<think>.*", "", result, flags=re.DOTALL)
    return result.strip()


class EMFeedback(AFeedbackModel):
    """
    Reward = EM(generated_answer, gold_answer).

    At the final step the feedback model:
      1. Generates a short answer via the LLMGenerator given the retrieved
         context (``obs['pred_chunks']``).
      2. Strips Qwen3 thinking tokens from the prediction.
      3. Returns 1.0 * reward_scaling if ``compute_exact_match`` (SQuAD
         normalization) matches the gold answer, else 0.0.
    """

    FEEDBACK_MODEL_NAME = "em"

    def __init__(
        self,
        llm_generator: LLMGenerator,
        only_at_final: bool = True,
        reward_scaling: float = 1.0,
        never_terminate: bool = True,
    ):
        super().__init__(never_terminate=never_terminate)
        self.llm = llm_generator
        self.only_at_final = only_at_final
        self.reward_scaling = reward_scaling

    def reset(self, obs, info) -> None:
        super().reset(obs, info)

    def reward(self, obs, info, is_final: bool = False) -> float:
        if self.only_at_final and not is_final:
            return 0.0

        question = obs["question"]
        pred_chunks = obs.get("pred_chunks", [])
        gold = info.get("answer", "")

        raw_pred = self.llm.generate_answer(question, pred_chunks)
        pred = _strip_thinking(raw_pred)
        score = float(compute_exact_match(pred, gold))
        logger.debug(
            "EMFeedback: pred=%r  gold=%r  em=%.0f", pred[:60], gold[:60], score
        )
        return score * self.reward_scaling

    def copy(self) -> "EMFeedback":
        return EMFeedback(
            llm_generator=self.llm,
            only_at_final=self.only_at_final,
            reward_scaling=self.reward_scaling,
            never_terminate=self.never_terminate,
        )
