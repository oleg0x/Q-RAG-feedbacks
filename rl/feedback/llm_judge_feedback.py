"""
LLM-as-a-Judge Feedback for Q-RAG training.

Reward = 1.0  if the judge LLM says the predicted answer is correct
         0.0  otherwise (including API errors)

Uses:
  - A vLLM-backed LLMGenerator to produce a short answer from the retrieved
    context.
  - The shared JUDGE_USER_PROMPT from prompts_and_metrics.llm_judge_prompt
    (single source of truth, same prompt used in scripts/eval_answer_quality.py).
  - A vLLM /v1/chat/completions call with ``max_tokens=8`` and greedy decoding;
    the judge model replies "yes" or "no".

Custom hyperparameters live in configs/feedback/base_rewards.yaml:
  feedback.judge_model      — served-model-name for the judge vLLM instance
  feedback.judge_api_url    — judge vLLM base URL (defaults to feedback.api_base_url)
  feedback.judge_max_tokens — max output tokens for the judge (default: 8)

General hyperparameters (api_base_url, model, reward_scaling, …) are taken
from the same file, mirroring the feedback.* namespace of defaults.yaml.
"""

from __future__ import annotations

import re
import logging
import time

import aiohttp

from rl.feedback.feedback import AFeedbackModel
from rl.feedback.llm_feedback import LLMGenerator
from rl.feedback.vllm_http_utils import run_coroutine_sync
from prompts_and_metrics.llm_judge_prompt import JUDGE_USER_PROMPT

logger = logging.getLogger(__name__)


# ── Text helpers ──────────────────────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from Qwen3-style outputs."""
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    result = re.sub(r"<think>.*", "", result, flags=re.DOTALL)
    return result.strip()


# ── Feedback class ────────────────────────────────────────────────────────────

class LLMJudgeFeedback(AFeedbackModel):
    """
    Reward = LLM-as-a-judge correctness score.

    At the final step the feedback model:
      1. Generates a short answer via the LLMGenerator given the retrieved
         context (``obs['pred_chunks']``).
      2. Sends the question, gold answer, and predicted answer to the judge
         model using JUDGE_USER_PROMPT (greedy, max_tokens=8, no thinking).
      3. Returns reward_scaling if the judge replies "yes", else 0.0.

    The judge vLLM server can be the same as the generator server or a
    separate instance on a different port.
    """

    FEEDBACK_MODEL_NAME = "llm_judge"

    def __init__(
        self,
        llm_generator: LLMGenerator,
        judge_api_url: str = "http://localhost:8000/v1",
        judge_model_name: str = "Qwen/Qwen3-4B",
        judge_api_key: str = "",
        judge_max_tokens: int = 8,
        only_at_final: bool = True,
        reward_scaling: float = 1.0,
        never_terminate: bool = True,
    ):
        super().__init__(never_terminate=never_terminate)
        self.llm = llm_generator
        self.judge_api_url = judge_api_url.rstrip("/")
        self.judge_model_name = judge_model_name
        self.judge_api_key = judge_api_key
        self.judge_max_tokens = judge_max_tokens
        self.only_at_final = only_at_final
        self.reward_scaling = reward_scaling

    # ── AFeedbackModel interface ──────────────────────────────────────────────

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

        score = run_coroutine_sync(lambda: self._judge_async(question, pred, gold))
        logger.debug(
            "LLMJudgeFeedback: pred=%r  gold=%r  judge=%.0f",
            pred[:60],
            gold[:60],
            score,
        )
        return score * self.reward_scaling

    def copy(self) -> "LLMJudgeFeedback":
        return LLMJudgeFeedback(
            llm_generator=self.llm,
            judge_api_url=self.judge_api_url,
            judge_model_name=self.judge_model_name,
            judge_api_key=self.judge_api_key,
            judge_max_tokens=self.judge_max_tokens,
            only_at_final=self.only_at_final,
            reward_scaling=self.reward_scaling,
            never_terminate=self.never_terminate,
        )

    # ── Judge call ────────────────────────────────────────────────────────────

    async def _judge_async(self, question: str, pred: str, gold: str) -> float:
        """Call the judge vLLM server; return 1.0 for 'yes', 0.0 for 'no'/error."""
        prompt = JUDGE_USER_PROMPT.format(question=question, gold=gold, pred=pred)
        payload = {
            "model": self.judge_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.judge_max_tokens,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        if self.judge_api_key:
            headers["Authorization"] = f"Bearer {self.judge_api_key}"

        timeout = aiohttp.ClientTimeout(total=30.0, connect=5.0)
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, headers=headers
            ) as session:
                async with session.post(
                    f"{self.judge_api_url}/chat/completions", json=payload
                ) as resp:
                    if resp.status != 200:
                        logger.warning(
                            "LLMJudgeFeedback: judge API returned HTTP %d", resp.status
                        )
                        return 0.0
                    data = await resp.json()
            answer = data["choices"][0]["message"]["content"].strip().lower()
            return 1.0 if answer.startswith("yes") else 0.0
        except Exception as exc:
            logger.warning("LLMJudgeFeedback: judge API error: %s", exc)
            return 0.0
