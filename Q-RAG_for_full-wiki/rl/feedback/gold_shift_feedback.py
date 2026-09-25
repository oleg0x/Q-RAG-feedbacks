"""
Gold-Shift Feedback Model for Q-RAG training.

Implements reward:
  r(x, c) = log p_θ(y | x, c) − β(y)

where β(y) = log p_θ(y | x, c₀) is the gold answer log-likelihood
without context (computed once at reset), and the reward is the
log-likelihood shift when Q-RAG-provided context is added.

Requires a running vLLM server.
"""

from __future__ import annotations
import asyncio
import logging
import os
from typing import List

import aiohttp

from rl.feedback.feedback import AFeedbackModel

logger = logging.getLogger(__name__)


# ── Prompt templates ───────────────────────────────────────────────────────
QA_INSTRUCTION_PROMPT = """You are a factoid question answering system.
Return only the answer itself.
The answer must be a single entity, number, date, yes/no, or short phrase.
No explanations, no reasoning, no extra text.
Do not write labels such as 'Final answer:' or 'Answer:'."""

QA_PROMPT = """
GIVEN PASSAGES:
{context}

QUESTION:
{question}

Return only the short answer."""


class GoldShiftFeedback(AFeedbackModel):
    """
    Reward = log p(gold_answer | question, context) − β(gold_answer)

    β is computed once at reset() with empty context.
    Reward is computed at the final step with Q-RAG-provided context.
    Total: 2 vLLM calls per episode.
    """
    FEEDBACK_MODEL_NAME = "gold_shift"

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        model_name: str = "Qwen/Qwen3-4B",
        only_at_final: bool = True,
        never_terminate: bool = True,
        reward_scaling: float = 1.0,
        api_key: str = None,
    ):
        super().__init__(never_terminate=never_terminate)
        api_base_url = api_base_url.rstrip("/")
        self.api_base_url = (
            api_base_url[:-3] if api_base_url.endswith("/v1") else api_base_url
        )
        self.model_name = model_name
        self.only_at_final = only_at_final
        self.reward_scaling = reward_scaling
        self.api_key = api_key or os.getenv("VLLM_API_KEY")
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        # Per-episode state
        self.question = None
        self.answer = None
        self.beta = None

    # ── Public interface ───────────────────────────────────────────────────

    def reset(self, obs, info):
        super().reset(obs, info)
        self.question = obs["question"]
        self.answer = info["answer"]

        # Compute β(y) = log p(y | x, c₀) with empty context
        self.beta = self._score_answer(self.question, "", self.answer)
        logger.debug(f"GoldShiftFeedback reset: β = {self.beta:.4f}")

    def reward(self, obs, info, is_final=False) -> float:
        if self.only_at_final and not is_final:
            return 0.0

        if self.beta is None:
            logger.warning("GoldShiftFeedback: beta is None (reset not called?)")
            return 0.0

        pred_chunks = obs.get("pred_chunks", [])
        context_str = "\n\n".join(pred_chunks)

        log_p = self._score_answer(self.question, context_str, self.answer)
        if log_p is None:
            logger.warning("GoldShiftFeedback: scoring returned None")
            return 0.0

        reward = (log_p - self.beta) * self.reward_scaling
        return reward

    def copy(self):
        return GoldShiftFeedback(
            api_base_url=self.api_base_url,
            model_name=self.model_name,
            only_at_final=self.only_at_final,
            never_terminate=self.never_terminate,
            reward_scaling=self.reward_scaling,
            api_key=self.api_key,
        )

    # ── Scoring ────────────────────────────────────────────────────────────

    def _score_answer(self, question: str, context: str, answer: str) -> float | None:
        """Synchronous wrapper: compute log p(answer | prompt(question, context))."""

        async def _run():
            prompt_ids = await self._tokenize_chat(question, context)
            return await self._score_one(prompt_ids, answer)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(_run(), loop).result()
            else:
                return asyncio.run(_run())
        except RuntimeError:
            return asyncio.run(_run())

    async def _tokenize_chat(self, question: str, context: str) -> List[int]:
        """Tokenize the chat prompt (system + user) via vLLM /tokenize endpoint."""
        user_content = QA_PROMPT.format(context=context, question=question)
        messages = [
            {"role": "system", "content": QA_INSTRUCTION_PROMPT},
            {"role": "user", "content": user_content},
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }

        connector = aiohttp.TCPConnector(limit=1, force_close=True)
        async with aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30.0),
        ) as session:
            async with session.post(
                f"{self.api_base_url}/tokenize",
                json=payload,
                headers=self.headers,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["tokens"]

    async def _tokenize_text(self, session: aiohttp.ClientSession, text: str) -> List[int]:
        """Tokenize plain text via vLLM /tokenize endpoint."""
        payload = {"model": self.model_name, "prompt": text}
        async with session.post(
            f"{self.api_base_url}/tokenize",
            json=payload,
            headers=self.headers,
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["tokens"]

    async def _score_one(self, prompt_ids: List[int], answer: str) -> float | None:
        """Compute log p(answer | prompt) using prompt_logprobs."""
        try:
            connector = aiohttp.TCPConnector(limit=1, force_close=True)
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=30.0),
            ) as session:
                # Tokenize answer
                answer_ids = await self._tokenize_text(session, answer.strip())
                if not answer_ids:
                    return None

                n_prompt = len(prompt_ids)
                full_ids = prompt_ids + answer_ids

                # Get logprobs
                payload = {
                    "model": self.model_name,
                    "prompt": full_ids,
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "prompt_logprobs": 0,
                }

                async with session.post(
                    f"{self.api_base_url}/v1/completions",
                    json=payload,
                    headers=self.headers,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                prompt_logprobs = data["choices"][0].get("prompt_logprobs")
                if prompt_logprobs is None:
                    logger.warning("prompt_logprobs not returned")
                    return None

                total_logprob = 0.0
                for i in range(n_prompt, len(prompt_logprobs)):
                    entry = prompt_logprobs[i]
                    if entry is None:
                        continue
                    for _, info in entry.items():
                        total_logprob += info["logprob"]
                        break

                return total_logprob
        except Exception as e:
            logger.warning(f"Error scoring answer '{answer[:30]}': {e}")
            return None
