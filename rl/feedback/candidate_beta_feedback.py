"""
Candidate-Beta Feedback Model for Q-RAG training.

Implements reward from Eq. 5 of the paper:
  r(x, c) = mean_{g∈G⁺}(log p(g|x,c) - β(g)) - mean_{g∈G⁻}(log p(g|x,c) - β(g))

Requires a running vLLM server with the feedback LLM (e.g. Qwen3-4B).
"""

from __future__ import annotations
import asyncio
import logging
import os
from typing import List, Union

import aiohttp

from rl.feedback.feedback import AFeedbackModel
from rl.feedback.vllm_http_utils import (
    normalize_vllm_http_base_url,
    run_coroutine_sync,
    token_ids_from_tokenize_json,
)

logger = logging.getLogger(__name__)


# ── Prompt templates (must match gig_common.QA_SYSTEM_PROMPT / QA_USER_TEMPLATE
#    byte-for-byte so beta and reward are on the same likelihood scale) ──────
QA_INSTRUCTION_PROMPT = """You are a factoid question answering system.
Return only the answer itself.
The answer must be a single entity, number, date, yes/no, or short phrase.
No explanations, no reasoning, no extra text.
Do not write labels such as 'Final answer:' or 'Answer:'."""

QA_PROMPT = """GIVEN PASSAGES:
{context}

QUESTION:
{question}

Return only the short answer."""


class CandidateBetaFeedback(AFeedbackModel):
    """
    Reward = mean log-likelihood shift of correct candidates
             minus mean log-likelihood shift of incorrect candidates.
    """
    FEEDBACK_MODEL_NAME = "candidate_beta"

    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        model_name: str = "Qwen/Qwen3-4B",
        max_concurrent: int = 20,
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
        self.max_concurrent = max_concurrent
        self.only_at_final = only_at_final
        self.reward_scaling = reward_scaling
        self.api_key = api_key or os.getenv("VLLM_API_KEY")
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        # Per-episode state
        self.candidates = None
        self.judgements = None
        self.betas = None

    def reset(self, obs, info):
        super().reset(obs, info)
        self.candidates = info.get("candidates")
        self.judgements = info.get("judgements")
        self.betas = info.get("betas")

    def reward(self, obs, info, is_final=False) -> float:
        if self.only_at_final and not is_final:
            return 0.0

        if self.candidates is None or self.judgements is None or self.betas is None:
            logger.warning("CandidateBetaFeedback: missing candidates/judgements/betas")
            return 0.0

        question = obs["question"]
        pred_chunks = obs.get("pred_chunks", [])

        # Build context from retrieved chunks
        context_str = "\n\n".join(pred_chunks)

        # Score all candidates with context
        log_probs = self._score_candidates(question, context_str, self.candidates)

        # Compute reward per Eq. 5
        G_plus_shifts = []
        G_minus_shifts = []

        for lp, j, beta in zip(log_probs, self.judgements, self.betas):
            if lp is None or beta == float("-inf"):
                continue
            shift = lp - beta
            if j == 1:
                G_plus_shifts.append(shift)
            else:
                G_minus_shifts.append(shift)

        # Convention: if G+ or G- is empty, that term is 0
        mean_plus = sum(G_plus_shifts) / len(G_plus_shifts) if G_plus_shifts else 0.0
        mean_minus = sum(G_minus_shifts) / len(G_minus_shifts) if G_minus_shifts else 0.0

        reward = (mean_plus - mean_minus) * self.reward_scaling
        return reward

    def _score_candidates(
        self, question: str, context: str, candidates: List[str]
    ) -> List[float]:
        """Score all candidates via vLLM log-likelihood API."""

        async def _score_all():
            # First, tokenize the prompt (shared across candidates)
            prompt_ids = await self._tokenize_chat(question, context)

            sem = asyncio.Semaphore(self.max_concurrent)
            tasks = [
                self._score_one(prompt_ids, cand, sem) for cand in candidates
            ]
            return await asyncio.gather(*tasks)

        return run_coroutine_sync(lambda: _score_all())

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

    async def _score_one(
        self, prompt_ids: List[int], candidate: str, sem: asyncio.Semaphore
    ) -> float:
        """Compute log p(candidate | prompt) using prompt_logprobs."""
        async with sem:
            try:
                connector = aiohttp.TCPConnector(limit=1, force_close=True)
                async with aiohttp.ClientSession(
                    connector=connector,
                    timeout=aiohttp.ClientTimeout(total=30.0),
                ) as session:
                    # Tokenize candidate
                    cand_ids = await self._tokenize_text(session, candidate.strip())
                    if not cand_ids:
                        return None

                    n_prompt = len(prompt_ids)
                    full_ids = prompt_ids + cand_ids

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
                logger.warning(f"Error scoring candidate '{candidate[:30]}': {e}")
                return None

    def copy(self):
        return CandidateBetaFeedback(
            api_base_url=self.api_base_url,
            model_name=self.model_name,
            max_concurrent=self.max_concurrent,
            only_at_final=self.only_at_final,
            never_terminate=self.never_terminate,
            reward_scaling=self.reward_scaling,
            api_key=self.api_key,
        )
