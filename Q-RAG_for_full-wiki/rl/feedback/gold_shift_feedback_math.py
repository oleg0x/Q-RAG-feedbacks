"""Math-specific Gold Shift feedback using Q-ICL examples as context."""

import asyncio
from typing import List, Optional, Tuple

import aiohttp

from envs.dataloaders.base import separator
from prompts_and_metrics import prompts
from rl.feedback.gold_shift_feedback import GoldShiftFeedback
from rl.feedback.feedback import AFeedbackModel


def prepare_examples(examples) -> List[Tuple[str, str]]:
    prepared = []
    for example in examples:
        parts = example.split(separator, 1)
        if len(parts) == 2:
            prepared.append((parts[0].strip(), parts[1].strip()))
    return prepared


class GoldShiftFeedbackMATH(GoldShiftFeedback):
    FEEDBACK_MODEL_NAME = "gold_shift_math"

    def reset(self, obs, info):
        AFeedbackModel.reset(self, obs, info)
        self.question = obs["question"]
        self.answer = info["answer"]
        self.beta = self._score_math_answer(self.question, self.answer, [])

    def reward(self, obs, info, is_final=False):
        if self.only_at_final and not is_final:
            return 0.0
        if self.beta is None:
            return 0.0
        log_probability = self._score_math_answer(
            self.question,
            self.answer,
            prepare_examples(obs.get("pred_chunks", [])),
        )
        if log_probability is None:
            return 0.0
        return (log_probability - self.beta) * self.reward_scaling

    def _score_math_answer(self, question, answer, examples):
        async def run():
            prompt_ids = await self._tokenize_math_chat(question, examples)
            return await self._score_one(prompt_ids, answer)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(run(), loop).result()
            return asyncio.run(run())
        except RuntimeError:
            return asyncio.run(run())

    async def _tokenize_math_chat(self, question, examples):
        messages = [{"role": "system", "content": prompts.sys_math}]
        for user_input, assistant_output in examples:
            messages.extend(
                [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": assistant_output},
                ]
            )
        messages.append({"role": "user", "content": question})
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
            ) as response:
                response.raise_for_status()
                return (await response.json())["tokens"]

    def copy(self):
        return GoldShiftFeedbackMATH(
            api_base_url=self.api_base_url,
            model_name=self.model_name,
            only_at_final=self.only_at_final,
            never_terminate=self.never_terminate,
            reward_scaling=self.reward_scaling,
            api_key=self.api_key,
        )
