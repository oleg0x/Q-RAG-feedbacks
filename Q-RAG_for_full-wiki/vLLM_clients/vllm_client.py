"""OpenAI-SDK client used by Q-ICL feedback models."""

import asyncio
import logging
import os
from typing import List, Optional, Tuple

import httpx
import numpy as np
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)

# Разделитель чанков в контексте ридера. Пустая строка, а не перевод строки:
# ровно так склеивает контекст эвал (`answer_judge_llms.py`), а награда
# обучения обязана быть той же величиной, что и колонка в RESULTS.md. Разница
# в один символ меняет токенизацию всего контекста и, значит, ответ ридера.
CHUNK_SEPARATOR = "\n\n"


class BasicVllmClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        max_tokens: int,
        thinking: bool,
        sys_prompt: str,
        api_key: Optional[str] = None,
        max_concurrent: int = 50,
        trust_env: bool = False,
    ):
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        effective_api_key = api_key or os.getenv("VLLM_API_KEY") or "EMPTY"
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        client_kwargs = {"api_key": effective_api_key, "base_url": base_url}
        self._sync_http_client = httpx.Client(trust_env=trust_env)
        self._async_http_client = httpx.AsyncClient(trust_env=trust_env)
        self._client = OpenAI(
            **client_kwargs,
            http_client=self._sync_http_client,
        )
        self._async_client = AsyncOpenAI(
            **client_kwargs,
            http_client=self._async_http_client,
        )
        self._model = model
        self._max_tokens = max_tokens
        self._thinking = thinking
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._system_message = {"role": "system", "content": sys_prompt}
        logger.info(
            "BasicVllmClient initialized: model=%s base_url=%s max_tokens=%s thinking=%s",
            model,
            base_url,
            max_tokens,
            thinking,
        )

    def _prepare_messages(
        self,
        query: Optional[str],
        examples: Optional[List[Tuple[str, str]]] = None,
        passages: Optional[List[str]] = None,
        two_answers: Optional[Tuple[str, str]] = None,
    ) -> List[dict]:
        messages = [self._system_message]
        if query and examples is not None:
            for user_input, assistant_output in examples:
                messages.append({"role": "user", "content": user_input})
                messages.append({"role": "assistant", "content": assistant_output})
            messages.append({"role": "user", "content": query})
        elif query and passages is not None:
            context = CHUNK_SEPARATOR.join(passages)
            messages.append(
                {
                    "role": "user",
                    "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nFinal Answer:",
                }
            )
        elif query and two_answers is not None:
            prediction, ground_truth = two_answers
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"QUESTION: {query}\n"
                        f"PREDICTED ANSWER: {prediction}\n"
                        f"GROUNDTRUTH ANSWER: {ground_truth}"
                    ),
                }
            )
        elif query:
            messages.append({"role": "user", "content": query})
        else:
            raise ValueError("query must be provided")
        return messages

    @staticmethod
    def _extract_response_data(response) -> Tuple[str, Optional[float]]:
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            return "[unfinished content]", None
        text = choice.message.content
        if not text:
            raise ValueError("Response content is empty")
        mean_logprob = None
        if choice.logprobs and choice.logprobs.content:
            mean_logprob = float(np.mean([item.logprob for item in choice.logprobs.content]))
        return text, mean_logprob

    def _request_kwargs(self, messages: List[dict]) -> dict:
        return {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "extra_body": {
                "top_k": 1,
                "chat_template_kwargs": {"enable_thinking": self._thinking},
            },
        }

    def chat_completion(
        self,
        query: Optional[str] = None,
        examples: Optional[List[Tuple[str, str]]] = None,
        passages: Optional[List[str]] = None,
        two_answers: Optional[Tuple[str, str]] = None,
    ) -> Tuple[str, Optional[float]]:
        messages = self._prepare_messages(query, examples, passages, two_answers)
        response = self._client.chat.completions.create(**self._request_kwargs(messages))
        return self._extract_response_data(response)

    async def chat_completion_async(
        self,
        query: str,
        examples: Optional[List[Tuple[str, str]]] = None,
        passages: Optional[List[str]] = None,
    ) -> Tuple[str, Optional[float]]:
        async with self._semaphore:
            messages = self._prepare_messages(query, examples, passages)
            response = await self._async_client.chat.completions.create(
                **self._request_kwargs(messages)
            )
            return self._extract_response_data(response)
