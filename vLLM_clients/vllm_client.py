"""OpenAI-SDK client used by Q-ICL feedback models."""

import asyncio
import logging
import os
from typing import List, Optional, Tuple

import httpx
import numpy as np
from openai import AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)


class BasicVllmClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        max_tokens: int,
        thinking: bool,
        sys_prompt: str = None,
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
            context = "\n".join(passages)
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
    def _choice_cumulative_logprob(choice) -> Optional[float]:
        """Sum of token log-probabilities for the sampled completion (Eq. 13 style)."""
        if not choice.logprobs or not getattr(choice.logprobs, "content", None):
            return None
        tokens = choice.logprobs.content
        if not tokens:
            return None
        return float(sum(item.logprob for item in tokens))
    
    @staticmethod
    def _extract_response_data(response, get_cumulative=None) -> Tuple[str, Optional[float]]:
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            return "[unfinished content]", None
        text = choice.message.content
        if not text:
            raise ValueError("Response content is empty")
        if get_cumulative:
            return text, self._choice_cumulative_logprob(choice)
        
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
        get_cumulative: Optional[bool] = None
    ) -> Tuple[str, Optional[float]]:
        messages = self._prepare_messages(query, examples, passages, two_answers)
        response = self._client.chat.completions.create(**self._request_kwargs(messages))
        return self._extract_response_data(response, get_cumulative=get_cumulative)

    async def chat_completion_async(
        self,
        query: str,
        examples: Optional[List[Tuple[str, str]]] = None,
        passages: Optional[List[str]] = None,
        get_cumulative: Optional[bool] = None
    ) -> Tuple[str, Optional[float]]:
        async with self._semaphore:
            messages = self._prepare_messages(query, examples, passages)
            response = await self._async_client.chat.completions.create(
                **self._request_kwargs(messages)
            )
            return self._extract_response_data(response, get_cumulative=get_cumulative)

    def _extra_body(self, temperature: float) -> dict:
        body = {"chat_template_kwargs": {"enable_thinking": self._thinking}}
        if temperature == 0.0:
            body["top_k"] = 1
        return body
    
    def _extract_all_response_data(self, response) -> List[Tuple[str, Optional[float]]]:
        """Extract (text, cumulative_logprob_or_None) from every choice."""
        results = []
        for choice in response.choices:
            if choice.finish_reason != "stop":
                results.append(("[unfinished content]", None))
                continue
            text = choice.message.content or ""
            cumulative_lp = self._choice_cumulative_logprob(choice)
            results.append((text, cumulative_lp))
        return results
    
    def chat_completion_n(self,
        query: str,
        n: int,
        examples: Optional[List[Tuple[str, str]]] = None,
        passages: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        *,
        logprobs: bool = False,
    ) -> List[Tuple[str, Optional[float]]]:
        """Return ``n`` independent completions as ``[(text, cumulative_logprob), ...]``.

        ``temperature`` and ``top_p`` override the instance defaults for this
        call only, allowing callers (e.g. ``InfoGainFeedback``) to request
        stochastic diversity without changing the client's default behaviour.
        When ``n == 1`` this delegates to the regular ``chat_completion`` so
        the code path is identical to before.

        When ``logprobs=True``, requests token logprobs and returns the **sum** of
        sampled-token log probabilities per completion (sequence score).
        """
        if n == 1:
            return [self.chat_completion(query, examples, passages, logprobs=logprobs)]

        t = temperature if temperature is not None else self._temperature
        p = top_p if top_p is not None else self._top_p
        messages = self._prepare_messages(query, examples, passages)

        try:
            lp_kwargs = {}
            if logprobs:
                lp_kwargs["logprobs"] = True
                lp_kwargs["top_logprobs"] = 1

            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=t,
                top_p=p,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=n,
                extra_body=self._extra_body(t),
                **lp_kwargs,
            )
            return self._extract_all_response_data(response)

        except Exception as e:
            logger.error(f"Error in chat_completion_n (n={n}): {str(e)}")
            raise

    async def chat_completion_n_async(self,
        query: str,
        n: int,
        examples: Optional[List[Tuple[str, str]]] = None,
        passages: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        *,
        logprobs: bool = False,
    ) -> List[Tuple[str, Optional[float]]]:
        """Async variant of ``chat_completion_n``."""
        if n == 1:
            return [
                await self.chat_completion_async(
                    query, examples, passages, logprobs=logprobs
                )
            ]

        t = temperature if temperature is not None else self._temperature
        p = top_p if top_p is not None else self._top_p

        async with BasicVllmClient._semaphore:
            BasicVllmClient._active_requests += 1
            messages = self._prepare_messages(query, examples, passages)

            try:
                lp_kwargs = {}
                if logprobs:
                    lp_kwargs["logprobs"] = True
                    lp_kwargs["top_logprobs"] = 1

                response = await self._async_client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=self._max_tokens,
                    temperature=t,
                    top_p=p,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    n=n,
                    extra_body=self._extra_body(t),
                    **lp_kwargs,
                )
                return self._extract_all_response_data(response)

            except Exception as e:
                logger.error(f"Error in chat_completion_n_async (n={n}): {str(e)}")
                raise
            finally:
                BasicVllmClient._active_requests -= 1
