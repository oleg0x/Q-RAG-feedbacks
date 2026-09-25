"""aiohttp clients for concurrent vLLM chat-completion requests."""

import asyncio
import os
from typing import Dict, List, Optional

import aiohttp


class AsyncVllmClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        system_prompt: str,
        api_key: Optional[str] = None,
        thinking: Optional[bool] = None,
        concurrency_limit: int = 50,
        timeout: int = 120,
        **generation_params,
    ):
        base_url = base_url.rstrip("/")
        api_root = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
        self.url = f"{api_root}/chat/completions"
        self.system_prompt = system_prompt
        self.payload = {"model": model, "stream": False, **generation_params}
        if thinking is not None:
            self.payload["chat_template_kwargs"] = {"enable_thinking": thinking}
        self.concurrency_limit = concurrency_limit
        self.timeout = timeout
        effective_api_key = api_key or os.getenv("VLLM_API_KEY")
        self.headers = {"Content-Type": "application/json"}
        if effective_api_key:
            self.headers["Authorization"] = f"Bearer {effective_api_key}"

    async def _send_request(
        self, session: aiohttp.ClientSession, messages: List[Dict[str, str]]
    ) -> Dict:
        payload = dict(self.payload)
        payload["messages"] = [
            {"role": "system", "content": self.system_prompt},
            *messages,
        ]
        for attempt in range(3):
            try:
                async with session.post(
                    self.url, json=payload, headers=self.headers
                ) as response:
                    response.raise_for_status()
                    return await response.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt == 2:
                    return {"error": str(exc), "status": "failed"}
                await asyncio.sleep(2**attempt)
        raise AssertionError("unreachable")

    async def generate_batch(self, requests: List[str]) -> List[Dict]:
        if not requests:
            return []
        connector = aiohttp.TCPConnector(
            limit=self.concurrency_limit,
            limit_per_host=self.concurrency_limit,
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                self._send_request(session, [{"role": "user", "content": request}])
                for request in requests
            ]
            return await asyncio.gather(*tasks)


class AsyncBatchedVllmClient(AsyncVllmClient):
    async def generate_batch(self, requests: List[str]) -> List[Dict]:
        results = []
        for start in range(0, len(requests), self.concurrency_limit):
            results.extend(
                await super().generate_batch(
                    requests[start : start + self.concurrency_limit]
                )
            )
        return results
