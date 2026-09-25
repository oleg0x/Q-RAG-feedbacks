# Вендорная байт-копия ../Q-RAG-feedback/vLLM_clients/sync_vllm_client.py
# (в Q-RAG_for_full-wiki лежит тот же файл байт в байт). Копия, а не свой
# клиент: иначе регрессия форка против оригинала проверяла бы наш транспорт,
# а не нашу логику. Ниже этой шапки — оригинал без изменений; дрейф копии
# ловит tests/test_vendored.py там, где оригинал доступен.
"""Synchronous requests-based client for an OpenAI-compatible vLLM server."""

import os
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def extract_response_text(json_response) -> str:
    if not json_response or "choices" not in json_response:
        raise ValueError("Invalid response structure")
    try:
        return json_response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"Unexpected response format: {exc}") from exc


class SyncVllmClient:
    def __init__(
        self,
        llm: str,
        base_url: str,
        system_prompt: str,
        timeout: int = 180,
        api_key: Optional[str] = None,
        thinking: Optional[bool] = None,
        **generation_params,
    ):
        base_url = base_url.rstrip("/")
        api_root = base_url if base_url.endswith("/v1") else f"{base_url}/v1"
        self.url = f"{api_root}/chat/completions"
        self.system_prompt = system_prompt
        self.timeout = timeout
        effective_api_key = api_key or os.getenv("VLLM_API_KEY")
        self.headers = {"Content-Type": "application/json"}
        if effective_api_key:
            self.headers["Authorization"] = f"Bearer {effective_api_key}"
        self.payload = {"model": llm, "stream": False, **generation_params}
        if thinking is not None:
            self.payload["chat_template_kwargs"] = {"enable_thinking": thinking}
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _build_messages(
        self, request: str, examples: Optional[List[Tuple[str, str]]] = None
    ) -> List[Dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for question, answer in examples or []:
            messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
        messages.append({"role": "user", "content": request})
        return messages

    def chat_completion(
        self, request: str, examples: Optional[List[Tuple[str, str]]] = None
    ):
        payload = dict(self.payload)
        payload["messages"] = self._build_messages(request, examples)
        response = self.session.post(
            self.url,
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise RuntimeError(data["error"].get("message", "Unknown API error"))
        return data

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
