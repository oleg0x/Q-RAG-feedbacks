"""Shared vLLM HTTP helpers for feedback models and scripts."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def run_coroutine_sync(factory: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
    """
    Run ``await factory()`` from sync code (Python 3.12-safe; no get_event_loop).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())

    def _in_thread() -> _T:
        return asyncio.run(factory())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_in_thread).result()


def normalize_vllm_http_base_url(url: str) -> str:
    """
    vLLM serves POST /tokenize and POST /v1/completions off the same *root* host.
    Strip a trailing /v1 so /tokenize resolves while /v1/completions still works.
    """
    u = url.rstrip("/")
    if u.lower().endswith("/v1"):
        u = u[:-3].rstrip("/")
    return u


def token_ids_from_tokenize_json(
    data: dict[str, Any],
    *,
    http_status: int,
    hint: str = "",
) -> list[int]:
    """Parse token ids from vLLM /tokenize JSON; raise on API/validation errors."""
    if "tokens" in data and isinstance(data["tokens"], list):
        return data["tokens"]
    if "token_ids" in data and isinstance(data["token_ids"], list):
        return data["token_ids"]
    if "input_ids" in data and isinstance(data["input_ids"], list):
        return data["input_ids"]

    if "error" in data:
        err = data["error"]
        if isinstance(err, dict):
            msg = err.get("message", err.get("type", str(err)))
        else:
            msg = str(err)
        msg_s = str(msg).lower()
        model_hint = ""
        if "does not exist" in msg_s or "model_not_found" in msg_s or "invalid model" in msg_s:
            model_hint = (
                " The JSON `model` field must match vLLM's *served* id (often set via "
                "`--served-model-name`), not necessarily the HuggingFace repo id — "
                "check GET /v1/models on the same host and pass that id as --model_name."
            )
        raise RuntimeError(
            f"Tokenize API returned error (status={http_status}, {hint}): {msg}{model_hint}"
        )

    if "detail" in data:
        raise RuntimeError(
            f"Tokenize request failed (status={http_status}, {hint}): {data['detail']!r}. "
            "If you used --api_base_url ending in /v1, omit /v1 (use the server root, e.g. http://127.0.0.1:8000)."
        )

    logger.error("Unexpected tokenize JSON (status=%s): %s", http_status, data)
    raise KeyError(
        f"Tokenize response has no token list; keys={list(data.keys())!r} status={http_status} {hint}"
    )


def parse_vllm_completions_json(raw: str, *, http_status: int) -> dict[str, Any]:
    """
    Parse the JSON body from POST /v1/completions.

    Raises ``RuntimeError`` on HTTP errors, validation failures, or bodies that
    are not a successful OpenAI-style completion (missing ``choices``).
    """
    try:
        data: Any = json.loads(raw) if raw else {}
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"vLLM /v1/completions returned non-JSON (HTTP {http_status}): {raw[:800]!r}"
        ) from e
    if not isinstance(data, dict):
        raise RuntimeError(
            f"vLLM /v1/completions JSON root must be an object (HTTP {http_status})"
        )
    if http_status != 200:
        raise RuntimeError(
            f"vLLM /v1/completions HTTP {http_status}: {data!r}"[:4000]
        )
    choices = data.get("choices")
    if not choices:
        raise RuntimeError(
            f"vLLM /v1/completions missing 'choices' (HTTP {http_status}): {data!r}"[:4000]
        )
    return data


def logprob_from_openai_style_entry(entry: Any, token_id: int) -> float:
    """
    Read the log-probability for ``token_id`` from one ``prompt_logprobs`` slot.

    vLLM encodes each slot as a dict mapping token id to metadata; JSON uses
    string keys for integer ids.
    """
    if entry is None:
        raise ValueError("prompt_logprobs entry is None for a scored token position")
    if not isinstance(entry, dict):
        raise TypeError(f"expected dict logprob entry, got {type(entry).__name__}")
    sid = str(token_id)
    info = entry.get(token_id)
    if info is None:
        info = entry.get(sid)
    if info is None and len(entry) == 1:
        info = next(iter(entry.values()))
    if info is None:
        raise KeyError(
            f"logprob entry has no key for token_id={token_id} "
            f"(keys={list(entry.keys())[:12]!r})"
        )
    if isinstance(info, dict):
        return float(info["logprob"])
    return float(getattr(info, "logprob"))


def sum_prompt_suffix_logprobs(
    full_token_ids: list[int],
    n_prefix: int,
    prompt_logprobs: list[Any] | None,
) -> float:
    """
    Sum ``log p(full_token_ids[i] | full_token_ids[:i])`` for
    ``i in n_prefix .. len(full_token_ids)-1`` using vLLM ``prompt_logprobs``.
    """
    if prompt_logprobs is None:
        raise ValueError("prompt_logprobs is None")
    total = 0.0
    for i in range(n_prefix, len(full_token_ids)):
        if i >= len(prompt_logprobs):
            raise ValueError(
                f"prompt_logprobs length {len(prompt_logprobs)} < index {i} "
                f"(full length {len(full_token_ids)})"
            )
        total += logprob_from_openai_style_entry(prompt_logprobs[i], full_token_ids[i])
    return total
