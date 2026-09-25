"""
Shared utilities for the Group IG (candidate_beta) dataset pipeline.

Holds the QA prompt template, vLLM HTTP helpers, atomic JSONL I/O,
and the resume + parallel scaffolding shared by generate_candidates.py,
judge_candidates.py and compute_beta.py.
"""

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Callable, Iterable

# ─── defaults ───
VLLM_BASE_URL = "http://localhost:8000"
VLLM_API_KEY = os.environ.get("VLLM_API_KEY")
DEFAULT_MAX_RETRIES = 3
DEFAULT_SAVE_EVERY = 500
DEFAULT_TIMEOUT = 120

# ─── shared prompts (candidate generation + beta scoring) ───
QA_SYSTEM_PROMPT = """You are a factoid question answering system.
Return only the answer itself.
The answer must be a single entity, number, date, yes/no, or short phrase.
No explanations, no reasoning, no extra text.
Do not write labels such as 'Final answer:' or 'Answer:'."""

QA_USER_TEMPLATE = """GIVEN PASSAGES:
{context}

QUESTION:
{question}

Return only the short answer."""


def qa_messages(question: str, context: str = "") -> list[dict]:
    """Build the QA chat messages. Pass context='' for the no-context baseline."""
    return [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": QA_USER_TEMPLATE.format(context=context, question=question)},
    ]


def strip_candidate_prefix(text: str) -> str:
    """Return the substring after the last 'Final answer:' / 'Answer:' label.

    Handles three patterns observed in generated candidates:
      'Final answer: Paris'        -> 'Paris'
      'Answer: 42'                 -> '42'
      'Loud\\nFinal answer: Loud'  -> 'Loud'
      'Paris'                      -> 'Paris'  (no label, unchanged)

    Used before beta scoring so that records in *_final.jsonl contain exactly
    the string that was scored — and that reward-time scoring will re-tokenize.
    """
    t = text.strip()
    lower = t.lower()
    for label in ("final answer:", "answer:"):
        idx = lower.rfind(label)
        if idx != -1:
            return t[idx + len(label):].strip()
    return t


# ─── JSONL I/O ───
def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl_atomic(path: str, records: Iterable[dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


# ─── vLLM HTTP helpers ───
def _vllm_post(url: str, payload: dict, timeout: int = DEFAULT_TIMEOUT) -> dict:
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def tokenize_chat(
    question: str,
    model: str,
    base_url: str = VLLM_BASE_URL,
) -> list[int]:
    """Tokenize the QA chat prompt (with enable_thinking=False) and return token ids."""
    data = _vllm_post(f"{base_url}/tokenize", {
        "model": model,
        "messages": qa_messages(question),
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    return data["tokens"]


def tokenize_text(
    text: str,
    model: str,
    base_url: str = VLLM_BASE_URL,
) -> list[int]:
    data = _vllm_post(f"{base_url}/tokenize", {
        "model": model,
        "prompt": text,
    })
    return data["tokens"]


def score_text(
    prompt_token_ids: list[int],
    text: str,
    model: str,
    base_url: str = VLLM_BASE_URL,
) -> float:
    """Sum of token-level log-probs of `text` conditioned on `prompt_token_ids`.

    Does NOT normalize `text` — pass a pre-normalized string so that the same
    bytes are scored at beta-time and at reward-time.
    """
    candidate_token_ids = tokenize_text(text, model, base_url)
    if not candidate_token_ids:
        return float("-inf")

    n_prompt = len(prompt_token_ids)
    data = _vllm_post(f"{base_url}/v1/completions", {
        "model": model,
        "prompt": prompt_token_ids + candidate_token_ids,
        "max_tokens": 1,
        "temperature": 0.0,
        "prompt_logprobs": 0,
    })

    prompt_logprobs = data["choices"][0].get("prompt_logprobs")
    if prompt_logprobs is None:
        raise ValueError("prompt_logprobs not returned by vLLM")

    total = 0.0
    for i in range(n_prompt, len(prompt_logprobs)):
        entry = prompt_logprobs[i]
        if entry is None:
            continue
        for _, info in entry.items():
            total += info["logprob"]
            break
    return total


def chat_completion(
    messages: list[dict],
    model: str,
    base_url: str = VLLM_BASE_URL,
    max_tokens: int = 8,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    n: int = 1,
) -> list[str]:
    """Return n stripped chat completions. Caller uses [0] when n=1."""
    data = _vllm_post(f"{base_url}/v1/chat/completions", {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "n": n,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    })
    return [c["message"]["content"].strip() for c in data["choices"]]


# ─── resume + parallel scaffolding ───
def run_with_resume(
    samples: list[dict],
    output_path: str,
    process_fn: Callable[[dict], dict],
    max_workers: int,
    save_every: int = DEFAULT_SAVE_EVERY,
    progress_every: int = 2000,
    sanity_check: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Apply process_fn to every sample, writing results to output_path.

    - Skips samples whose "id" already appears in output_path (resume).
    - Runs process_fn concurrently with max_workers threads.
    - Flushes output_path atomically every save_every completed records.
    """
    print(f"Input: {len(samples)} samples")

    results: list[dict] = []
    done_ids: set = set()
    if os.path.exists(output_path):
        results = read_jsonl(output_path)
        done_ids = {r["id"] for r in results}
        print(f"Resuming: {len(done_ids)} already computed")

    remaining = [s for s in samples if s["id"] not in done_ids]
    print(f"Remaining to compute: {len(remaining)}")

    if not remaining:
        print("Nothing to do!")
        return results

    if sanity_check is not None:
        sanity_check(remaining[0])

    write_lock = Lock()
    counter = {"n": 0}

    def _track(sample):
        result = process_fn(sample)
        with write_lock:
            results.append(result)
            counter["n"] += 1
            if counter["n"] % save_every == 0:
                write_jsonl_atomic(output_path, results)
                print(f"  Saved {len(results)} total")
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_track, s): s for s in remaining}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            try:
                future.result()
            except Exception as e:
                sid = futures[future].get("id", "?")
                print(f"[FATAL] Sample {sid}: {e}")
            if done_count % progress_every == 0:
                print(f"Progress: {done_count}/{len(remaining)}")

    write_jsonl_atomic(output_path, results)
    print(f"\nDone! Total computed: {len(results)}")
    return results
