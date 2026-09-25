"""
Judge candidates for HotpotQA using qwen3-4b via vLLM.

For each sample × each candidate, asks the judge whether the candidate
is semantically equivalent to the gold answer (one candidate per request).
Produces hotpot_judged.jsonl with a "judgements" field (list of 0/1).

Usage:
    python judge_candidates.py [--input FILE] [--output FILE] [--workers N]

Supports resume: skips already-judged sample IDs on restart.
Zero external dependencies — uses only Python stdlib.
"""

import argparse
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ─── defaults ───
BASE_URL = "http://localhost:8765"
MODEL_NAME = "Qwen/Qwen3-32B"
VLLM_API_KEY = os.environ.get("VLLM_API_KEY")
MAX_WORKERS = 64
SAVE_EVERY = 500
MAX_RETRIES = 3

# JUDGE_SYSTEM_PROMPT = """You are an answer equivalence judge.
# You will be given a question, a gold (reference) answer, and a single candidate answer.
# Decide whether the candidate is semantically equivalent to the gold answer.
# - Ignore minor differences in formatting, punctuation, articles, capitalization.
# - "Murder" and "murder." are equivalent.
# - A candidate that contains the gold answer is correct.

# Reply with ONLY one word: YES or NO."""

JUDGE_SYSTEM_PROMPT = """You are an answer verification system for question-answering tasks.
Your task is to compare a PREDICTED ANSWER against the GROUNDTRUTH ANSWER and determine if they are semantically equivalent.
Strictly follow these rules:
Focus on factual equivalence, not exact string matching.
Ignore differences in wording, phrasing, or formatting (e.g., "USA" vs "United States", "2" vs "two").
For numeric answers, accept equivalent representations (e.g., "1,000" = "1000", "50%" = "0.5").
For dates, accept equivalent representations (e.g., "05/01/2026" vs "May 1, 2026" vs "01.05.2026").
If the predicted answer contains the correct information but adds extra details, reply YES.
If the predicted answer is partially correct but misses key information present in the groundtruth, reply NO.

Reply with ONLY one word: YES or NO."""

JUDGE_USER_TEMPLATE = """Question: {question}
Gold answer: {gold_answer}
Candidate answer: {candidate}

Is the candidate equivalent to the gold answer? Reply YES or NO:"""


def strip_candidate_prefix(text: str) -> str:
    if text.lower().startswith("final answer:"):
        text = text[len("final answer:"):].strip()
    # strip trailing period for cleaner comparison
    return text.strip().rstrip(".")


def parse_yes_no(response_text: str) -> int | None:
    """Parse a YES/NO response. Returns 1, 0, or None on failure."""
    text = response_text.strip().upper()
    # Take just the first word/token
    first_word = text.split()[0] if text.split() else ""
    first_word = re.sub(r'[^A-Z]', '', first_word)
    if first_word == "YES":
        return 1
    elif first_word == "NO":
        return 0
    # Fallback: search anywhere
    if "YES" in text and "NO" not in text:
        return 1
    if "NO" in text and "YES" not in text:
        return 0
    return None


def vllm_request(messages: list, max_tokens: int = 8, temperature: float = 0.0) -> str:
    """Send a chat completion request to vLLM and return the response text."""
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")

    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def judge_single_candidate(question: str, gold_answer: str, candidate: str) -> int:
    """Judge one candidate. Returns 1 (correct), 0 (incorrect), or -1 (failed)."""
    clean_candidate = strip_candidate_prefix(candidate)
    user_content = JUDGE_USER_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        candidate=clean_candidate,
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(MAX_RETRIES):
        try:
            text = vllm_request(messages)
            result = parse_yes_no(text)
            if result is not None:
                return result
            else:
                if attempt == MAX_RETRIES - 1:
                    print(f"[DEBUG] Parse failed, response: {text[:100]}")
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"[ERROR] {e}")
    return -1


def judge_one_sample(sample: dict) -> dict:
    """Judge all candidates for a single sample (sequential per-candidate)."""
    judgements = []
    for cand in sample["candidates"]:
        j = judge_single_candidate(sample["question"], sample["answer"], cand)
        judgements.append(j)
    return {**sample, "judgements": judgements}


def save_results(path: str, records: list):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def main(input_path: str, output_path: str, max_workers: int):
    # Load input
    with open(input_path) as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} samples from {input_path}")

    # Resume support
    done_ids = set()
    results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                rec = json.loads(line)
                done_ids.add(rec["id"])
                results.append(rec)
        print(f"Resuming: {len(done_ids)} already judged")

    remaining = [s for s in samples if s["id"] not in done_ids]
    print(f"Remaining to judge: {len(remaining)}")

    if not remaining:
        print("Nothing to do!")
        return

    write_lock = Lock()
    counter = {"n": 0}

    def process_and_track(sample):
        result = judge_one_sample(sample)
        with write_lock:
            results.append(result)
            counter["n"] += 1
            if counter["n"] % SAVE_EVERY == 0:
                save_results(output_path, results)
                print(f"  Saved {len(results)} total")
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_and_track, s): s for s in remaining}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            try:
                future.result()
            except Exception as e:
                sid = futures[future].get("id", "?")
                print(f"[FATAL] Sample {sid}: {e}")
            if done_count % 2000 == 0:
                print(f"Progress: {done_count}/{len(remaining)}")

    # Final save
    save_results(output_path, results)
    print(f"\nDone! Total judged: {len(results)}")

    # Stats
    all_correct = sum(1 for d in results if all(j == 1 for j in d["judgements"]))
    all_wrong = sum(1 for d in results if all(j == 0 for j in d["judgements"]))
    unresolved = sum(1 for d in results if -1 in d["judgements"])
    mixed = len(results) - all_correct - all_wrong - unresolved
    print(f"All correct: {all_correct}, All wrong: {all_wrong}, Mixed: {mixed}, Unresolved: {unresolved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge HotpotQA candidates")
    parser.add_argument("--input", default="/home/a.anokhin/Judge/hotpot_candidates_new_prompt.jsonl")
    parser.add_argument("--output", default="/home/a.anokhin/Judge/hotpot_judged_new_prompt.jsonl")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()
    MODEL_NAME = args.model
    BASE_URL = args.base_url
    main(args.input, args.output, args.workers)
