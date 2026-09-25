"""
LLM-as-Judge evaluation script.

Reads the eval_llm_openqa.py output JSON, compares full_model_output vs
ground_truth using an LLM judge via vLLM (Qwen3-32B or Qwen3-4B, set by the
MODEL_NAME constant below), computes accuracy, and writes output with an added
LLM_AS_JUDGE field.

Usage:
    python llm_as_judge_eval.py [--input FILE] [--output FILE] [--workers N]
"""

import argparse
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ─── defaults ───
VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-4B"
MAX_WORKERS = 64
SAVE_EVERY = 500
MAX_RETRIES = 3

DEFAULT_INPUT = "/home/a.anokhin/Judge/Q-RAG-pqn/runs/May12_22-04-39_QRAG_musique/llm_eval_max_steps=6.json"
DEFAULT_OUTPUT = "/home/a.anokhin/Judge/Q-RAG-pqn/runs/May12_22-04-39_QRAG_musique/llm_eval_with_llm_judge_32b_6.json"

# JUDGE_SYSTEM_PROMPT = """You are a strict QA answer-equivalence judge.

# You will receive:
# - a question
# - a gold (reference) answer
# - a candidate answer

# Your task is to decide whether the candidate should receive credit as semantically equivalent to the gold answer.

# Judging rules:
# 1. Ignore superficial differences:
#    - capitalization
#    - punctuation
#    - articles
#    - surrounding boilerplate such as "Final answer:"
#    - minor formatting differences

# 2. Accept clear aliases / alternative surface forms of the same entity:
#    - full name vs commonly used shorter name
#    - standard abbreviations
#    - alternate spellings/transliterations
#    - singular/plural when meaning is unchanged
#    Example: "Frank Iero" = "Frank Anthony Iero, Jr."

# 3. Accept answers where the candidate contains the gold answer with only harmless extra context.
#    Example: "Naples, Italy" = "Naples"

# 4. Do NOT accept answers that are broader, less specific, or only partially correct.
#    Example: "Italy" != "Naples"
#    Example: "animal" != "dog"

# 5. Do NOT accept near-misses, related entities, same category answers, or answers to a different slot in the question.
#    The candidate must identify the same person / place / thing / value as the gold answer.

# 6. For occupations, roles, and common nouns, accept only close paraphrases or clearly equivalent forms.
#    Example: "film director" = "director"
#    Example: "no" = "No."
#    But do NOT accept a merely broader category unless it would normally be considered the same answer.

# 7. For numbers, dates, and times, accept normalized equivalents only when they denote the same value.
#    Example: "94,000" = "94000"
#    Example: "November 9, 1984" = "1984-11-09"
#    Otherwise, mark NO.

# 8. If the candidate includes extra information that makes it wrong, mark NO.
#    Example: "Naples, France" != "Naples"

# 9. When uncertain, prefer NO.
#    Only answer YES if the candidate is clearly equivalent to the gold answer for this question.

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
    """Remove common prefixes like 'Final answer:' and trailing periods."""
    if text.lower().startswith("final answer:"):
        text = text[len("final answer:"):].strip()
    return text.strip().rstrip(".")


def parse_yes_no(response_text: str) -> int | None:
    """Parse a YES/NO response. Returns 1, 0, or None on failure."""
    text = response_text.strip().upper()
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

    req = urllib.request.Request(
        VLLM_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


def judge_single(question: str, gold_answer: str, candidate: str) -> int:
    """Judge one candidate. Returns 1 (YES), 0 (NO), or -1 (failed)."""
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


def main(input_path: str, output_path: str, max_workers: int):
    # Load input
    with open(input_path) as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} samples from {input_path}")

    # Check for resume: load already-judged items from output if it exists
    done_indices = set()
    results = [None] * len(samples)
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        for i, item in enumerate(existing):
            if "LLM_AS_JUDGE" in item:
                done_indices.add(i)
                results[i] = item
        print(f"Resuming: {len(done_indices)} already judged")

    remaining = [(i, s) for i, s in enumerate(samples) if i not in done_indices]
    print(f"Remaining to judge: {len(remaining)}")

    if not remaining:
        print("Nothing to do!")
        return

    write_lock = Lock()
    counter = {"n": 0}

    def process(idx_sample):
        idx, sample = idx_sample
        judgement = judge_single(
            sample["question"],
            sample["ground_truth"],
            sample["full_model_output"],
        )
        result = {**sample, "LLM_AS_JUDGE": judgement}
        with write_lock:
            results[idx] = result
            counter["n"] += 1
            if counter["n"] % SAVE_EVERY == 0:
                # Intermediate save: fill in unjudged items without the field
                to_save = [r if r is not None else samples[i] for i, r in enumerate(results)]
                with open(output_path + ".tmp", "w") as f:
                    json.dump(to_save, f, ensure_ascii=False, indent=2)
                os.replace(output_path + ".tmp", output_path)
                print(f"  Saved progress: {counter['n']}/{len(remaining)} done")
        return result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process, item): item for item in remaining}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            try:
                future.result()
            except Exception as e:
                idx, sample = futures[future]
                print(f"[FATAL] Sample {idx}: {e}")
            if done_count % 2000 == 0:
                print(f"Progress: {done_count}/{len(remaining)}")

    # Final save
    final = [r if r is not None else samples[i] for i, r in enumerate(results)]
    with open(output_path + ".tmp", "w") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    os.replace(output_path + ".tmp", output_path)

    # Compute accuracy
    judged = [r for r in results if r is not None and "LLM_AS_JUDGE" in r]
    total = len(judged)
    correct = sum(1 for r in judged if r["LLM_AS_JUDGE"] == 1)
    incorrect = sum(1 for r in judged if r["LLM_AS_JUDGE"] == 0)
    failed = sum(1 for r in judged if r["LLM_AS_JUDGE"] == -1)

    print(f"\n{'='*50}")
    print(f"LLM-as-Judge Results")
    print(f"{'='*50}")
    print(f"Total samples:  {total}")
    print(f"Correct (YES):  {correct}")
    print(f"Incorrect (NO): {incorrect}")
    print(f"Failed:         {failed}")
    if total > 0:
        accuracy = correct / total * 100
        print(f"Accuracy:       {accuracy:.2f}%")
    print(f"{'='*50}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Input JSON file path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output JSON file path (with LLM_AS_JUDGE field)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help="Number of parallel workers")
    args = parser.parse_args()
    main(args.input, args.output, args.workers)
