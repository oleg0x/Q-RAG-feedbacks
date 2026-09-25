"""
Generate K candidate answers from a frozen LLM under empty context.

Replaces the inline candidate-generation cells from hotpot_pqn.ipynb /
musique_pqn.ipynb. Produces one record per question:

    {"id", "question", "answer", "candidates": [str]*K}

Two modes:
    --mode sample  (default): sample K candidates from p_theta(.|x, c0)
                              using the shared QA prompt. Output schema
                              unchanged from previous versions.

    --mode guided:            two prompted calls per question — one asking
                              for K_CORRECT valid variants of the gold
                              answer, one for K_WRONG plausible-but-wrong
                              distractors. Output keeps the same
                              `candidates: [str]*(K_CORRECT+K_WRONG)`
                              schema (correct first, then wrong) plus a
                              `guided_ok` flag marking samples where the
                              numbered-list parse matched the requested
                              counts exactly.

Usage:
    python generate_candidates.py \
        --dataset hotpot \
        --output output/hotpotqa/hotpot_candidates.jsonl

    python generate_candidates.py \
        --dataset musique --mode guided \
        --output output/musique/musique_candidates_guided.jsonl

    python generate_candidates.py \
        --dataset 2wiki \
        --output output/2wiki/2wiki_candidates.jsonl

Supports resume: skips already-generated sample IDs on restart.
"""

import argparse
import json
import re

import gig_common as G

# ─── defaults ───
MODEL_NAME = "Qwen/Qwen3-4B"
K = 8
MAX_TOKENS_SAMPLE = 64
MAX_TOKENS_GUIDED = 256
TEMPERATURE_SAMPLE = 1.0
TEMPERATURE_GUIDED = 0.8
MAX_WORKERS = 64

# guided-mode split
K_CORRECT = 4
K_WRONG = 4

GUIDED_CORRECT_PROMPT = """You are given a question and its correct answer.

QUESTION:
{question}

CORRECT ANSWER:
{answer}

Generate exactly 4 different valid ways to express the same correct answer.
Use aliases, alternative spellings, paraphrases, abbreviations, or rewordings.
Each variant must still be a clearly correct answer to the question.

Return them as a numbered list, one per line:
1. ...
2. ...
3. ...
4. ...
No extra commentary."""

GUIDED_WRONG_PROMPT = """You are given a question and its correct answer.

QUESTION:
{question}

CORRECT ANSWER:
{answer}

Generate exactly 4 plausible but INCORRECT answers to the question.
Each wrong answer must:
- be of the same type as the correct answer (same kind of entity, person, place, number, date);
- look like a reasonable guess for someone who doesn't know the answer;
- NOT be equivalent to the correct answer in any form.

Return them as a numbered list, one per line:
1. ...
2. ...
3. ...
4. ...
No extra commentary."""

_NUMBERED_LINE = re.compile(r"^\s*\d+[\.\)]\s*(.+?)\s*$", re.MULTILINE)

DATA_ROOT = "/home/a.anokhin/Judge/datasets/data_sources"
HOTPOT_PATHS = {
    "train": [f"{DATA_ROOT}/hotpotqa/hotpot_train_v1.1.json"],
    "dev":   [f"{DATA_ROOT}/hotpotqa/hotpot_dev_distractor_v1.json"],
}
HOTPOT_PATHS["all"] = HOTPOT_PATHS["train"] + HOTPOT_PATHS["dev"]

MUSIQUE_PATHS = {
    "train": [f"{DATA_ROOT}/musique/musique_ans_v1.0_train.jsonl"],
    "dev":   [f"{DATA_ROOT}/musique/musique_ans_v1.0_dev.jsonl"],
}
MUSIQUE_PATHS["all"] = MUSIQUE_PATHS["train"] + MUSIQUE_PATHS["dev"]

TWOWIKI_PATHS = {
    "train": [f"{DATA_ROOT}/2WikiMultiHopQA/data_ids_april7/train.json"],
    "dev":   [f"{DATA_ROOT}/2WikiMultiHopQA/data_ids_april7/dev.json"],
}
TWOWIKI_PATHS["all"] = TWOWIKI_PATHS["train"] + TWOWIKI_PATHS["dev"]


def load_hotpot(split: str) -> list[dict]:
    out = []
    for path in HOTPOT_PATHS[split]:
        with open(path) as f:
            for s in json.load(f):
                out.append({"id": s["_id"], "question": s["question"], "answer": s["answer"]})
    return out


def load_musique(split: str) -> list[dict]:
    out = []
    for path in MUSIQUE_PATHS[split]:
        with open(path) as f:
            for line in f:
                s = json.loads(line)
                out.append({"id": s["id"], "question": s["question"], "answer": s["answer"]})
    return out


def load_2wiki(split: str) -> list[dict]:
    out = []
    for path in TWOWIKI_PATHS[split]:
        with open(path) as f:
            for sample in json.load(f):
                if sample.get("type") == "bridge_comparison":
                    continue
                out.append({
                    "id": sample["_id"],
                    "question": sample["question"],
                    "answer": sample["answer"],
                })
    return out


LOADERS = {"hotpot": load_hotpot, "musique": load_musique, "2wiki": load_2wiki}


def make_sample_process_fn(model: str, base_url: str, k: int, max_tokens: int, temperature: float):
    def process(sample: dict) -> dict:
        candidates = G.chat_completion(
            G.qa_messages(sample["question"]),
            model=model, base_url=base_url,
            max_tokens=max_tokens, temperature=temperature, n=k,
        )
        return {
            "id": sample["id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "candidates": candidates,
        }
    return process


def _parse_numbered_list(text: str) -> list[str]:
    return [m.group(1).strip() for m in _NUMBERED_LINE.finditer(text)]


def _fit_to_size(items: list[str], n: int, pad_with: str) -> tuple[list[str], bool]:
    """Return (items_of_length_n, exact_match). Truncates or pads as needed."""
    if len(items) == n:
        return items, True
    if len(items) > n:
        return items[:n], False
    return items + [pad_with] * (n - len(items)), False


def make_guided_process_fn(
    model: str,
    base_url: str,
    k_correct: int,
    k_wrong: int,
    max_tokens: int,
    temperature: float,
):
    def process(sample: dict) -> dict:
        q, a = sample["question"], sample["answer"]
        correct_resp = G.chat_completion(
            [{"role": "user", "content": GUIDED_CORRECT_PROMPT.format(question=q, answer=a)}],
            model=model, base_url=base_url,
            max_tokens=max_tokens, temperature=temperature, n=1,
        )[0]
        wrong_resp = G.chat_completion(
            [{"role": "user", "content": GUIDED_WRONG_PROMPT.format(question=q, answer=a)}],
            model=model, base_url=base_url,
            max_tokens=max_tokens, temperature=temperature, n=1,
        )[0]
        correct_items, correct_ok = _fit_to_size(_parse_numbered_list(correct_resp), k_correct, pad_with=a)
        wrong_items, wrong_ok = _fit_to_size(_parse_numbered_list(wrong_resp), k_wrong, pad_with="")
        return {
            "id": sample["id"],
            "question": q,
            "answer": a,
            "candidates": correct_items + wrong_items,
            "guided_ok": correct_ok and wrong_ok,
        }
    return process


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate K answer candidates per question")
    parser.add_argument("--dataset", choices=list(LOADERS), required=True)
    parser.add_argument("--split", choices=["train", "dev", "all"], default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["sample", "guided"], default="sample")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--base-url", default=G.VLLM_BASE_URL)
    parser.add_argument("--k", type=int, default=K, help="sample mode: n candidates")
    parser.add_argument("--k-correct", type=int, default=K_CORRECT, help="guided mode")
    parser.add_argument("--k-wrong", type=int, default=K_WRONG, help="guided mode")
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    if args.max_tokens is None:
        args.max_tokens = MAX_TOKENS_GUIDED if args.mode == "guided" else MAX_TOKENS_SAMPLE
    if args.temperature is None:
        args.temperature = TEMPERATURE_GUIDED if args.mode == "guided" else TEMPERATURE_SAMPLE

    samples = LOADERS[args.dataset](args.split)
    print(f"Loaded {len(samples)} {args.dataset}/{args.split} samples (mode={args.mode})")

    if args.mode == "sample":
        process_fn = make_sample_process_fn(
            args.model, args.base_url, args.k, args.max_tokens, args.temperature,
        )
    else:
        process_fn = make_guided_process_fn(
            args.model, args.base_url, args.k_correct, args.k_wrong,
            args.max_tokens, args.temperature,
        )

    def sanity(sample):
        result = process_fn(sample)
        print(f"Sanity: question='{sample['question'][:80]}...'")
        print(f"  K={len(result['candidates'])} candidates")
        for i, c in enumerate(result["candidates"]):
            print(f"    [{i}] {c[:80]!r}")
        if "guided_ok" in result:
            print(f"  guided_ok={result['guided_ok']}")
        print()

    G.run_with_resume(samples, args.output, process_fn, max_workers=args.workers, sanity_check=sanity)
