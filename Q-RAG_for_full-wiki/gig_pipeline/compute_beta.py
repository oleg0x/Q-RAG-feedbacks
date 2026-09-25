"""
Compute beta(g) = log p(g | question, no-context) for each candidate.

Reads <judged>.jsonl (with id, question, answer, candidates, judgements) and
writes <final>.jsonl with an additional "betas" field. By default, the gold
answer is appended as an extra candidate with judgement=1 and its own beta
(this replaces the old compute_gold_beta.py --mode score + --mode merge flow).

Usage:
    python compute_beta.py \\
        --input  output/hotpotqa/hotpot_judged_new_prompt.jsonl \\
        --output output/hotpotqa/hotpot_final_new_prompt.jsonl

    # Score generated candidates only, no gold append:
    python compute_beta.py --no-append-gold ...

Supports resume: skips already-processed sample IDs on restart.
"""

import argparse
import statistics

import gig_common as G

MODEL_NAME = "Qwen/Qwen3-4B"
MAX_WORKERS = 64


def make_process_fn(model: str, base_url: str, append_gold: bool):
    def process(sample: dict) -> dict:
        prompt_ids = G.tokenize_chat(sample["question"], model, base_url)

        # Normalize candidates so the strings written to JSONL are *exactly*
        # the strings beta scores — and reward will re-tokenize at runtime.
        candidates = [G.strip_candidate_prefix(c) for c in sample["candidates"]]
        judgements = list(sample["judgements"])
        if append_gold:
            candidates.append(G.strip_candidate_prefix(sample["answer"]))
            judgements.append(1)

        betas = []
        for cand in candidates:
            for attempt in range(G.DEFAULT_MAX_RETRIES):
                try:
                    betas.append(G.score_text(prompt_ids, cand, model, base_url))
                    break
                except Exception as e:
                    if attempt == G.DEFAULT_MAX_RETRIES - 1:
                        print(f"[ERROR] {sample['id']} '{cand[:40]}': {e}")
                        betas.append(float("-inf"))

        return {**sample, "candidates": candidates, "judgements": judgements, "betas": betas}

    return process


def print_stats(records: list[dict], append_gold: bool) -> None:
    all_betas = [b for r in records for b in r["betas"]]
    valid = [b for b in all_betas if b > float("-inf")]
    print(f"\nValid betas: {len(valid)}/{len(all_betas)}")
    if valid:
        print(f"  mean={statistics.mean(valid):.4f}  median={statistics.median(valid):.4f}  "
              f"min={min(valid):.4f}  max={max(valid):.4f}")
    if append_gold:
        n_correct = sum(1 for r in records if all(j == 1 for j in r["judgements"]))
        n_wrong = sum(1 for r in records if all(j == 0 for j in r["judgements"]))
        n_mixed = len(records) - n_correct - n_wrong
        print(f"Judgements: all_correct={n_correct}, all_wrong={n_wrong}, mixed={n_mixed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute beta log-probs for candidates")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--base-url", default=G.VLLM_BASE_URL)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--append-gold", dest="append_gold", action="store_true", default=True,
                        help="Append gold answer as K+1-th candidate with judgement=1 (default)")
    parser.add_argument("--no-append-gold", dest="append_gold", action="store_false")
    args = parser.parse_args()

    samples = G.read_jsonl(args.input)
    process_fn = make_process_fn(args.model, args.base_url, args.append_gold)

    def sanity(sample: dict) -> None:
        prompt_ids = G.tokenize_chat(sample["question"], args.model, args.base_url)
        beta = G.score_text(prompt_ids, sample["candidates"][0], args.model, args.base_url)
        print(f"Sanity: q='{sample['question'][:60]}...' cand='{sample['candidates'][0][:40]}' "
              f"beta={beta:.4f}\n")

    results = G.run_with_resume(samples, args.output, process_fn,
                                max_workers=args.workers, sanity_check=sanity)
    print_stats(results, args.append_gold)
