"""
Test gold-shift reward: compare gold chunks vs random chunks.

For 5 HotpotQA samples, checks that reward with gold (supporting fact)
chunks exceeds reward with random non-gold chunks.

Requires a running vLLM server on localhost:8000.
"""

import json
import sys
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl.feedback.gold_shift_feedback import GoldShiftFeedback

# Path to raw HotpotQA train data
DATA_FILE = "/home/a.anokhin/Judge/datasets/data_sources/hotpotqa/hotpot_train_v1.1.json"
N_SAMPLES = 5


def load_hotpotqa_samples(n: int):
    """Load n HotpotQA samples that have >=2 supporting facts and >=6 chunks."""
    with open(DATA_FILE) as f:
        all_data = json.load(f)

    samples = []
    for raw in all_data:
        chunks = []
        sf_idx = []
        sp_titles = set(sup[0] for sup in raw.get("supporting_facts", []))
        for idx, (title, sents) in enumerate(raw.get("context", [])):
            if title in sp_titles:
                sf_idx.append(idx)
            chunks.append(title + " " + " ".join(sents))
        if len(sf_idx) >= 2 and len(chunks) >= 6:
            samples.append({
                "id": raw.get("_id", str(len(samples))),
                "question": raw["question"],
                "answer": raw["answer"],
                "chunks": chunks,
                "sf_idx": sf_idx,
            })
        if len(samples) >= n:
            break
    return samples


def main():
    random.seed(42)

    samples = load_hotpotqa_samples(N_SAMPLES)
    print(f"Selected {len(samples)} samples with >= 2 gold chunks\n")

    fb = GoldShiftFeedback(
        api_base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B",
        only_at_final=True,
        never_terminate=True,
        reward_scaling=1.0,  # no scaling for test visibility
    )

    results = []

    for i, sample in enumerate(samples):
        chunks = sample["chunks"]
        sf_idx = sample["sf_idx"]
        non_sf_idx = [j for j in range(len(chunks)) if j not in sf_idx]

        # Prepare obs/info for reset (needs question in obs, answer in info)
        obs_reset = {"question": sample["question"]}
        info_reset = {"answer": sample["answer"]}

        # --- Gold chunks ---
        gold_chunks = [chunks[j] for j in sf_idx[:2]]
        obs_gold = {
            "question": sample["question"],
            "sample_id": sample["id"],
            "pred_idx": sf_idx[:2],
            "pred_chunks": gold_chunks,
        }

        fb.reset(obs_reset, info_reset)
        r_gold = fb.reward(obs_gold, info_reset, is_final=True)

        # --- Random (non-gold) chunks ---
        rand_idx = random.sample(non_sf_idx, min(2, len(non_sf_idx)))
        rand_chunks = [chunks[j] for j in rand_idx]
        obs_rand = {
            "question": sample["question"],
            "sample_id": sample["id"],
            "pred_idx": rand_idx,
            "pred_chunks": rand_chunks,
        }

        fb.reset(obs_reset, info_reset)
        r_rand = fb.reward(obs_rand, info_reset, is_final=True)

        diff = r_gold - r_rand
        results.append((r_gold, r_rand, diff))

        print(f"Sample {i}: {sample['id']}")
        print(f"  Q: {sample['question'][:80]}")
        print(f"  A: {sample['answer']}")
        print(f"  Gold chunks idx:   {sf_idx[:2]}")
        print(f"  Random chunks idx: {rand_idx}")
        print(f"  β (no context):  {fb.beta:+.4f}")
        print(f"  Reward (gold):   {r_gold:+.4f}")
        print(f"  Reward (random): {r_rand:+.4f}")
        print(f"  Δ (gold-random): {diff:+.4f}  {'✓ gold > random' if diff > 0 else '✗ random >= gold'}")
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    gold_wins = sum(1 for _, _, d in results if d > 0)
    avg_gold = sum(r[0] for r in results) / len(results)
    avg_rand = sum(r[1] for r in results) / len(results)
    avg_diff = sum(r[2] for r in results) / len(results)
    print(f"  Gold wins: {gold_wins}/{len(results)}")
    print(f"  Avg reward (gold):   {avg_gold:+.4f}")
    print(f"  Avg reward (random): {avg_rand:+.4f}")
    print(f"  Avg Δ:               {avg_diff:+.4f}")


if __name__ == "__main__":
    main()
