"""
Sanity check: load a few samples, instantiate CandidateBetaFeedback,
and verify reward computation works end-to-end with the running vLLM server.
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.feedback.candidate_beta_feedback import CandidateBetaFeedback


def main():
    # Load a few samples
    data_path = "/home/a.anokhin/Judge/hotpot_candidate_train.jsonl"
    samples = []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= 2:
                break
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} test samples")

    # Instantiate feedback model
    fb = CandidateBetaFeedback(
        api_base_url="http://localhost:8000",
        model_name="Qwen/Qwen3-4B",
        max_concurrent=9,
        only_at_final=True,
        never_terminate=True,
        reward_scaling=1.0,
    )

    for idx, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Sample {idx}: id={sample['_id']}")
        print(f"  Question: {sample['question'][:80]}...")
        print(f"  Answer: {sample['answer']}")
        print(f"  Candidates: {len(sample['candidates'])}")
        print(f"  Judgements: {sample['judgements']}")

        # Simulate env: pick 2 chunks as "retrieved"
        chunks = []
        for title, sents in sample["context"][:2]:
            chunks.append(title + " " + " ".join(sents))

        obs = {
            "question": sample["question"],
            "sample_id": sample["_id"],
            "pred_idx": [0, 1],
            "pred_chunks": chunks,
        }
        info = {
            "sf_idx": [0, 1],
            "sf_chunks": chunks,
            "answer": sample["answer"],
            "candidates": sample["candidates"],
            "judgements": sample["judgements"],
            "betas": sample["betas"],
        }

        # Reset
        fb.reset(obs, info)

        # Compute reward (not final → should be 0)
        r_mid = fb.reward(obs, info, is_final=False)
        print(f"  Reward (mid-episode): {r_mid}")
        assert r_mid == 0.0, f"Expected 0 for non-final step, got {r_mid}"

        # Compute reward (final → should be non-zero)
        r_final = fb.reward(obs, info, is_final=True)
        print(f"  Reward (final): {r_final:.6f}")

        import math
        assert not math.isnan(r_final), "Reward is NaN!"
        assert not math.isinf(r_final), "Reward is Inf!"
        print(f"  ✓ Reward is finite: {r_final:.6f}")

    # Test copy
    fb2 = fb.copy()
    print(f"\n✓ copy() works: {type(fb2).__name__}")

    print("\n" + "="*60)
    print("All sanity checks passed!")


if __name__ == "__main__":
    main()
