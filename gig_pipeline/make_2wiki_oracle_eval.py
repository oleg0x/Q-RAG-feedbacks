"""
Build an oracle retriever-eval JSONL for 2WikiMultihopQA.

The output mirrors eval_retriever.py logs, but uses gold supporting-fact chunks
as the predicted chunks:

    pred_idx = sf_idx
    pred_texts = sf_texts

By default this uses the 2Wiki dev split because Q-RAG configs use dev as the
test_dataset, and the local test.json does not contain gold supporting_facts.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any


DATA_ROOT = Path("/home/a.anokhin/Judge/datasets/data_sources")
TWOWIKI_DIR = DATA_ROOT / "2WikiMultiHopQA/data_ids_april7"
DEFAULT_OUTPUT = Path("Q-RAG-pqn/runs/oracle_2wiki/eval_seed42.jsonl")


def calc_fact_f1_em(predicted_support_idxs: list[int], gt_support_idxs: list[int]) -> tuple[float, float]:
    pred_sf = set(map(int, predicted_support_idxs))
    gt_sf = set(map(int, gt_support_idxs))

    tp = len(pred_sf & gt_sf)
    fp = len(pred_sf - gt_sf)
    fn = len(gt_sf - pred_sf)

    prec = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if gt_sf.issubset(pred_sf) else 0.0

    if not pred_sf and not gt_sf:
        return 1.0, 1.0
    return f1, em


def load_2wiki_samples(
    data_dir: Path,
    split: str,
    seed: int,
    skip_bridge_comparison: bool,
    limit: int,
) -> list[dict[str, Any]]:
    path = data_dir / f"{split}.json"
    with path.open(encoding="utf-8") as f:
        samples = json.load(f)

    if skip_bridge_comparison:
        samples = [sample for sample in samples if sample.get("type") != "bridge_comparison"]

    rng = random.Random(seed)
    rng.shuffle(samples)

    if limit > 0:
        samples = samples[:limit]
    return samples


def make_chunks_and_sf_idx(sample: dict[str, Any]) -> tuple[list[str], list[int]]:
    support_titles = {fact[0] for fact in sample.get("supporting_facts", [])}
    chunks: list[str] = []
    sf_idx: list[int] = []

    for idx, (title, sentences) in enumerate(sample["context"]):
        if title in support_titles:
            sf_idx.append(idx)
        chunks.append(title + " " + " ".join(sentences))

    return chunks, sf_idx


def whitespace_text_len(question: str, chunks: list[str]) -> int:
    return len(question.split()) + sum(len(chunk.split()) for chunk in chunks)


def build_oracle_records(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    missing_sf = 0

    for sample in samples:
        if not sample.get("supporting_facts"):
            missing_sf += 1
            continue

        chunks, sf_idx = make_chunks_and_sf_idx(sample)
        if not sf_idx:
            missing_sf += 1
            continue

        sf_texts = [chunks[idx] for idx in sf_idx]
        f1, em = calc_fact_f1_em(sf_idx, sf_idx)
        records.append({
            "id": sample["_id"],
            "question": sample["question"],
            "answer": sample["answer"],
            "sf_idx": [int(idx) for idx in sf_idx],
            "pred_idx": [int(idx) for idx in sf_idx],
            "q_values": [1.0 for _ in sf_idx],
            "sf_texts": sf_texts,
            "pred_texts": list(sf_texts),
            "return": 1.0,
            "text_len": whitespace_text_len(sample["question"], chunks),
            "f1": f1,
            "em": em,
        })

    return records, missing_sf


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build oracle eval_seed42-style JSONL for 2Wiki")
    parser.add_argument("--data-dir", type=Path, default=TWOWIKI_DIR)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument(
        "--include-bridge-comparison",
        action="store_true",
        help="Include 2Wiki bridge_comparison samples instead of matching Q-RAG configs.",
    )
    args = parser.parse_args()

    samples = load_2wiki_samples(
        data_dir=args.data_dir,
        split=args.split,
        seed=args.seed,
        skip_bridge_comparison=not args.include_bridge_comparison,
        limit=args.limit,
    )
    records, missing_sf = build_oracle_records(samples)
    if not records:
        raise ValueError(
            f"No oracle records were built from {args.data_dir / (args.split + '.json')}. "
            "This usually means the split has no gold supporting_facts."
        )

    write_jsonl(args.output, records)

    print(f"Loaded samples:  {len(samples)}")
    print(f"Missing SF:      {missing_sf}")
    print(f"Written records: {len(records)}")
    print(f"Output:          {args.output}")


if __name__ == "__main__":
    main()
