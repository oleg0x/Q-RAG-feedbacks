"""
Merge candidate-beta JSONL with raw HotpotQA / MuSiQue context and write
the training file consumed by CandidateDatasetAdapter.

Filters out samples where ALL candidates are correct (G^- empty).

Usage:
    python prepare_dataset.py \\
        --dataset hotpot \\
        --final  output/hotpotqa/hotpot_final_new_prompt.jsonl \\
        --output output/hotpotqa/hotpot_candidate_train.jsonl

    python prepare_dataset.py \\
        --dataset musique \\
        --final  output/musique/musique_final.jsonl \\
        --output output/musique/musique_candidate_train.jsonl
"""

import argparse
import json

import gig_common as G

DATA_ROOT = "/home/a.anokhin/Judge/datasets/data_sources"
HOTPOT_RAW = [
    f"{DATA_ROOT}/hotpotqa/hotpot_train_v1.1.json",
    f"{DATA_ROOT}/hotpotqa/hotpot_dev_distractor_v1.json",
]
MUSIQUE_RAW = [
    f"{DATA_ROOT}/musique/musique_ans_v1.0_train.jsonl",
    f"{DATA_ROOT}/musique/musique_ans_v1.0_dev.jsonl",
]


def load_hotpot_originals() -> dict:
    out = {}
    for path in HOTPOT_RAW:
        with open(path) as f:
            for s in json.load(f):
                out[s["_id"]] = s
    return out


def load_musique_originals() -> dict:
    out = {}
    for path in MUSIQUE_RAW:
        for s in G.read_jsonl(path):
            out[s["id"]] = s
    return out


def hotpot_context(orig: dict) -> tuple[list, list]:
    return orig["context"], orig["supporting_facts"]


def musique_context(orig: dict) -> tuple[list, list]:
    """Convert MuSiQue paragraphs to HotpotQA-style context + supporting_facts."""
    context = []
    supporting_facts = []
    for para in orig["paragraphs"]:
        title = para["title"]
        sents = [s.strip() for s in para["paragraph_text"].split(". ") if s.strip()]
        context.append([title, sents])
        if para.get("is_supporting"):
            supporting_facts.append([title, 0])
    return context, supporting_facts


CONFIGS = {
    "hotpot":  (load_hotpot_originals,  hotpot_context),
    "musique": (load_musique_originals, musique_context),
}


def main(dataset: str, final_path: str, output_path: str) -> None:
    load_originals, build_context = CONFIGS[dataset]

    print(f"Loading raw {dataset} samples...")
    id_to_original = load_originals()
    print(f"  total originals: {len(id_to_original)}")

    print(f"\nMerging candidates from {final_path}...")
    records = G.read_jsonl(final_path)

    merged = []
    all_correct = 0
    missing = 0
    for rec in records:
        if all(j == 1 for j in rec["judgements"]):
            all_correct += 1
            continue

        orig = id_to_original.get(rec["id"])
        if orig is None:
            missing += 1
            continue

        context, supporting_facts = build_context(orig)
        merged.append({
            "_id": rec["id"],
            "question": rec["question"],
            "answer": rec["answer"],
            "context": context,
            "supporting_facts": supporting_facts,
            "level": orig.get("level", ""),
            "type": orig.get("type", ""),
            "candidates": rec["candidates"],
            "judgements": rec["judgements"],
            "betas": rec["betas"],
        })

    G.write_jsonl_atomic(output_path, merged)

    print(f"\nDone!")
    print(f"  Total input:  {len(records)}")
    print(f"  All-correct:  {all_correct}")
    print(f"  Missing orig: {missing}")
    print(f"  Written:      {len(merged)}")
    print(f"  Output:       {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge candidate-beta JSONL with raw QA data")
    parser.add_argument("--dataset", choices=list(CONFIGS), required=True)
    parser.add_argument("--final", required=True,
                        help="candidate-beta jsonl (output of compute_beta.py)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.dataset, args.final, args.output)
