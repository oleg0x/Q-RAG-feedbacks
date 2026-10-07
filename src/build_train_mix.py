#!/usr/bin/env python3
"""Raw NQ + HotpotQA mix for training: manifest, holdout and episode weights.

The input is `train.parquet` of the `PeterJinGo/nq_hotpotqa_train` dataset
(the one Search-R1 trains on), stored in `train_data/raw/`. The script
deliberately neither filters nor deduplicates: training uses the full set,
the same as Search-R1, so that differences from it cannot be attributed to
data preparation. It produces three artifacts.

``manifest.json``: row count, per-``data_source`` breakdown and shares, and
the sha256 of both parquet files, so that "what was this trained on" can be
answered without recounting 170k rows.

``holdout.json``: ``--holdout-per-source`` examples from each half of the
mix, with a fixed seed. It is the same for all training arms: evaluation must
cover the same mix as training, with separate curves, and arms can only be
compared with each other on a shared set.

``weights/a1.jsonl``: the weight of every example, 1 everywhere and 0 on the
holdout. The environment draws episodes through this file
(``envs/search_env.py``), so excluding the holdout from training and future
filtered arms A2/A3 share one mechanism and one format.

The example key is ``"{data_source}:{id}"`` rather than the bare ``id``: both
halves of the mix are numbered from zero, and all 79,168 NQ identifiers
collide with HotpotQA identifiers.

    python src/build_train_mix.py --check   # only recompute and check the manifest
    python src/build_train_mix.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

import pandas as pd

import runlib


LOG = logging.getLogger("build-train-mix")

# train_data/ lives at the repository root, this module in src/.
DEFAULT_ROOT = runlib.REPO / "train_data"
# sha256 of the seven-benchmark Search-R1 `test.parquet`: the inputs in
# `runs/shared/searchr1/` are cut from it, and a mismatch would mean that the
# seven published rows were computed from a different file.
SEARCHR1_TEST_SHA256 = (
    "30aa887b6d47e06e8c0f6f5307c88fe4e13461ac25a20ec0a5433ad7a4fe25dc"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def example_key(data_source: str, sample_id: str) -> str:
    return f"{data_source}:{sample_id}"


def load_mix(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path, columns=["id", "question", "golden_answers", "data_source"]
    )
    frame["key"] = [
        example_key(source, sample_id)
        for source, sample_id in zip(frame["data_source"], frame["id"])
    ]
    if frame["key"].nunique() != len(frame):
        raise ValueError(
            f"{path}: example keys are not unique, so holdout and weights cannot "
            "address them"
        )
    return frame


def describe(frame: pd.DataFrame) -> dict[str, Any]:
    """Mix breakdown: number of examples and answer aliases in each half."""
    counts = frame["data_source"].value_counts()
    variants = frame["golden_answers"].apply(len)
    sources = {}
    for source in sorted(counts.index):
        subset = variants[frame["data_source"] == source]
        sources[source] = {
            "rows": int(counts[source]),
            "share": float(counts[source] / len(frame)),
            "answer_variants_mean": float(subset.mean()),
            "answer_variants_max": int(subset.max()),
            "multi_answer_share": float((subset > 1).mean()),
        }
    return {
        "rows": int(len(frame)),
        "answer_variants_mean": float(variants.mean()),
        "answer_variants_max": int(variants.max()),
        "multi_answer_share": float((variants > 1).mean()),
        "sources": sources,
    }


def pick_holdout(frame: pd.DataFrame, per_source: int, seed: int) -> list[str]:
    """``per_source`` keys from each half of the mix, with a fixed seed.

    Sampling runs over sorted keys rather than parquet row order: the file
    order is an upstream choice, and depending on it would yield a different
    holdout whenever the dataset is re-saved.
    """
    chosen: list[str] = []
    for source in sorted(frame["data_source"].unique()):
        keys = sorted(frame.loc[frame["data_source"] == source, "key"])
        if per_source > len(keys):
            raise ValueError(
                f"{source}: requested {per_source} examples, only {len(keys)} exist"
            )
        rng = random.Random(f"{seed}:{source}")
        chosen.extend(sorted(rng.sample(keys, per_source)))
    return chosen


def write_weights(path: Path, frame: pd.DataFrame, holdout: set[str]) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    drawable = 0
    with path.open("w", encoding="utf-8") as sink:
        for key in frame["key"]:
            weight = 0.0 if key in holdout else 1.0
            drawable += weight > 0
            sink.write(json.dumps({"key": key, "w": weight}) + "\n")
    return {"rows": int(len(frame)), "drawable": int(drawable)}


def build(root: Path, per_source: int, seed: int, check: bool) -> dict[str, Any]:
    raw = root / "raw"
    train_file = raw / "train.parquet"
    test_file = raw / "test.parquet"
    for path in (train_file, test_file):
        if not path.is_file():
            raise SystemExit(f"{path} is missing: download the dataset first")

    test_sha = sha256_file(test_file)
    if test_sha != SEARCHR1_TEST_SHA256:
        raise SystemExit(
            f"{test_file}: sha256 {test_sha}, expected "
            f"{SEARCHR1_TEST_SHA256}. The seven Search-R1 rows in runs/shared/ "
            "were computed from a different file; stop and investigate."
        )

    frame = load_mix(train_file)
    holdout = pick_holdout(frame, per_source, seed)
    manifest: dict[str, Any] = {
        "dataset": "PeterJinGo/nq_hotpotqa_train",
        "revision": "b7d80abfee334a7a91cb377544f09180d58b34f6",
        "created_utc": runlib.utc_now(),
        "files": {
            "train.parquet": {
                "sha256": sha256_file(train_file),
                "bytes": train_file.stat().st_size,
            },
            "test.parquet": {"sha256": test_sha, "bytes": test_file.stat().st_size},
        },
        "test_parquet_matches_searchr1": True,
        "train": describe(frame),
        "holdout": {
            "seed": seed,
            "per_source": per_source,
            "size": len(holdout),
        },
    }

    if check:
        stored = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        stored_holdout = json.loads(
            (root / "holdout.json").read_text(encoding="utf-8")
        )["ids"]
        same = (
            stored["train"] == manifest["train"]
            and stored["files"] == manifest["files"]
            and stored_holdout == holdout
        )
        LOG.info("Manifest and holdout check: %s", "match" if same else "MISMATCH")
        if not same:
            raise SystemExit("Artifacts disagree with the recomputation")
        return manifest

    (root / "holdout.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "per_source": per_source,
                "source": "train_data/raw/train.parquet",
                "key_format": "{data_source}:{id}",
                "ids": holdout,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    stats = write_weights(root / "weights" / "a1.jsonl", frame, set(holdout))
    manifest["weights"] = {"a1.jsonl": stats}
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    LOG.info(
        "Mix: %d rows, holdout %d, drawable %d",
        manifest["train"]["rows"],
        len(holdout),
        stats["drawable"],
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--holdout-per-source", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check",
        action="store_true",
        help="only recompute and compare with the existing artifacts",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build(args.root, args.holdout_per_source, args.seed, args.check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
