#!/usr/bin/env python3
"""Derive evaluation variants from an existing retrieval JSONL.

Three of the Phase-0 baselines need no GPU and no second retrieval pass,
because everything they require is already stored per record:

``--context truncate``
    Keep the first N selected chunks. Valid only for runs whose selection is
    prefix-consistent, i.e. ``--reranker none --mode fixed``: the candidate
    pool is fixed and the ranking does not depend on the state, so the
    2- and 4-step baselines are literally prefixes of the 6-step one. Q-RAG
    runs are *not* prefix-consistent -- each hop re-ranks against an updated
    state -- so truncating one is refused.

``--context none``
    Empty context, i.e. the closed-book reader baseline.

``--context oracle``
    Gold supporting sentences as context, i.e. the reader upper bound. These
    are HotpotQA sentences rather than Wiki-18 chunks, so this bounds
    "perfect retrieval over HotpotQA", not over Wiki-18.

    The gold sentences must come from ``--oracle-source
    hotpot_dev_distractor_v1.json``, *not* from the ``sf_texts`` field of the
    retrieval run. The ``context`` of ``hotpot_dev_fullwiki_v1.json`` is the
    top-10 output of the original HotpotQA full-wiki retriever, not the gold
    paragraphs: 5316 of 7405 dev examples are missing at least one gold title
    there, so ``sf_texts`` is silently partial (present for 6162 records, and
    complete for far fewer). The distractor dev file carries the same ids in
    the same order and contains every gold paragraph.

``--context keep``
    Leave the context alone and only recompute the title metrics. Used to
    rescore the published Q-RAG runs with normalized title matching.

Gold-title metrics are always recomputed from the resulting context with
:func:`fullwiki_qrag.title_metrics`, so every variant is scored by exactly
the same code as a fresh retrieval run.

Examples:

    # 2- and 4-step first-stage baselines out of the 6-step run.
    python src/build_eval_variants.py --context truncate --steps 2 \
      --input runs/fullwiki_gte_only_steps6.jsonl \
      --output runs/fullwiki_gte_only_steps2.jsonl

    # Reader bounds.
    python src/build_eval_variants.py --context none \
      --input runs/fullwiki_gte_only_steps6.jsonl \
      --output runs/fullwiki_no_retrieval.jsonl

    # Rescore a published Q-RAG run with normalized titles.
    python src/build_eval_variants.py --context keep \
      --input runs/fullwiki_qrag_fixed_steps6.jsonl \
      --output runs/fullwiki_qrag_fixed_steps6_rescored.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from fullwiki_qrag import (
    iter_jsonl,
    load_input_samples,
    sample_id,
    supporting_fact_texts,
    title_metrics,
    wiki_title,
)


LOG = logging.getLogger("build-eval-variants")

CONTEXTS = ("keep", "truncate", "none", "oracle")
SELECTION_KEYS = ("pred_idx", "pred_texts", "q_values", "retrieval_hops")


def truncate_record(record: dict[str, Any], steps: int) -> dict[str, Any]:
    reranker = record.get("reranker")
    if reranker != "none":
        raise ValueError(
            "Refusing to truncate a run with reranker="
            f"{reranker!r}: only the first-stage baseline is prefix-consistent. "
            "Q-RAG re-ranks against an updated state at every hop, so its "
            "2-step selection is not a prefix of its 6-step selection."
        )
    if record.get("retrieval_mode") != "fixed":
        raise ValueError(
            "Refusing to truncate a refresh-mode run: the candidate pool "
            "changes between hops, so prefixes are not comparable."
        )
    available = len(record["pred_texts"])
    if steps > available:
        raise ValueError(f"Cannot truncate to {steps} steps: record has {available}")
    result = dict(record)
    for key in SELECTION_KEYS:
        if key in result and isinstance(result[key], list):
            result[key] = result[key][:steps]
    return result


def load_gold_sentences(path: Path) -> dict[str, list[str]]:
    """Map sample id to its gold supporting sentences, "Title sentence"."""
    gold = {}
    for index, sample in enumerate(load_input_samples(path)):
        texts = supporting_fact_texts(sample)
        if not texts:
            raise ValueError(
                f"Sample {sample_id(sample, index)!r} in {path} has no recoverable "
                "supporting facts; this file cannot serve as the oracle source"
            )
        gold[sample_id(sample, index)] = texts
    return gold


def oracle_record(
    record: dict[str, Any],
    gold_sentences: dict[str, list[str]],
) -> dict[str, Any]:
    key = str(record.get("id"))
    if key not in gold_sentences:
        raise ValueError(f"Oracle source has no sample with id {key!r}")
    result = dict(record)
    result["pred_texts"] = list(gold_sentences[key])
    result["pred_idx"] = []
    result["q_values"] = []
    result["retrieval_hops"] = []
    return result


def empty_record(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    for key in SELECTION_KEYS:
        result[key] = []
    return result


def retrieved_titles_for(record: dict[str, Any], context: str) -> list[str]:
    if context == "oracle":
        # sf_texts are "Title sentence", not '"Title"\npassage', so they are
        # not parseable by wiki_title. The oracle context is gold by
        # construction, and scoring it against gold_titles keeps the metric
        # definition identical rather than special-casing the number.
        return list(record.get("gold_titles", []))
    return [wiki_title(text) for text in record["pred_texts"]]


def build(
    records: Sequence[dict[str, Any]],
    context: str,
    steps: int | None,
    gold_sentences: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    results = []
    for record in records:
        if context == "truncate":
            assert steps is not None
            result = truncate_record(record, steps)
        elif context == "none":
            result = empty_record(record)
        elif context == "oracle":
            assert gold_sentences is not None
            result = oracle_record(record, gold_sentences)
        else:
            result = dict(record)
        result["retrieved_titles"] = retrieved_titles_for(result, context)
        result.update(
            title_metrics(result.get("gold_titles", []), result["retrieved_titles"])
        )
        result["eval_variant"] = (
            f"{context}-{steps}" if context == "truncate" else context
        )
        results.append(result)
    return results


def summarize(results: Sequence[dict[str, Any]]) -> dict[str, float]:
    return {
        "samples": len(results),
        "context_chunks_min": min(len(r["pred_texts"]) for r in results),
        "context_chunks_max": max(len(r["pred_texts"]) for r in results),
        "title_recall": float(np.mean([r["title_recall"] for r in results])),
        "title_em": float(np.mean([r["title_em"] for r in results])),
        "title_recall_exact": float(
            np.mean([r["title_recall_exact"] for r in results])
        ),
        "title_em_exact": float(np.mean([r["title_em_exact"] for r in results])),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", choices=CONTEXTS, required=True)
    parser.add_argument(
        "--steps", type=int, default=None, help="required for --context truncate"
    )
    parser.add_argument(
        "--oracle-source",
        type=Path,
        default=None,
        help=(
            "required for --context oracle: hotpot_dev_distractor_v1.json, "
            "the only dev file whose context holds every gold paragraph"
        ),
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    args = parser.parse_args(argv)
    if args.context == "truncate" and (args.steps is None or args.steps <= 0):
        parser.error("--context truncate requires a positive --steps")
    if args.context != "truncate" and args.steps is not None:
        parser.error("--steps is only meaningful with --context truncate")
    if args.context == "oracle" and args.oracle_source is None:
        parser.error("--context oracle requires --oracle-source")
    if args.context != "oracle" and args.oracle_source is not None:
        parser.error("--oracle-source is only meaningful with --context oracle")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if destination == source:
        raise ValueError("Refusing to overwrite the input file in place")
    records = list(iter_jsonl(source))
    if not records:
        raise ValueError(f"No records in {source}")
    gold_sentences = (
        load_gold_sentences(args.oracle_source.expanduser().resolve())
        if args.context == "oracle"
        else None
    )
    results = build(records, args.context, args.steps, gold_sentences)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
    LOG.info("Wrote %s", destination)
    LOG.info("%s", json.dumps(summarize(results), indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOG.error("Interrupted")
        raise SystemExit(130)
    except Exception as error:
        LOG.error("Failed: %s", error)
        raise
