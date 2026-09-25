#!/usr/bin/env python3
"""Measure how much the LLM-judge metric loses to strict string matching.

``answer_judge_llms.py`` scores a judgement as correct only when the reply is
byte-for-byte ``CORRECT`` after upper-casing, so a reply of ``CORRECT.`` is
silently counted as wrong. How often that happens is unknown, because the
judge's raw reply is never stored -- only the resulting float.

This script re-runs the judge over predictions that are already saved in a
reader/judge output file, using the same prompt, model, and decoding
parameters, and keeps the raw replies. It reports how many replies are not
exactly ``CORRECT``/``INCORRECT`` and how the score would move under the same
normalization that EM already applies.

It changes nothing: ``answer_judge_llms.py`` is read-only here and the
canonical result files are untouched. The point is to decide whether a metric
fix is worth breaking comparability with the published numbers.

Example:

    python src/judge_diagnostics.py \
      --input runs/fullwiki_qrag_fixed_steps6_answer_judge_mt1000.json \
      --output runs/judge_diagnostics_steps6.json \
      --samples 500 --base-url http://127.0.0.1:8010/v1 --model Qwen3-4B
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
from pathlib import Path
import random
from typing import Any, Sequence

from build_index_wiki_qrag import add_qrag_repo_to_path


LOG = logging.getLogger("judge-diagnostics")

CANONICAL = ("CORRECT", "INCORRECT")


def load_judge_pieces(qrag_repo: Path) -> tuple[Any, Any, Any, Any, str]:
    add_qrag_repo_to_path(qrag_repo)
    from answer_judge_llms import final_answer, normalize_answer
    from prompts_and_metrics import prompts
    from vLLM_clients import SyncVllmClient, extract_response_text

    return (
        SyncVllmClient,
        extract_response_text,
        final_answer,
        normalize_answer,
        prompts.sys_judge,
    )


def judge_requests(records: Sequence[dict[str, Any]]) -> list[str]:
    return [
        f"QUESTION: {record['question']}\n"
        f"PREDICTED ANSWER: {record['prediction']}\n"
        f"GROUNDTRUTH ANSWER: {record['answer']}"
        for record in records
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parser.add_argument("--model", default="Qwen3-4B")
    parser.add_argument("--judge-max-tokens", type=int, default=100)
    parser.add_argument(
        "--qrag-repo", type=Path, default=Path("/home/a.anokhin/Judge/Q-RAG-feedback")
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    (
        SyncVllmClient,
        extract_response_text,
        final_answer,
        normalize_answer,
        sys_judge,
    ) = load_judge_pieces(args.qrag_repo.expanduser().resolve())

    with args.input.expanduser().resolve().open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    random.seed(args.seed)
    sampled = (
        records
        if args.samples >= len(records)
        else random.sample(records, args.samples)
    )
    LOG.info("Re-judging %d of %d records", len(sampled), len(records))

    replies = []
    with SyncVllmClient(
        llm=args.model,
        base_url=args.base_url,
        api_key=None,
        system_prompt=sys_judge,
        thinking=False,
        max_tokens=args.judge_max_tokens,
        temperature=0.0,
    ) as judge:
        for index, request in enumerate(judge_requests(sampled), start=1):
            replies.append(extract_response_text(judge.chat_completion(request)))
            if index % 100 == 0:
                LOG.info("%d/%d", index, len(sampled))

    strict_hits = normalized_hits = 0
    noncanonical: collections.Counter[str] = collections.Counter()
    flipped = []
    for record, reply in zip(sampled, replies):
        verdict = final_answer(reply).upper()
        strict = float(verdict == "CORRECT")
        normalized = float(normalize_answer(verdict).upper() == "CORRECT")
        strict_hits += strict
        normalized_hits += normalized
        if verdict not in CANONICAL:
            noncanonical[verdict[:80]] += 1
        if strict != normalized:
            flipped.append(
                {
                    "id": record.get("id"),
                    "verdict": verdict[:120],
                    "strict": strict,
                    "normalized": normalized,
                }
            )

    total = len(sampled)
    report = {
        "input": str(args.input),
        "samples": total,
        "strict_judge_score": strict_hits / total,
        "normalized_judge_score": normalized_hits / total,
        "score_delta": (normalized_hits - strict_hits) / total,
        "noncanonical_replies": sum(noncanonical.values()),
        "noncanonical_share": sum(noncanonical.values()) / total,
        "flipped_by_normalization": len(flipped),
        "most_common_noncanonical": noncanonical.most_common(10),
        "flipped_examples": flipped[:10],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        LOG.info("Wrote %s", destination)
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
