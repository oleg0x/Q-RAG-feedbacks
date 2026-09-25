"""
Filter per-question success-rate experiment results.

Examples:
    python filter_success_rate_results.py \
        --input output/success_rate_experiment/hotpotqa_dev_qrag_chunks_success_rates.jsonl \
        --output output/success_rate_experiment/hotpotqa_dev_filtered.jsonl \
        --max-question-only 0.25 \
        --min-with-support 0.75

    python filter_success_rate_results.py \
        --input output/success_rate_experiment/2wiki_dev_qrag_chunks_success_rates.jsonl \
        --output output/success_rate_experiment/2wiki_dev_big_delta.jsonl \
        --min-delta 0.5

    python filter_success_rate_results.py \
        --input output/success_rate_experiment/hotpotqa_dev_qrag_chunks_success_rates.jsonl \
        --output output/success_rate_experiment/hotpotqa_dev_expr.jsonl \
        --expr "s >= 0.75 and q <= 0.25 and delta >= 0.5"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(path: str, records: list[dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def mode_counts(record: dict, mode: str) -> tuple[int, int, int]:
    judgements = record.get(mode, {}).get("judgements", [])
    correct = sum(1 for value in judgements if value == 1)
    failed = sum(1 for value in judgements if value == -1)
    return correct, len(judgements), failed


def write_csv_summary(path: str, records: list[dict]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    fields = [
        "dataset",
        "split",
        "id",
        "question",
        "answer",
        "type",
        "level",
        "question_only_success_rate",
        "with_support_facts_success_rate",
        "delta_success_rate",
        "question_only_correct",
        "question_only_total",
        "question_only_failed_judge",
        "with_support_facts_correct",
        "with_support_facts_total",
        "with_support_facts_failed_judge",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            q_correct, q_total, q_failed = mode_counts(record, "question_only")
            s_correct, s_total, s_failed = mode_counts(record, "with_support_facts")
            q_rate = float(record.get("question_only_success_rate", 0.0))
            s_rate = float(record.get("with_support_facts_success_rate", 0.0))
            writer.writerow({
                "dataset": record.get("dataset", ""),
                "split": record.get("split", ""),
                "id": record.get("id", ""),
                "question": record.get("question", ""),
                "answer": record.get("answer", ""),
                "type": record.get("type", ""),
                "level": record.get("level", ""),
                "question_only_success_rate": q_rate,
                "with_support_facts_success_rate": s_rate,
                "delta_success_rate": s_rate - q_rate,
                "question_only_correct": q_correct,
                "question_only_total": q_total,
                "question_only_failed_judge": q_failed,
                "with_support_facts_correct": s_correct,
                "with_support_facts_total": s_total,
                "with_support_facts_failed_judge": s_failed,
            })


def in_optional_range(value: float, lower: float | None, upper: float | None) -> bool:
    if lower is not None and value < lower:
        return False
    if upper is not None and value > upper:
        return False
    return True


def eval_expr(expr: str, record: dict, q: float, s: float, delta: float) -> bool:
    allowed_names: dict[str, Any] = {
        "q": q,
        "question_only": q,
        "s": s,
        "with_support": s,
        "delta": delta,
        "record": record,
    }
    return bool(eval(expr, {"__builtins__": {}}, allowed_names))


def keep_record(record: dict, args: argparse.Namespace) -> bool:
    q = float(record.get("question_only_success_rate", 0.0))
    s = float(record.get("with_support_facts_success_rate", 0.0))
    delta = s - q

    if not in_optional_range(q, args.min_question_only, args.max_question_only):
        return False
    if not in_optional_range(s, args.min_with_support, args.max_with_support):
        return False
    if not in_optional_range(delta, args.min_delta, args.max_delta):
        return False
    if args.require_support_improves and delta <= 0.0:
        return False
    if args.expr is not None and not eval_expr(args.expr, record, q, s, delta):
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter success-rate JSONL records")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output", default=None)

    parser.add_argument("--min-question-only", type=float, default=None)
    parser.add_argument("--max-question-only", type=float, default=None)
    parser.add_argument("--min-with-support", type=float, default=None)
    parser.add_argument("--max-with-support", type=float, default=None)
    parser.add_argument("--min-delta", type=float, default=None,
                        help="Minimum with_support_facts_success_rate - question_only_success_rate")
    parser.add_argument("--max-delta", type=float, default=None,
                        help="Maximum with_support_facts_success_rate - question_only_success_rate")
    parser.add_argument("--require-support-improves", action="store_true", default=False)
    parser.add_argument("--expr", default=None,
                        help="Optional Python expression over q, s, delta, record")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    kept = []
    for record in records:
        try:
            if keep_record(record, args):
                kept.append(record)
        except Exception as exc:
            rid = record.get("id", "?")
            raise SystemExit(f"Failed to evaluate record {rid}: {exc}") from exc

    write_jsonl_atomic(args.output, kept)
    if args.csv_output is not None:
        write_csv_summary(args.csv_output, kept)

    print(f"Input records:  {len(records)}")
    print(f"Kept records:   {len(kept)}")
    print(f"Output JSONL:   {args.output}")
    if args.csv_output is not None:
        print(f"Output CSV:     {args.csv_output}")


if __name__ == "__main__":
    main()
