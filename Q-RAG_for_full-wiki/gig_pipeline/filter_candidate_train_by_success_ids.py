"""
Filter a prepared candidate_train JSONL by ids selected from success-rate results.

Default selection is the q0_s1 subset:
  question_only_success_rate == 0.0
  with_support_facts_success_rate == 1.0

Examples:
    python3 filter_candidate_train_by_success_ids.py \
        --success-input output/success_rate_experiment/hotpotqa_train_qrag_chunks_success_rates.jsonl \
        --candidate-input output/hotpotqa/hotpot_candidate_train.jsonl \
        --ids-output output/success_rate_experiment/hotpotqa_train_q0_s1_ids.txt \
        --output output/hotpotqa/hotpot_candidate_train_q0_s1.jsonl

    python3 filter_candidate_train_by_success_ids.py \
        --success-input output/success_rate_experiment/2wiki_train_qrag_chunks_success_rates.jsonl \
        --candidate-input output/2wiki/2wiki_candidate_train.jsonl \
        --ids-output output/success_rate_experiment/2wiki_train_q0_s1_ids.txt \
        --output output/2wiki/2wiki_candidate_train_q0_s1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from collections.abc import Iterable
from typing import Any


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl_atomic(path: str, records: Iterable[dict[str, Any]]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def write_ids_atomic(path: str, ids: Iterable[str]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for item_id in ids:
            f.write(f"{item_id}\n")
    os.replace(tmp, path)


def rate_matches(value: Any, expected: float, tolerance: float) -> bool:
    return abs(float(value) - expected) <= tolerance


def select_success_records(
    records: list[dict[str, Any]],
    question_only_rate: float,
    with_support_rate: float,
    tolerance: float,
) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        if rate_matches(record.get("question_only_success_rate", 0.0), question_only_rate, tolerance) and rate_matches(
            record.get("with_support_facts_success_rate", 0.0), with_support_rate, tolerance
        ):
            selected.append(record)
    return selected


def unique_ids_in_order(records: list[dict[str, Any]], id_field: str) -> tuple[list[str], int]:
    ids: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    for record in records:
        item_id = str(record[id_field])
        if item_id in seen:
            duplicates += 1
            continue
        seen.add(item_id)
        ids.append(item_id)
    return ids, duplicates


def value_counts(records: list[dict[str, Any]], field: str) -> Counter[str] | None:
    if not any(field in record for record in records):
        return None
    return Counter(str(record.get(field, "")) or "<empty>" for record in records)


def print_distribution(label: str, records: list[dict[str, Any]], field: str) -> None:
    counts = value_counts(records, field)
    if counts is None:
        print(f"{label} {field}: <field absent>")
        return

    rendered = ", ".join(
        f"{key}={value}"
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )
    print(f"{label} {field}: {rendered}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter candidate_train JSONL by ids selected from success-rate JSONL"
    )
    parser.add_argument("--success-input", required=True)
    parser.add_argument("--candidate-input", required=True)
    parser.add_argument("--ids-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--success-id-field", default="id")
    parser.add_argument("--candidate-id-field", default="_id")
    parser.add_argument("--question-only-rate", type=float, default=0.0)
    parser.add_argument("--with-support-rate", type=float, default=1.0)
    parser.add_argument("--rate-tolerance", type=float, default=1e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    success_records = read_jsonl(args.success_input)
    selected_success_records = select_success_records(
        success_records,
        question_only_rate=args.question_only_rate,
        with_support_rate=args.with_support_rate,
        tolerance=args.rate_tolerance,
    )
    selected_ids, duplicate_success_ids = unique_ids_in_order(
        selected_success_records,
        args.success_id_field,
    )
    selected_id_set = set(selected_ids)

    candidate_records = read_jsonl(args.candidate_input)
    filtered_records = [
        record
        for record in candidate_records
        if str(record.get(args.candidate_id_field, "")) in selected_id_set
    ]
    candidate_ids = {
        str(record[args.candidate_id_field])
        for record in candidate_records
        if args.candidate_id_field in record
    }
    missing_ids = selected_id_set - candidate_ids
    outside_ids = [
        str(record.get(args.candidate_id_field, ""))
        for record in filtered_records
        if str(record.get(args.candidate_id_field, "")) not in selected_id_set
    ]

    write_ids_atomic(args.ids_output, selected_ids)
    write_jsonl_atomic(args.output, filtered_records)

    print(f"Success-rate input records:           {len(success_records)}")
    print(f"Selected q0_s1 success-rate ids:      {len(selected_ids)}")
    print(f"Duplicate selected success-rate ids:  {duplicate_success_ids}")
    print(f"Candidate input records:              {len(candidate_records)}")
    print(f"Filtered candidate records:           {len(filtered_records)}")
    print(f"Selected ids missing in candidate:    {len(missing_ids)}")
    print(f"Filtered records outside id set:      {len(outside_ids)}")
    print_distribution("Selected success-rate", selected_success_records, "type")
    print_distribution("Selected success-rate", selected_success_records, "level")
    print_distribution("Filtered candidate", filtered_records, "type")
    print_distribution("Filtered candidate", filtered_records, "level")
    print(f"IDs output:                            {args.ids_output}")
    print(f"Filtered JSONL output:                 {args.output}")

    if outside_ids:
        raise SystemExit("Filtered output contains records outside selected id set")


if __name__ == "__main__":
    main()
