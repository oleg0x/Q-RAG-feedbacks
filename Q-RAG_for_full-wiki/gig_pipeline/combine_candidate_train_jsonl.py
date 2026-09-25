"""
Combine prepared candidate_train JSONL files into one training file.

Each input is passed as SOURCE=PATH. Records are copied as-is with an added
source field, so CandidateDatasetAdapter can still read _id/context/candidates/
judgements/betas, and analysis can recover the original dataset.

Example:
    python3 combine_candidate_train_jsonl.py \
        --input hotpotqa=output/hotpotqa/hotpot_candidate_train_q0_s1.jsonl \
        --input 2WikiMultihopQA=output/2wiki/2wiki_candidate_train_q0_s1.jsonl \
        --output output/combined/hotpotqa_2wiki_candidate_train_q0_s1.jsonl
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


def parse_input_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Invalid input spec {spec!r}; expected SOURCE=PATH"
        )
    source, path = spec.split("=", 1)
    source = source.strip()
    path = path.strip()
    if not source or not path:
        raise argparse.ArgumentTypeError(
            f"Invalid input spec {spec!r}; expected non-empty SOURCE=PATH"
        )
    return source, path


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
        description="Combine candidate_train JSONL files"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_input_spec,
        metavar="SOURCE=PATH",
        help="Input JSONL with source label, e.g. hotpotqa=file.jsonl",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-field", default="source")
    parser.add_argument("--id-field", default="_id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    combined: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    source_id_counts: Counter[tuple[str, str]] = Counter()
    raw_id_counts: Counter[str] = Counter()

    for source, path in args.input:
        records = read_jsonl(path)
        source_counts[source] += len(records)
        print(f"Input {source}: {len(records)} records from {path}")

        for record in records:
            out = dict(record)
            out[args.source_field] = source
            combined.append(out)

            if args.id_field in out:
                item_id = str(out[args.id_field])
                source_id_counts[(source, item_id)] += 1
                raw_id_counts[item_id] += 1

    duplicate_source_ids = sum(count - 1 for count in source_id_counts.values() if count > 1)
    duplicate_raw_ids = sum(count - 1 for count in raw_id_counts.values() if count > 1)

    write_jsonl_atomic(args.output, combined)

    print(f"Output records: {len(combined)}")
    print(f"Duplicate source+{args.id_field} records: {duplicate_source_ids}")
    print(f"Duplicate raw {args.id_field} records: {duplicate_raw_ids}")
    print_distribution("Combined", combined, args.source_field)
    print_distribution("Combined", combined, "type")
    print_distribution("Combined", combined, "level")
    print(f"Output JSONL: {args.output}")


if __name__ == "__main__":
    main()
