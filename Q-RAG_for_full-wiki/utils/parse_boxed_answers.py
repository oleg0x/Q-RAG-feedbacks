"""Add extracted ``answer`` and stable ``id`` fields to MATH JSONL data."""

import argparse
import json

from utils.process_latex import extract_boxed_expression


def convert(input_path: str, output_path: str) -> int:
    samples = []
    with open(input_path, "r", encoding="utf-8") as source:
        for index, line in enumerate(source, 1):
            if not line.strip():
                continue
            sample = json.loads(line)
            sample["answer"] = extract_boxed_expression(sample["solution"])
            sample.setdefault("id", index)
            samples.append(sample)
    with open(output_path, "w", encoding="utf-8") as destination:
        for sample in samples:
            destination.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()
    count = convert(args.input_path, args.output_path)
    print(f"Saved {count} samples to {args.output_path}")


if __name__ == "__main__":
    main()
