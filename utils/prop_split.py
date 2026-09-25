"""Create a reproducible stratified train/test split for MATH JSONL data."""

import argparse
import json
from collections import Counter


def calculate_proportions(data):
    counts = Counter(item["type"] for item in data)
    total = len(data)
    return ({name: count / total * 100 for name, count in counts.items()}, total)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path")
    parser.add_argument("train_path")
    parser.add_argument("test_path")
    parser.add_argument("--test-size", type=int, default=1000)
    args = parser.parse_args()

    from sklearn.model_selection import train_test_split

    with open(args.input_path, encoding="utf-8") as source:
        data = [json.loads(line) for line in source if line.strip()]
    train_data, test_data = train_test_split(
        data,
        test_size=args.test_size,
        random_state=42,
        shuffle=True,
        stratify=[item["type"] for item in data],
    )
    for path, samples in ((args.train_path, train_data), (args.test_path, test_data)):
        with open(path, "w", encoding="utf-8") as destination:
            for item in samples:
                destination.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
