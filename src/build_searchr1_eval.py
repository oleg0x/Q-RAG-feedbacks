#!/usr/bin/env python3
"""Split the Search-R1 eval table into seven inputs for our pipeline.

`PeterJinGo/nq_hotpotqa_train` is named a train dataset, but its
``test.parquet`` holds all seven Search-R1 benchmarks in one table, keyed by
the ``data_source`` column: nq, triviaqa, popqa, hotpotqa, 2wikimultihopqa,
musique, bamboogle. Here it is split into seven JSONL files, each of which
``fullwiki_qrag.py`` reads as a regular input.

Two things are done here on purpose; without them the numbers are wrong.

**``supporting_facts`` is not written at all.** None of the seven datasets
has gold titles in the parquet, and an empty list is not the same as a
missing field: on empty gold ``title_metrics`` returns ``title_recall =
title_em = 1.0``, so the run would report a perfect title hit rate. Without
the field ``add_eval_fields`` simply skips title metrics and ``score_run``
sets them to ``null``.

**All answer aliases are kept.** Search-R1 computes EM as the maximum over
all ``golden_answers``, whereas ``answer_judge_llms.py`` (read-only) compares
against a single ``answer``. So the first answer goes to ``answer`` and the
rest to ``answer_aliases``, where ``export_eval_json.py`` and
``report_phase0.score_run`` pick them up for alias versions of EM and F1.
Without this TriviaQA (14 accepted answers per question on average, 88% of
questions with several) would be understated several-fold.

Example:

    python src/build_searchr1_eval.py \
      --input runs/shared/searchr1/test.parquet \
      --output-dir runs/shared/searchr1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterator, Sequence

from fullwiki_qrag import load_input_samples, sample_id


LOG = logging.getLogger("build-searchr1-eval")

# Expected table composition, checked during conversion: a different number
# of samples means a different dataset snapshot was downloaded, and then our
# numbers cannot be put next to the published Search-R1 ones.
EXPECTED_ROWS: dict[str, int] = {
    "nq": 3610,
    "triviaqa": 11313,
    "popqa": 14267,
    "hotpotqa": 7405,
    "2wikimultihopqa": 12576,
    "musique": 2417,
    "bamboogle": 125,
}

# Fields that must not appear in the output. The list exists for
# ``supporting_facts``: an empty list silently turns title metrics into 100%,
# see the module docstring.
FORBIDDEN_FIELDS = ("supporting_facts", "context", "sf_idx")


def convert_sample(row: dict[str, Any], source: str) -> dict[str, Any]:
    question = row.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{source}: sample {row.get('id')!r} has no question")
    # Not `or []`: pandas returns golden_answers as a numpy array, and the truth
    # value of an array longer than one element raises instead of being empty.
    golden = row.get("golden_answers")
    answers = [] if golden is None else [str(value) for value in golden]
    answers = [value for value in answers if value.strip()]
    if not answers:
        raise ValueError(f"{source}: sample {row.get('id')!r} has no answers")
    return {
        # The parquet id is unique only within its data_source: the first row
        # of both nq and popqa is called `test_0`. The prefix makes the id
        # unique across the whole run registry.
        "_id": f"{source}_{row['id']}",
        "data_source": source,
        "question": question,
        "answer": answers[0],
        "answer_aliases": answers[1:],
    }


def convert(rows: Sequence[dict[str, Any]], source: str) -> Iterator[dict[str, Any]]:
    for row in rows:
        record = convert_sample(row, source)
        leaked = [field for field in FORBIDDEN_FIELDS if field in record]
        if leaked:
            raise RuntimeError(
                f"{source}: the record contains fields {leaked}; an empty "
                "supporting_facts gives title EM = 100% on empty gold"
            )
        yield record


def verify_output(path: Path, expected: Sequence[dict[str, Any]]) -> None:
    """Re-read the output with the same code the pipeline will read it with.

    What is checked is not the file as text but what
    ``fullwiki_qrag.load_input_samples`` extracts from it: the same order,
    questions and identifiers, and ``supporting_facts`` still absent.
    """
    reread = load_input_samples(path)
    if len(reread) != len(expected):
        raise RuntimeError(
            f"{path.name}: re-read {len(reread)} samples instead of {len(expected)}"
        )
    for index, (actual, want) in enumerate(zip(reread, expected)):
        if actual["question"] != want["question"]:
            raise RuntimeError(f"{path.name}: question {index} differs after writing")
        if sample_id(actual, index) != want["_id"]:
            raise RuntimeError(f"{path.name}: id {index} differs after writing")
        if actual.get("supporting_facts") is not None:
            raise RuntimeError(
                f"{path.name}: sample {index} gained supporting_facts"
            )


def write_source(
    rows: Sequence[dict[str, Any]], source: str, directory: Path
) -> tuple[Path, int]:
    destination = directory / f"{source}.jsonl"
    records = list(convert(rows, source))
    # Through a temporary file: an interrupted conversion must not leave a
    # plausible but incomplete input for a multi-hour run.
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    verify_output(destination, records)
    with_aliases = sum(1 for record in records if record["answer_aliases"])
    LOG.info(
        "%-16s %6d samples, %5d with aliases (%.0f%%) → %s",
        source,
        len(records),
        with_aliases,
        100 * with_aliases / len(records),
        destination.name,
    )
    return destination, len(records)


def load_table(path: Path) -> dict[str, list[dict[str, Any]]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    missing = {"id", "question", "golden_answers", "data_source"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: the table lacks columns {sorted(missing)}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in frame.to_dict("records"):
        grouped.setdefault(str(row["data_source"]), []).append(row)
    return grouped


def check_composition(grouped: dict[str, list[dict[str, Any]]]) -> None:
    actual = {source: len(rows) for source, rows in grouped.items()}
    if actual != EXPECTED_ROWS:
        raise ValueError(
            "Table composition differs from the expected one; our numbers "
            "cannot be put next to the published Search-R1 ones.\n"
            f"expected: {EXPECTED_ROWS}\n"
            f"got:      {actual}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    source_path = args.input.expanduser().resolve()
    directory = args.output_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)

    grouped = load_table(source_path)
    LOG.info("Read %d datasets: %s", len(grouped), source_path)
    check_composition(grouped)

    total = 0
    # Iterate in EXPECTED_ROWS order, not table order, so that the log reads
    # the same from launch to launch.
    for source in EXPECTED_ROWS:
        total += write_source(grouped[source], source, directory)[1]
    LOG.info("Wrote %d samples to %s", total, directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
