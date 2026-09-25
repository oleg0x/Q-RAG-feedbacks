#!/usr/bin/env python3
"""Measure how many gold titles a QA dataset shares with the Wiki-18 corpus.

``title_em`` cannot reach 1.0 on HotpotQA: the questions were written against
a 2017 Wikipedia dump while Wiki-18 is a 2018 dump sliced into 100-word
chunks, so some gold articles were renamed, deleted, or never made it into
the DPR slice. Any retrieval number is only interpretable against that
ceiling, and the ceiling has to be measured rather than assumed.

The scan streams the corpus once and extracts only the title line of each
``contents`` field, so it costs one sequential read rather than 21M JSON
parses.

Example:

    python src/corpus_title_coverage.py \
      --corpus datasets/data_sources/full-wiki/.../wiki_dump.jsonl \
      --dataset datasets/data_sources/hotpotqa/hotpot_dev_fullwiki_v1.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any, Iterator, Sequence

from build_index_wiki_qrag import parse_wiki_title
from fullwiki_qrag import (
    load_input_samples,
    normalize_title,
    ordered_unique,
    sample_id,
)


LOG = logging.getLogger("corpus-title-coverage")

CONTENTS_KEY = b'"contents": "'
ESCAPED_NEWLINE = b"\\n"


def scan_corpus_titles(corpus: Path, progress_every: int) -> set[str]:
    """Collect every distinct article title in the corpus.

    Reads the raw bytes up to the first escaped newline instead of decoding
    each record: the title is the only field needed, and full JSON parsing
    of 21M rows would dominate the runtime.
    """
    titles: set[str] = set()
    started = time.monotonic()
    with corpus.open("rb") as stream:
        for row, line in enumerate(stream, start=1):
            start = line.find(CONTENTS_KEY)
            if start < 0:
                raise ValueError(f"{corpus}:{row} has no contents field")
            start += len(CONTENTS_KEY)
            end = line.find(ESCAPED_NEWLINE, start)
            if end < 0:
                raise ValueError(f"{corpus}:{row} has no title line")
            titles.add(parse_wiki_title(json.loads(f'"{line[start:end].decode()}"')))
            if progress_every and row % progress_every == 0:
                LOG.info(
                    "%d rows, %d distinct titles, %.0fs",
                    row,
                    len(titles),
                    time.monotonic() - started,
                )
    LOG.info(
        "Corpus scan finished: %d distinct titles in %.0fs",
        len(titles),
        time.monotonic() - started,
    )
    return titles


def verify_scan(corpus: Path, titles: set[str], samples: int) -> None:
    """Cross-check the byte-level parser against a full JSON decode."""
    checked = 0
    with corpus.open("rb") as stream:
        for row, line in enumerate(stream):
            if row >= samples * 97:
                break
            if row % 97:
                continue
            contents = json.loads(line)["contents"]
            expected = parse_wiki_title(contents.partition("\n")[0])
            if expected not in titles:
                raise RuntimeError(f"{corpus}:{row} title {expected!r} was not scanned")
            checked += 1
    LOG.info("Verified %d sampled rows against a full JSON decode", checked)


def gold_titles(sample: dict[str, Any]) -> list[str]:
    supporting = sample.get("supporting_facts")
    if not isinstance(supporting, list):
        return []
    return ordered_unique(
        str(fact[0]) for fact in supporting if isinstance(fact, list) and fact
    )


def coverage(
    samples: Sequence[dict[str, Any]],
    titles: set[str],
) -> dict[str, Any]:
    normalized = {normalize_title(title) for title in titles}
    total = complete_exact = complete_normalized = 0
    hit_exact = hit_normalized = 0
    missing: list[str] = []
    for index, sample in enumerate(samples):
        gold = gold_titles(sample)
        if not gold:
            continue
        total += 1
        exact = [title in titles for title in gold]
        normal = [normalize_title(title) in normalized for title in gold]
        hit_exact += sum(exact)
        hit_normalized += sum(normal)
        complete_exact += all(exact)
        complete_normalized += all(normal)
        missing.extend(
            title for title, found in zip(gold, normal) if not found
        )
    gold_total = sum(len(gold_titles(sample)) for sample in samples)
    return {
        "examples": total,
        "gold_titles": gold_total,
        "per_title_exact": hit_exact / gold_total if gold_total else 0.0,
        "per_title_normalized": hit_normalized / gold_total if gold_total else 0.0,
        "title_em_ceiling_exact": complete_exact / total if total else 0.0,
        "title_em_ceiling_normalized": complete_normalized / total if total else 0.0,
        "missing_distinct_titles": len(set(missing)),
        "missing_examples": sorted(set(missing))[:20],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=3_000_000)
    parser.add_argument("--verify-samples", type=int, default=200)
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
    corpus = args.corpus.expanduser().resolve()
    titles = scan_corpus_titles(corpus, args.progress_every)
    if args.verify_samples:
        verify_scan(corpus, titles, args.verify_samples)
    samples = load_input_samples(args.dataset.expanduser().resolve())
    report = {
        "corpus": str(corpus),
        "dataset": str(args.dataset.expanduser().resolve()),
        "distinct_corpus_titles": len(titles),
        **coverage(samples, titles),
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
