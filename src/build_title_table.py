#!/usr/bin/env python3
"""Build the Wiki-18 title table: index row -> article identifier.

Direct search over all 21,015,324 chunks masks candidates **before** ``topk``:
the query contains the text of already selected chunks, so their neighbours
from the same article become the query's nearest neighbours and would
otherwise fill the whole pool. A per-title quota of ``N=2`` chunks means the
title of any of the 21M rows must be known at every step. Reading a hundred
corpus rows per step for this is not an option (the corpus is a 14 GB file on
disk), so titles are laid out once into an 84 MB ``int32`` table that lives
entirely in process memory.

Artifacts (not committed to git, rebuilt by this script):

* ``<output>.npy``: ``int32[rows]``, the title identifier of every row;
* ``<output>.titles.jsonl.gz``: the titles themselves, line ``i`` is
  ``title_id == i`` (one JSON string per title: a title may contain anything,
  including a newline, and raw text would break the file);
* ``<output>.json``: metadata: corpus identity, number of rows and titles.

Example:

    python src/build_title_table.py \
      --index-dir datasets/data_sources/full-wiki/wiki18-gte \
      --output datasets/data_sources/full-wiki/wiki18-gte/corpus-title-ids.npy
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from build_index_wiki_gte import atomic_write_json, read_json
from build_index_wiki_qrag import parse_wiki_title


LOG = logging.getLogger("build-title-table")

MANIFEST_FILE = "manifest.json"
CONTENTS_KEY = b'"contents": "'
ESCAPED_NEWLINE = b"\\n"
# int32 is enough: Wiki-18 has about 5.2M titles and the signed limit is 2.1B.
TITLE_ID_DTYPE = np.int32


def titles_path(output: Path) -> Path:
    return output.with_suffix(".titles.jsonl.gz")


def metadata_path(output: Path) -> Path:
    return output.with_suffix(".json")


def corpus_stat_identity(corpus: Path) -> dict[str, Any]:
    """Exactly the same corpus identity as the row-offset table uses.

    Matching fields let both tables be checked the same way and catch a
    replaced corpus, not just a truncated one.
    """
    stat = corpus.stat()
    return {
        "path": str(corpus.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def row_title(line: bytes, corpus: Path, row: int) -> str:
    """Extract the title from a raw JSONL line without parsing all of it.

    A full ``json.loads`` over 21M rows costs more than the disk pass itself,
    and only the title is needed: the part of ``contents`` up to the first
    escaped newline.
    """
    start = line.find(CONTENTS_KEY)
    if start < 0:
        raise ValueError(f"{corpus}:{row} has no contents field")
    start += len(CONTENTS_KEY)
    end = line.find(ESCAPED_NEWLINE, start)
    if end < 0:
        raise ValueError(f"{corpus}:{row} has no title line")
    return parse_wiki_title(json.loads(f'"{line[start:end].decode()}"'))


def scan_corpus(
    corpus: Path,
    expected_rows: int,
    table: np.ndarray,
    progress_every: int,
) -> list[str]:
    """One streaming pass: fill ``table`` and return the titles by id."""
    title_ids: dict[str, int] = {}
    titles: list[str] = []
    started = time.monotonic()
    scanned = 0
    with corpus.open("rb", buffering=16 * 1024 * 1024) as source:
        for row, line in enumerate(source):
            if row >= expected_rows:
                raise ValueError(
                    f"{corpus} has more rows than the manifest expects: {expected_rows}"
                )
            title = row_title(line, corpus, row)
            title_id = title_ids.get(title)
            if title_id is None:
                title_id = len(titles)
                title_ids[title] = title_id
                titles.append(title)
            table[row] = title_id
            scanned = row + 1
            if progress_every and scanned % progress_every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                LOG.info(
                    "%d/%d rows, %d distinct titles (%.0f rows/s)",
                    scanned,
                    expected_rows,
                    len(titles),
                    scanned / elapsed,
                )
    if scanned != expected_rows:
        raise ValueError(
            f"{corpus} has {scanned} rows, the manifest expects {expected_rows}"
        )
    if len(titles) > np.iinfo(TITLE_ID_DTYPE).max:
        raise ValueError(f"Too many distinct titles for int32: {len(titles)}")
    LOG.info(
        "Corpus scan finished: %d rows, %d distinct titles in %.0fs",
        scanned,
        len(titles),
        time.monotonic() - started,
    )
    return titles


def verify_table(
    corpus: Path,
    table: np.ndarray,
    titles: Sequence[str],
    samples: int,
    seed: int,
) -> int:
    """Check random table rows against a full JSON parse of the corpus.

    The byte-level title parser is an order of magnitude faster than a full
    parse, which is exactly why it cannot be taken on trust: a sample is read
    a second time and compared with ``json.loads`` of the whole record.
    """
    if samples <= 0:
        return 0
    rows = np.unique(
        np.random.default_rng(seed).integers(0, len(table), size=samples)
    )
    wanted = {int(value) for value in rows}
    checked = 0
    with corpus.open("rb", buffering=16 * 1024 * 1024) as source:
        for row, line in enumerate(source):
            if row not in wanted:
                continue
            item = json.loads(line)
            if str(item.get("id")) != str(row):
                raise RuntimeError(
                    f"Corpus row/id mismatch: row={row}, id={item.get('id')!r}"
                )
            expected = parse_wiki_title(str(item["contents"]).partition("\n")[0])
            actual = titles[int(table[row])]
            if actual != expected:
                raise RuntimeError(
                    f"{corpus}:{row} title mismatch: table={actual!r} "
                    f"corpus={expected!r}"
                )
            checked += 1
            if checked == len(wanted):
                break
    if checked != len(wanted):
        raise RuntimeError(f"Verified {checked} of {len(wanted)} sampled rows")
    LOG.info("Verified %d sampled rows against a full JSON decode", checked)
    return checked


def write_titles(path: Path, titles: Sequence[str]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8") as destination:
        for title in titles:
            destination.write(json.dumps(title, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def read_titles(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def build_title_table(
    corpus: Path,
    output: Path,
    expected_rows: int,
    *,
    progress_every: int = 1_000_000,
    verify_samples: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp-{os.getpid()}{output.suffix}")
    if temporary.exists():
        raise RuntimeError(f"Temporary artifact already exists: {temporary}")

    LOG.info("Building title table: rows=%d corpus=%s", expected_rows, corpus)
    table = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=TITLE_ID_DTYPE,
        shape=(expected_rows,),
    )
    try:
        titles = scan_corpus(corpus, expected_rows, table, progress_every)
        table.flush()
        checked = verify_table(corpus, table, titles, verify_samples, seed)
        del table
        write_titles(titles_path(output), titles)
        metadata = {
            "schema_version": 1,
            "corpus": corpus_stat_identity(corpus),
            "rows": expected_rows,
            "distinct_titles": len(titles),
            "dtype": np.dtype(TITLE_ID_DTYPE).str,
            "titles_file": titles_path(output).name,
            "verified_rows": checked,
        }
        atomic_write_json(metadata_path(output), metadata)
        os.replace(temporary, output)
    finally:
        # An unfinished table must not look complete: the temporary file name
        # is unique per pid, so it is safe to delete.
        if temporary.exists():
            temporary.unlink()
    LOG.info(
        "Title table complete: %s (%.1f MiB), titles=%s",
        output,
        output.stat().st_size / (1024**2),
        titles_path(output),
    )
    return metadata


def manifest_rows(index_dir: Path) -> tuple[int, Path]:
    manifest = read_json(index_dir / MANIFEST_FILE)
    corpus = manifest.get("corpus", {})
    rows = int(corpus["rows"])
    source_path = corpus.get("source_path")
    if not source_path:
        raise ValueError(f"{index_dir / MANIFEST_FILE} has no corpus.source_path")
    return rows, Path(source_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="index directory: the corpus and row count come from its manifest",
    )
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--verify-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
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
    rows = args.rows
    corpus = args.corpus
    if args.index_dir is not None:
        manifest_row_count, manifest_corpus = manifest_rows(
            args.index_dir.expanduser().resolve()
        )
        rows = manifest_row_count if rows is None else rows
        corpus = manifest_corpus if corpus is None else corpus
    if corpus is None or rows is None:
        raise ValueError("Pass --index-dir, or both --corpus and --rows")
    corpus = corpus.expanduser().resolve()
    if not corpus.is_file():
        raise FileNotFoundError(f"Wiki corpus not found: {corpus}")
    metadata = build_title_table(
        corpus,
        args.output.expanduser().resolve(),
        int(rows),
        progress_every=args.progress_every,
        verify_samples=args.verify_samples,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
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
