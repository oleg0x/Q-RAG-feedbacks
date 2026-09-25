#!/usr/bin/env python3
"""Собрать таблицу титулов Wiki-18: строка индекса → идентификатор статьи.

Прямой поиск по всем 21 015 324 чанкам маскирует кандидатов **до** ``topk``:
запрос содержит текст уже выбранного чанка, поэтому его соседи по статье
оказываются ближайшими соседями запроса и без маски забивают пул целиком.
Квота ``N=2`` чанка на титул означает, что на каждом шаге нужно знать титул
любой из 21 млн строк. Читать ради этого сто строк корпуса на шаг нельзя —
корпус лежит на диске и весит 14 ГБ, — поэтому титулы раскладываются один раз
в ``int32``-таблицу на 84 МБ, которая целиком живёт в памяти процесса.

Артефакты (в git не коммитятся, пересобираются этим скриптом):

* ``<output>.npy`` — ``int32[rows]``, идентификатор титула для каждой строки;
* ``<output>.titles.jsonl.gz`` — сами титулы, строка ``i`` это ``title_id == i``
  (по одной JSON-строке на титул: титул может содержать что угодно, включая
  перевод строки, и сырой текст здесь развалил бы файл);
* ``<output>.json`` — метаданные: идентичность корпуса, число строк и титулов.

Пример:

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
# int32 хватает: титулов в Wiki-18 около 5.2 млн, а знаковый предел 2.1 млрд.
TITLE_ID_DTYPE = np.int32


def titles_path(output: Path) -> Path:
    return output.with_suffix(".titles.jsonl.gz")


def metadata_path(output: Path) -> Path:
    return output.with_suffix(".json")


def corpus_stat_identity(corpus: Path) -> dict[str, Any]:
    """Ровно та же идентичность корпуса, что у таблицы смещений строк.

    Совпадение полей позволяет проверять обе таблицы одинаково и ловить
    подмену корпуса, а не только его усечение.
    """
    stat = corpus.stat()
    return {
        "path": str(corpus.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def row_title(line: bytes, corpus: Path, row: int) -> str:
    """Достать титул из сырой строки JSONL, не разбирая её целиком.

    Полный ``json.loads`` на 21 млн строк стоит дороже самого прохода по
    диску, а нужен из всей записи один титул — до первого экранированного
    перевода строки в поле ``contents``.
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
    """Один потоковый проход: заполнить ``table`` и вернуть титулы по id."""
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
    """Сверить случайные строки таблицы с полным JSON-разбором корпуса.

    Байтовый парсер титула быстрее полного разбора на порядок, и именно
    поэтому его нельзя принимать на веру: выборка читается второй раз и
    сравнивается с результатом ``json.loads`` всей записи.
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
        # Недостроенная таблица не должна выглядеть готовой: имя временного
        # файла уникально по pid, поэтому его можно спокойно удалить.
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
        help="каталог индекса: из его манифеста берутся корпус и число строк",
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
