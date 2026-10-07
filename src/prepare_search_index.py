#!/usr/bin/env python3
"""Point a downloaded or freshly built GTE index at the local corpus.

Direct search reads chunk texts through byte offsets of the corpus rows, and
the offsets record the corpus path, size and modification time: a copied or
downloaded corpus fails that check even when its content is identical. This
script records the local corpus path in the index manifest and rebuilds the
offsets table for it (about a minute for Wiki-18).

    python src/prepare_search_index.py --index-dir $DATA/full-wiki/wiki18-gte \\
        --corpus $DATA/full-wiki/wiki_dump.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from fullwiki_qrag import (
    DEFAULT_OFFSETS_FILE,
    build_corpus_offsets,
    load_and_validate_matrix_manifest,
    offsets_metadata_path,
)


LOG = logging.getLogger("prepare-search-index")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    index_dir = args.index_dir.expanduser().resolve()
    corpus = args.corpus.expanduser().resolve()
    if not corpus.is_file():
        raise SystemExit(f"Corpus not found: {corpus}")

    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corpus.stat().st_size != int(manifest["corpus"]["size_bytes"]):
        raise SystemExit(
            f"{corpus} has {corpus.stat().st_size} bytes, the index was built "
            f"from {manifest['corpus']['size_bytes']}"
        )
    manifest["corpus"]["source_path"] = str(corpus)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    LOG.info("Corpus path recorded in %s", manifest_path)

    # The offsets are rebuilt rather than patched: a stale table for another
    # corpus would otherwise pass the identity check.
    offsets = index_dir / DEFAULT_OFFSETS_FILE
    for stale in (offsets, offsets_metadata_path(offsets)):
        stale.unlink(missing_ok=True)
    build_corpus_offsets(corpus, offsets, int(manifest["corpus"]["rows"]))
    load_and_validate_matrix_manifest(index_dir)
    LOG.info("Ready: %s", index_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
