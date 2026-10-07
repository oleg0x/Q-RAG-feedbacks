"""Unit tests for ``prepare_search_index.py``: a downloaded index must end up
pointing at the local corpus, with offsets that pass the search-time check."""

from __future__ import annotations

import json

import numpy as np
import pytest

import fullwiki_qrag as retrieval
import prepare_search_index


ROWS = [
    {"id": "0", "contents": "\"Alpha\"\nfirst passage"},
    {"id": "1", "contents": "\"Beta\"\nsecond passage"},
    {"id": "2", "contents": "\"Gamma\"\nthird passage"},
]


def make_index(tmp_path, corpus_size: int):
    index = tmp_path / "wiki18-gte"
    index.mkdir()
    manifest = {
        "index": {"metric": "inner_product_on_l2_normalized_vectors", "dimension": 768},
        "corpus": {
            "faiss_id_mapping": "zero-based JSONL row number",
            "rows": len(ROWS),
            "size_bytes": corpus_size,
            "source_path": "/elsewhere/wiki_dump.jsonl",
        },
    }
    (index / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    # Offsets as they arrive in a download: built for a corpus at another path.
    np.save(index / "corpus-row-offsets.npy", np.zeros(len(ROWS) + 1, dtype="<u8"))
    (index / "corpus-row-offsets.json").write_text(
        json.dumps({"corpus": {"path": "/elsewhere/wiki_dump.jsonl"}, "rows": len(ROWS)}),
        encoding="utf-8",
    )
    return index


def write_corpus(tmp_path):
    corpus = tmp_path / "wiki_dump.jsonl"
    corpus.write_text("".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8")
    return corpus


def test_index_is_pointed_at_the_local_corpus(tmp_path) -> None:
    corpus = write_corpus(tmp_path)
    index = make_index(tmp_path, corpus.stat().st_size)

    assert prepare_search_index.main(["--index-dir", str(index), "--corpus", str(corpus)]) == 0

    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus"]["source_path"] == str(corpus.resolve())
    resolved = retrieval.resolve_corpus_path(manifest, None)
    offsets = retrieval.validate_offsets(
        resolved, index / "corpus-row-offsets.npy", len(ROWS)
    )
    reader = retrieval.WikiCorpus(
        resolved, index / "corpus-row-offsets.npy", len(ROWS), auto_prepare=False
    )
    assert int(offsets[-1]) == corpus.stat().st_size
    assert [row["id"] for row in reader.read_rows([2, 0])] == ["2", "0"]


def test_corpus_of_another_size_is_refused(tmp_path) -> None:
    corpus = write_corpus(tmp_path)
    index = make_index(tmp_path, corpus.stat().st_size + 1)

    with pytest.raises(SystemExit, match="bytes"):
        prepare_search_index.main(["--index-dir", str(index), "--corpus", str(corpus)])
    manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["corpus"]["source_path"] == "/elsewhere/wiki_dump.jsonl"
