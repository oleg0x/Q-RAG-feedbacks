from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import build_title_table as title_table


def write_corpus(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows() -> list[dict]:
    """A corpus reproducing the title shapes found in Wiki-18.

    Quotes inside a title, non-ASCII text and several chunks of one article:
    exactly the three cases that break naive byte-level title parsing.
    """
    return [
        {"id": "0", "contents": '"First title"\nFirst passage.'},
        {"id": "1", "contents": '"First title"\nSecond chunk of the same article.'},
        {"id": "2", "contents": '"Quoted \\"title\\""\nThird passage.'},
        {"id": "3", "contents": '"Zürich"\nUnicode passage.'},
        {"id": "4", "contents": '"First title"\nThird chunk of the same article.'},
    ]


def build(tmp_path: Path, rows: list[dict] | None = None, **kwargs):
    rows = sample_rows() if rows is None else rows
    corpus = tmp_path / "wiki.jsonl"
    write_corpus(corpus, rows)
    output = tmp_path / "corpus-title-ids.npy"
    metadata = title_table.build_title_table(
        corpus,
        output,
        len(rows),
        progress_every=0,
        **kwargs,
    )
    return corpus, output, metadata


def test_table_length_matches_corpus_rows(tmp_path: Path) -> None:
    rows = sample_rows()
    _, output, metadata = build(tmp_path)

    table = np.load(output, allow_pickle=False)
    assert table.dtype == np.int32
    assert table.shape == (len(rows),)
    assert metadata["rows"] == len(rows)


def test_chunks_of_one_article_share_a_title_id(tmp_path: Path) -> None:
    _, output, metadata = build(tmp_path)

    table = np.load(output, allow_pickle=False)
    titles = title_table.read_titles(title_table.titles_path(output))

    assert table[0] == table[1] == table[4]
    assert len({int(table[0]), int(table[2]), int(table[3])}) == 3
    assert metadata["distinct_titles"] == len(titles) == 3
    assert titles[int(table[0])] == "First title"
    assert titles[int(table[2])] == 'Quoted "title"'
    assert titles[int(table[3])] == "Zürich"


def test_every_row_matches_a_full_json_decode(tmp_path: Path) -> None:
    """Acceptance: check the table against a full JSON parse, not against itself."""
    rows = sample_rows()
    corpus, output, _ = build(tmp_path)

    table = np.load(output, allow_pickle=False)
    titles = title_table.read_titles(title_table.titles_path(output))
    for row, item in enumerate(rows):
        expected = json.loads(item["contents"].partition("\n")[0])
        assert titles[int(table[row])] == expected

    # The same path the script uses to check itself.
    assert title_table.verify_table(corpus, table, titles, samples=64, seed=1) > 0


def test_row_count_mismatch_leaves_no_artifact(tmp_path: Path) -> None:
    """An unfinished table must not look complete to the next launch."""
    corpus = tmp_path / "wiki.jsonl"
    write_corpus(corpus, sample_rows())
    output = tmp_path / "corpus-title-ids.npy"

    with pytest.raises(ValueError, match="rows"):
        title_table.build_title_table(corpus, output, 4, progress_every=0)

    assert not output.exists()
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_verification_catches_a_corrupted_table(tmp_path: Path) -> None:
    corpus, output, _ = build(tmp_path)
    table = np.load(output, allow_pickle=False)
    titles = title_table.read_titles(title_table.titles_path(output))
    table[3] = table[0]

    with pytest.raises(RuntimeError, match="title mismatch"):
        title_table.verify_table(corpus, table, titles, samples=256, seed=1)
