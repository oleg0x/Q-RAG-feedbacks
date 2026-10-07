"""Random access to Wiki-18 corpus rows through an offset table.

The corpus is a 14 GB JSONL, but each step needs only the chunks the policy
selected. The ``uint64[rows + 1]`` offset table is built by
``python src/fullwiki_qrag.py prepare`` (evaluation code at the repository
root); here it is only read.

``WikiCorpus`` from ``src/fullwiki_qrag.py`` is duplicated on purpose: the
evaluation code and the training code have different dependencies, and an
import across that boundary would tie training to the evaluation code's paths.
Only the artifact format is shared.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class CorpusReader:
    """Reads corpus rows by number; the number matches the matrix row."""

    def __init__(
        self,
        corpus: str | Path,
        offsets: str | Path,
        expected_rows: int | None = None,
    ) -> None:
        self.path = Path(corpus).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Wiki corpus not found: {self.path}")
        offsets_path = Path(offsets).expanduser().resolve()
        if not offsets_path.is_file():
            raise FileNotFoundError(
                f"Corpus offsets not found: {offsets_path}. "
                "Build them with `python src/fullwiki_qrag.py prepare`."
            )
        self.offsets = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
        if self.offsets.ndim != 1:
            raise ValueError(f"Offsets must be 1-D, got {self.offsets.shape}")
        self.rows = int(self.offsets.shape[0]) - 1
        if expected_rows is not None and self.rows != expected_rows:
            raise ValueError(
                f"Offsets cover {self.rows} rows, the action matrix has {expected_rows}"
            )
        if int(self.offsets[-1]) != self.path.stat().st_size:
            raise RuntimeError(
                f"Offsets do not match {self.path}: last offset "
                f"{int(self.offsets[-1])} != size {self.path.stat().st_size}"
            )
        self._fd: int | None = None

    def _descriptor(self) -> int:
        if self._fd is None:
            self._fd = os.open(self.path, os.O_RDONLY)
        return self._fd

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "CorpusReader":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def read_row(self, row_id: int) -> dict[str, Any]:
        row_id = int(row_id)
        if not 0 <= row_id < self.rows:
            raise IndexError(f"Corpus row outside range: {row_id}")
        start = int(self.offsets[row_id])
        end = int(self.offsets[row_id + 1])
        # `pread` instead of `seek` + `read`: chunks are read by a thread pool,
        # and seek-then-read on a shared descriptor is not atomic, so two
        # threads can get each other's rows.
        raw = os.pread(self._descriptor(), end - start, start)
        item = json.loads(raw)
        # The matrix row must equal the record ``id``: the index was built by
        # JSONL line number, so a mismatch means training reads different text
        # from what the encoder embedded.
        if str(item.get("id")) != str(row_id):
            raise RuntimeError(
                f"Corpus row/id mismatch: row={row_id}, id={item.get('id')!r}"
            )
        contents = item.get("contents")
        if not isinstance(contents, str):
            raise RuntimeError(f"Corpus row {row_id} has no string contents")
        return item

    def read_texts(self, row_ids: Sequence[int]) -> list[str]:
        return [str(self.read_row(row_id)["contents"]) for row_id in row_ids]
