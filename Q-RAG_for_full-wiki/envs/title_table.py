"""Wiki-18 title table: index row → article identifier.

The artifact is built by ``src/build_title_table.py`` (one streaming pass over
the corpus) and is only read here. It is needed in two places:

* the per-title quota of ``N`` chunks masks rows **before** ``topk``; without
  the table the title of each row would have to be read from the corpus, a
  hundred rows per step;
* the flag "both gold titles are in the corpus" is written to the episode log:
  38.5% of the training set lacks the full set, and without the flag the
  reward curve mixes unsolvable examples with solvable ones.

Format: ``<output>.npy`` (``int32[rows]``), ``<output>.titles.jsonl.gz`` (one
JSON string per title, line number equals ``title_id``) and ``<output>.json``
with metadata.
"""

from __future__ import annotations

import gzip
import html
import json
from pathlib import Path
from typing import Iterable, Sequence
import unicodedata

import numpy as np


def normalize_title(title: str) -> str:
    """Bring titles from two Wikipedia dumps to a comparable form.

    A copy of ``normalize_title`` from ``src/fullwiki_qrag.py``: HotpotQA stores
    titles of the 2017 dump with HTML entities (``Procter &amp; Gamble``), while
    Wiki-18 titles are already decoded. Strict equality underestimates gold-title
    coverage by about 4.6 pp. The original cannot be imported (it lives in a
    separate package), so the two definitions must match literally.
    """
    decoded = html.unescape(title)
    folded = unicodedata.normalize("NFKC", decoded).casefold()
    return " ".join(folded.replace("_", " ").split())


class TitleTable:
    """``int32[rows]`` holding the title id of every index row."""

    def __init__(
        self,
        title_ids: np.ndarray,
        titles: Sequence[str] | None = None,
        metadata: dict | None = None,
    ) -> None:
        if title_ids.ndim != 1:
            raise ValueError(f"Title table must be 1-D, got {title_ids.shape}")
        self.title_ids = title_ids
        self.titles = titles
        self.metadata = metadata or {}
        self._normalized: dict[str, int] | None = None

    @staticmethod
    def titles_path(path: Path) -> Path:
        return path.with_suffix(".titles.jsonl.gz")

    @staticmethod
    def metadata_path(path: Path) -> Path:
        return path.with_suffix(".json")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_rows: int | None = None,
        load_titles: bool = True,
        mmap: bool = False,
    ) -> "TitleTable":
        path = Path(path).expanduser().resolve()
        title_ids = np.load(
            path,
            mmap_mode="r" if mmap else None,
            allow_pickle=False,
        )
        if expected_rows is not None and title_ids.shape != (expected_rows,):
            raise ValueError(
                f"Title table has {title_ids.shape} rows, "
                f"the action matrix has {expected_rows}"
            )
        metadata = {}
        metadata_path = cls.metadata_path(path)
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if int(metadata.get("rows", len(title_ids))) != len(title_ids):
                raise ValueError(
                    f"{metadata_path} claims {metadata.get('rows')} rows, "
                    f"{path} has {len(title_ids)}"
                )
        titles = None
        if load_titles:
            titles_path = cls.titles_path(path)
            if not titles_path.is_file():
                raise FileNotFoundError(f"Title names not found: {titles_path}")
            with gzip.open(titles_path, "rt", encoding="utf-8") as source:
                titles = [json.loads(line) for line in source]
            distinct = int(metadata.get("distinct_titles", len(titles)))
            if distinct != len(titles):
                raise ValueError(
                    f"{titles_path} has {len(titles)} titles, "
                    f"metadata claims {distinct}"
                )
        return cls(title_ids, titles, metadata)

    def __len__(self) -> int:
        return int(self.title_ids.shape[0])

    def title_of(self, row_id: int) -> str:
        if self.titles is None:
            raise RuntimeError("Title names were not loaded (load_titles=False)")
        return self.titles[int(self.title_ids[int(row_id)])]

    def normalized_index(self) -> dict[str, int]:
        """Normalized title → ``title_id``.

        Built lazily and once: 5.2M entries in a Python dict cost hundreds of
        megabytes and are needed only while labelling the dataset; episodes
        then work with integer ids.
        """
        if self._normalized is None:
            if self.titles is None:
                raise RuntimeError("Title names were not loaded (load_titles=False)")
            index: dict[str, int] = {}
            for title_id, title in enumerate(self.titles):
                index.setdefault(normalize_title(title), title_id)
            self._normalized = index
        return self._normalized

    def release_normalized_index(self) -> None:
        """Free the title dictionary once the dataset is labelled."""
        self._normalized = None

    def title_ids_of(self, titles: Iterable[str]) -> list[int]:
        """Ids of the given titles that exist in the corpus at all."""
        index = self.normalized_index()
        found = []
        for title in titles:
            title_id = index.get(normalize_title(title))
            if title_id is not None:
                found.append(title_id)
        return found

    def all_titles_covered(self, titles: Iterable[str]) -> bool:
        """Flag: the full set of gold titles is present in the corpus."""
        index = self.normalized_index()
        titles = list(titles)
        if not titles:
            return False
        return all(normalize_title(title) in index for title in titles)
