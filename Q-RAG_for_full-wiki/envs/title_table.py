"""Таблица титулов Wiki-18: строка индекса → идентификатор статьи.

Артефакт собирается скриптом ``full-wiki/build_title_table.py`` (один
потоковый проход по корпусу) и здесь только читается. Нужен он в двух местах:

* квота ``N`` чанков на титул маскирует строки **до** ``topk``, а титул строки
  иначе пришлось бы узнавать чтением ста строк корпуса на каждом шаге;
* флаг «оба gold-титула есть в корпусе» пишется в лог эпизода: полного
  комплекта нет у 38.5% обучающей выборки, и без флага кривая награды
  смешивает нерешаемые примеры с решаемыми.

Формат: ``<output>.npy`` (``int32[rows]``), ``<output>.titles.jsonl.gz`` (по
JSON-строке на титул, номер строки равен ``title_id``) и ``<output>.json``
с метаданными.
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
    """Свести титулы двух дампов Википедии к сравнимому виду.

    Копия ``fullwiki_qrag.normalize_title`` из лаборатории full-wiki: HotpotQA
    хранит титулы дампа 2017 года с HTML-сущностями (``Procter &amp; Gamble``),
    Wiki-18 — уже раскодированные. Строгое равенство занижает покрытие
    gold-титулов примерно на 4.6 п.п. Импортировать оригинал нельзя: это
    отдельный репозиторий, — поэтому определения обязаны совпадать буквально.
    """
    decoded = html.unescape(title)
    folded = unicodedata.normalize("NFKC", decoded).casefold()
    return " ".join(folded.replace("_", " ").split())


class TitleTable:
    """``int32[rows]`` с идентификатором титула каждой строки индекса."""

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
        """Нормализованный титул → ``title_id``.

        Строится лениво и один раз: 5.2 млн строк в питоновском словаре стоят
        сотни мегабайт, а нужны они только на разметке датасета — дальше
        эпизод оперирует целыми идентификаторами.
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
        """Отпустить словарь титулов после разметки датасета."""
        self._normalized = None

    def title_ids_of(self, titles: Iterable[str]) -> list[int]:
        """Идентификаторы титулов, которые вообще есть в корпусе."""
        index = self.normalized_index()
        found = []
        for title in titles:
            title_id = index.get(normalize_title(title))
            if title_id is not None:
                found.append(title_id)
        return found

    def all_titles_covered(self, titles: Iterable[str]) -> bool:
        """Флаг «полный комплект gold-титулов есть в корпусе»."""
        index = self.normalized_index()
        titles = list(titles)
        if not titles:
            return False
        return all(normalize_title(title) in index for title in titles)
