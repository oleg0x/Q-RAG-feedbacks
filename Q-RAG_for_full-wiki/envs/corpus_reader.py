"""Произвольный доступ к строкам корпуса Wiki-18 по таблице смещений.

Корпус — JSONL на 14 ГБ, а нужны из него на каждом шаге ровно те чанки,
которые выбрала политика. Таблица смещений ``uint64[rows + 1]`` строится
командой ``python fullwiki_qrag.py prepare`` в лаборатории full-wiki; здесь
она только читается.

Дублирование ``WikiCorpus`` из ``full-wiki/fullwiki_qrag.py`` намеренное:
лаборатория и код обучения — разные репозитории с разными зависимостями, и
импорт через границу привязал бы обучение к путям лаборатории. Формат
артефакта общий, и это единственное, о чём нужно договариваться.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class CorpusReader:
    """Чтение строк корпуса по номеру, соответствующему строке матрицы."""

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
                "Build them with `python fullwiki_qrag.py prepare`."
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
        # `pread` вместо `seek` + `read`: чанки читает пул потоков, а пара
        # «переместить курсор, прочитать» на общем дескрипторе не атомарна —
        # два потока подряд отдают строку не того номера, который просили.
        raw = os.pread(self._descriptor(), end - start, start)
        item = json.loads(raw)
        # Строка матрицы обязана совпадать с ``id`` записи: индекс собран по
        # номеру строки JSONL, и рассинхронизация здесь означает, что
        # обучение читает не тот текст, который кодировал энкодер.
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
