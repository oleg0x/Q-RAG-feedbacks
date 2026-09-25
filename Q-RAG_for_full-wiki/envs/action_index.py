"""Матрица действий ``M`` и батчевый поиск по ней с маской до ``topk``.

``M`` — шарды ``wiki18-gte``, 21 015 324 × 768 fp32, 60.1 ГиБ резидентно на
GPU. Action-башня заморожена и побитово равна стоковой GTE, поэтому отдельного
action-индекса не нужно: вектор действия — это сырая строка ``M``, без
RoPE-поворота. Из этого следует тождество, на котором держится вся линия A:
скор поиска ``s @ M[i]`` и оценка ``Q(s, a_i)`` — одно и то же число.

Два инварианта реализации.

**Маска применяется до ``topk``.** Запрос содержит текст уже выбранного чанка,
поэтому соседи этого чанка по статье оказываются ближайшими соседями запроса.
Маскирование после ``topk`` оставило бы пул, целиком забитый одной статьёй.

**Поиск батчевый.** Он упирается в чтение 60 ГиБ из HBM, а не в арифметику:
один запрос стоит 15.9 мс, шестнадцать — 35 мс. Шагать среды циклом значит
платить за каждую отдельно.
"""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import torch
from torch import Tensor


SHARD_PATTERN = re.compile(r"^part-(\d{12})-(\d{12})\.npy$")

SearchResult = namedtuple(
    "SearchResult",
    ["scores", "row_ids", "vectors", "masked_argmax"],
)


def discover_shard_files(shard_dir: Path) -> list[tuple[Path, int, int]]:
    """Отсортированные шарды матрицы с проверкой непрерывности покрытия."""
    shards: list[tuple[Path, int, int]] = []
    for path in sorted(Path(shard_dir).iterdir()):
        match = SHARD_PATTERN.match(path.name)
        if not match:
            if path.name.endswith(".tmp"):
                continue
            raise RuntimeError(f"Unexpected file in shard directory: {path}")
        start, end = map(int, match.groups())
        shards.append((path, start, end))
    expected_start = 0
    for path, start, end in shards:
        if start != expected_start:
            raise RuntimeError(
                f"Non-contiguous shards: expected start={expected_start}, "
                f"got {start} in {path}"
            )
        expected_start = end
    if not shards:
        raise FileNotFoundError(f"No embedding shards found in {shard_dir}")
    return shards


class ActionIndex:
    """Поиск по всем строкам ``M`` одним матричным умножением."""

    def __init__(
        self,
        vectors: Tensor,
        title_ids: Tensor | np.ndarray | None = None,
    ) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"Action matrix must be 2-D, got {tuple(vectors.shape)}")
        self.vectors = vectors
        self.device = vectors.device
        self.title_ids: Tensor | None = None
        if title_ids is not None:
            if isinstance(title_ids, np.ndarray):
                title_ids = torch.from_numpy(np.ascontiguousarray(title_ids))
            if title_ids.shape != (vectors.shape[0],):
                raise ValueError(
                    f"Title table has {tuple(title_ids.shape)} rows, "
                    f"the action matrix has {vectors.shape[0]}"
                )
            self.title_ids = title_ids.to(device=self.device)

    @property
    def num_rows(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @classmethod
    def from_shards(
        cls,
        shard_dir: str | Path,
        *,
        device: str | torch.device,
        expected_rows: int | None = None,
        dim: int = 768,
        title_ids: Tensor | np.ndarray | None = None,
        log_every: int = 25,
    ) -> "ActionIndex":
        """Собрать матрицу на устройстве из шардов ``wiki18-gte``.

        Шарды читаются через ``mmap`` и копируются пошардово: держать в RAM
        ещё одну копию на 60 ГиБ рядом с GPU-версией незачем.
        """
        shards = discover_shard_files(Path(shard_dir).expanduser().resolve())
        rows = shards[-1][2]
        if expected_rows is not None and rows != expected_rows:
            raise RuntimeError(
                f"Shards cover {rows} rows, expected {expected_rows}"
            )
        vectors = torch.empty((rows, dim), dtype=torch.float32, device=device)
        for number, (path, start, end) in enumerate(shards, start=1):
            cached = np.load(path, mmap_mode="r", allow_pickle=False)
            if cached.shape != (end - start, dim):
                raise RuntimeError(f"Shard {path} has shape {cached.shape}")
            staging = torch.from_numpy(np.array(cached, dtype=np.float32, order="C"))
            vectors[start:end].copy_(staging)
            if log_every and (number % log_every == 0 or number == len(shards)):
                print(f"[INFO] action matrix: {number}/{len(shards)} shards")
        return cls(vectors, title_ids)

    def rows(self, row_ids: Tensor | Sequence[int]) -> Tensor:
        """Векторы действий по номерам строк, форма входа сохраняется."""
        if not isinstance(row_ids, Tensor):
            row_ids = torch.as_tensor(row_ids, dtype=torch.long, device=self.device)
        flat = row_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        gathered = self.vectors.index_select(0, flat)
        return gathered.reshape(*row_ids.shape, self.dim)

    @torch.no_grad()
    def search(
        self,
        queries: Tensor,
        top_k: int,
        blocked_rows: Sequence[Sequence[int]] | None = None,
        blocked_titles: Sequence[Sequence[int]] | None = None,
        *,
        with_vectors: bool = True,
    ) -> SearchResult:
        """Top-K по ``queries @ M.T`` с маской, применённой **до** ``topk``.

        ``masked_argmax`` — флаг «глобальный аргмакс запроса был замаскирован»:
        это метрика мониторинга, показывающая, насколько часто политика видит
        не тот максимум, который дал бы поиск без ограничений.
        """
        queries = queries.to(device=self.device, dtype=self.vectors.dtype)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError(f"Queries must be [B, {self.dim}], got {tuple(queries.shape)}")
        batch = queries.shape[0]
        if top_k <= 0:
            raise ValueError(f"top_k must be positive: {top_k}")
        top_k = min(top_k, self.num_rows)

        scores = queries @ self.vectors.T
        free_argmax = scores.argmax(dim=1)

        blocked = torch.zeros_like(scores, dtype=torch.bool)
        if blocked_rows is not None:
            for row, ids in enumerate(blocked_rows):
                if len(ids):
                    blocked[row, torch.as_tensor(
                        list(ids), dtype=torch.long, device=self.device
                    )] = True
        if blocked_titles is not None:
            if self.title_ids is None:
                raise RuntimeError("Title quota needs a title table")
            for row, title_list in enumerate(blocked_titles):
                for title_id in title_list:
                    blocked[row] |= self.title_ids == int(title_id)

        masked_argmax = blocked.gather(1, free_argmax[:, None]).squeeze(1)
        # -inf, а не «минимум минус единица»: замаскированная строка обязана
        # выпасть из topk при любом размере пула, а не оказаться его хвостом.
        scores = scores.masked_fill(blocked, float("-inf"))
        available = int((~blocked).sum(dim=1).min())
        if available < top_k:
            raise RuntimeError(
                f"Only {available} unmasked rows left, top_k={top_k}"
            )
        top_scores, top_ids = torch.topk(scores, top_k, dim=1, largest=True, sorted=True)
        vectors = self.rows(top_ids) if with_vectors else None
        if top_ids.shape != (batch, top_k):
            raise RuntimeError(f"Unexpected search shape: {tuple(top_ids.shape)}")
        return SearchResult(top_scores, top_ids, vectors, masked_argmax)

    def titles_of(self, row_ids: Tensor) -> Tensor:
        if self.title_ids is None:
            raise RuntimeError("Title table is not loaded")
        flat = row_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        return self.title_ids.index_select(0, flat).reshape(row_ids.shape)
