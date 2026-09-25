"""
Shared dataset wrappers for ``scripts/eval_*.py`` (MuSiQue and future eval loaders).

MuSiQue format matches ``envs/qa_dataset_adapter.py`` (``source == 'musique'``):
paragraphs with ``idx`` / ``is_supporting``, and ``question_decomposition`` with
``paragraph_support_idx`` pointing at paragraph ``idx`` values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def musique_references_idx_from_item(item: dict[str, Any]) -> list[int]:
    """Golden reference chunk positions in decomposition order (unique by paragraph)."""
    paras = item["paragraphs"]
    pos_by_idx = {p["idx"]: i for i, p in enumerate(paras)}
    references_idx: list[int] = []
    seen_ref: set[int] = set()
    for step in item["question_decomposition"]:
        pid = step["paragraph_support_idx"]
        pos = pos_by_idx.get(pid)
        if pos is not None and pos not in seen_ref:
            references_idx.append(pos)
            seen_ref.add(pos)
    return references_idx


def retrieval_slot_count(dataset: str, n_golden_refs: int, cli_n_select: int) -> int:
    """
    How many chunks the mocked retriever selects (``n_select``).

    For *hotpotqa* and *musique*, this equals the number of golden reference
    positions (``len(references_idx)``) so ``none`` / ``first`` / ``second`` /
    ``both`` always use a full context of that size (gold subset + distractors).

    For *babilong*, returns *cli_n_select* from the command line.
    """
    if dataset in ("hotpotqa", "musique", "hotpotqa_candidate", "musique_candidate"):
        return n_golden_refs
    return cli_n_select


class MusiqueQADataset:
    """
    MuSiQue ``musique_ans_*.jsonl`` → same keys as Hotpot-style eval helpers expect:

    - ``chunks``: one string per paragraph (document order in the JSON array).
    - ``references_idx``: list positions of supporting paragraphs in decomposition
      order (first hop, second hop, …), unique by first occurrence.
    - ``facts_idx``: positions of paragraphs marked ``is_supporting`` (gold pool
      for hard negatives), union with supporting positions if needed.
    """

    def __init__(self, jsonl_path: str | Path, n_refs_exactly: int | None = None):
        path = Path(jsonl_path)
        self.rows: list[dict[str, Any]] = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.rows.append(json.loads(line))
        if n_refs_exactly is not None and n_refs_exactly > 0:
            before = len(self.rows)
            self.rows = [
                r for r in self.rows
                if len(musique_references_idx_from_item(r)) == n_refs_exactly
            ]
            self._musique_filter = (n_refs_exactly, before, len(self.rows))
        else:
            self._musique_filter = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.rows[idx]
        paras = item["paragraphs"]
        chunks = [f"{p['title']}. {p['paragraph_text']}" for p in paras]

        references_idx = list(musique_references_idx_from_item(item))

        facts_idx = [i for i, p in enumerate(paras) if p.get("is_supporting")]
        if not facts_idx:
            facts_idx = list(references_idx)
        else:
            facts_idx = sorted(set(facts_idx) | set(references_idx))

        return {
            "question": item["question"],
            "answer": item["answer"],
            "chunks": chunks,
            "references_idx": references_idx,
            "facts_idx": facts_idx,
        }


def load_musique_for_eval(args: Any) -> MusiqueQADataset:
    """Build ``MusiqueQADataset`` using ``args.musique_path`` and optional ref-count filter."""
    nr = int(getattr(args, "musique_n_refs_exactly", 0) or 0)
    return MusiqueQADataset(
        args.musique_path,
        n_refs_exactly=nr if nr > 0 else None,
    )


def log_musique_filter_if_any(logger_instance: Any, dataset: Any) -> None:
    """Log one line when ``MusiqueQADataset`` applied ``n_refs_exactly`` filtering."""
    info = getattr(dataset, "_musique_filter", None)
    if info is None:
        return
    n, before, after = info
    logger_instance.info(
        "MuSiQue: exactly %d golden reference(s): %d → %d samples after filter",
        n,
        before,
        after,
    )
