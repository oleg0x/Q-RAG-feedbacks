from __future__ import annotations

from pathlib import Path

import runlib
import runs_index


def row(name: str, dataset: str | None, samples: int, status: str = "ok"):
    config = {"label": name}
    if dataset is not None:
        config["dataset"] = dataset
    return (Path("runs") / name, {"status": status, "config": config}, {"samples": samples})


def test_datasets_go_into_separate_groups() -> None:
    groups = runs_index.comparable(
        [
            row("hotpot-a", "hotpotqa_dev_fullwiki", 7405),
            row("2wiki-a", "2wiki_dev", 12576),
            row("musique-a", "musique_ans_dev", 2417),
        ]
    )
    assert list(groups) == ["hotpotqa_dev_fullwiki", "2wiki_dev", "musique_ans_dev"]
    assert [len(members) for members in groups.values()] == [1, 1, 1]


def test_run_without_dataset_counts_as_hotpotqa() -> None:
    # Все раны реестра до 2026-08-05 сделаны на HotpotQA и поля не имеют.
    groups = runs_index.comparable([row("old", None, 7405)])
    assert list(groups) == [runlib.DEFAULT_DATASET]


def test_smoke_run_is_dropped_inside_its_own_dataset() -> None:
    groups = runs_index.comparable(
        [
            row("full-a", "2wiki_dev", 12576),
            row("full-b", "2wiki_dev", 12576),
            row("smoke", "2wiki_dev", 20),
        ]
    )
    assert [run.name for run, _, _ in groups["2wiki_dev"]] == ["full-a", "full-b"]


def test_small_dataset_survives_next_to_a_large_one() -> None:
    # Прежний отбор брал модальное число примеров по всему реестру, поэтому
    # 2 417 вопросов MuSiQue выпадали из витрины рядом с 7 405 HotpotQA.
    groups = runs_index.comparable(
        [
            row("hotpot-a", "hotpotqa_dev_fullwiki", 7405),
            row("hotpot-b", "hotpotqa_dev_fullwiki", 7405),
            row("musique-a", "musique_ans_dev", 2417),
        ]
    )
    assert [run.name for run, _, _ in groups["musique_ans_dev"]] == ["musique-a"]


def test_failed_and_unscored_runs_never_reach_the_table() -> None:
    groups = runs_index.comparable(
        [
            row("ok", "2wiki_dev", 12576),
            row("failed", "2wiki_dev", 12576, status="failed"),
            (Path("runs/unscored"), {"status": "ok", "config": {"dataset": "2wiki_dev"}}, None),
        ]
    )
    assert [run.name for run, _, _ in groups["2wiki_dev"]] == ["ok"]
