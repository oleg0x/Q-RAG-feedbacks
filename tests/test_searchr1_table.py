from __future__ import annotations

import pytest

import runlib
import searchr1_table as table


def metrics(em_alias: float, em: float | None = None) -> dict:
    return {"em_alias": em_alias, "em": em if em is not None else em_alias}


def full_series(value: float = 0.3) -> dict:
    return {
        dataset: {arm: metrics(value) for arm, _ in table.ARMS}
        for dataset in table.SEARCHR1
    }


def test_every_reference_dataset_is_registered_in_runlib() -> None:
    """Otherwise the row is computed but lands in no results table."""
    assert set(table.SEARCHR1) <= set(runlib.DATASETS)
    assert set(table.SHORT) == set(table.SEARCHR1)


def test_neutral_set_excludes_both_sides_training_data() -> None:
    """Search-R1 was trained on NQ+HotpotQA, line A on HotpotQA+2Wiki.

    Only the four datasets that neither side saw can be neutral; if the list
    drifts from this fact, the "fair part of the table" silently stops being
    fair.
    """
    assert set(table.NEUTRAL) == set(table.SEARCHR1) - {
        "sr1_nq", "sr1_hotpotqa", "sr1_2wiki"
    }


def test_headline_prefers_alias_em_because_that_is_their_definition() -> None:
    assert table.headline({"em_alias": 0.42, "em": 0.31}) == 0.42
    # A dataset without aliases has no em_alias column at all.
    assert table.headline({"em": 0.31}) == 0.31
    assert table.headline({"em": None}) is None


def test_delta_column_compares_against_search_r1(capsys) -> None:
    found = full_series(0.5)
    rendered = table.render_main(found)
    # Search-R1-base has 19.6 on MuSiQue; our row at 50.0 gives +30.4.
    musique = [line for line in rendered.split("\n") if line.startswith("| MuSiQue")][0]
    assert "+30.4" in musique
    assert "**50.0**" in musique


def test_average_is_absent_until_every_dataset_of_the_group_has_a_run() -> None:
    """An average over an incomplete set is not the average but another number.

    Were one dataset skipped silently, the "average EM over seven" would become
    an average over six, off by several points with no sign in the table.
    """
    found = full_series(0.4)
    assert table.average(found, tuple(table.SEARCHR1), "best") == pytest.approx(0.4)
    del found["sr1_bamboogle"]["best"]
    assert table.average(found, tuple(table.SEARCHR1), "best") is None


def test_missing_runs_are_announced_not_silently_dropped() -> None:
    found = full_series()
    del found["sr1_nq"]["zeroshot"]
    built = table.build(found)
    assert "Incomplete data" in built and "NQ/zeroshot" in built
    assert "Incomplete data" not in table.build(full_series())


def test_empty_series_renders_dashes_rather_than_failing() -> None:
    built = table.build({})
    assert "—" in built
    assert "Incomplete data" in built
