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
    """Иначе строка посчитается, но не попадёт ни в одну таблицу витрины."""
    assert set(table.SEARCHR1) <= set(runlib.DATASETS)
    assert set(table.SHORT) == set(table.SEARCHR1)


def test_neutral_set_excludes_both_sides_training_data() -> None:
    """Search-R1 учился на NQ+HotpotQA, линия A — на HotpotQA+2Wiki.

    Нейтральными могут быть только те четыре, которых не видел никто; если
    список разъедется с этим фактом, «честная часть таблицы» перестанет быть
    честной молча.
    """
    assert set(table.NEUTRAL) == set(table.SEARCHR1) - {
        "sr1_nq", "sr1_hotpotqa", "sr1_2wiki"
    }


def test_headline_prefers_alias_em_because_that_is_their_definition() -> None:
    assert table.headline({"em_alias": 0.42, "em": 0.31}) == 0.42
    # У датасета без алиасов колонки em_alias нет вовсе.
    assert table.headline({"em": 0.31}) == 0.31
    assert table.headline({"em": None}) is None


def test_delta_column_compares_against_search_r1(capsys) -> None:
    found = full_series(0.5)
    rendered = table.render_main(found)
    # MuSiQue у Search-R1-base 19.6; наша строка 50.0 даёт +30.4.
    musique = [line for line in rendered.split("\n") if line.startswith("| MuSiQue")][0]
    assert "+30.4" in musique
    assert "**50.0**" in musique


def test_average_is_absent_until_every_dataset_of_the_group_has_a_run() -> None:
    """Среднее по неполному набору — не среднее, а другое число.

    Пропусти оно один датасет молча, и «средний EM по семи» стал бы средним по
    шести, отличаясь на несколько пунктов без единого признака в таблице.
    """
    found = full_series(0.4)
    assert table.average(found, tuple(table.SEARCHR1), "best") == pytest.approx(0.4)
    del found["sr1_bamboogle"]["best"]
    assert table.average(found, tuple(table.SEARCHR1), "best") is None


def test_missing_runs_are_announced_not_silently_dropped() -> None:
    found = full_series()
    del found["sr1_nq"]["zeroshot"]
    built = table.build(found)
    assert "Неполные данные" in built and "NQ/zeroshot" in built
    assert "Неполные данные" not in table.build(full_series())


def test_empty_series_renders_dashes_rather_than_failing() -> None:
    built = table.build({})
    assert "—" in built
    assert "Неполные данные" in built
