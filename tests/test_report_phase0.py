from __future__ import annotations

import json

import pytest

import report_phase0


def judge_record(**overrides) -> dict:
    """Запись судьи, у которой EM и F1 согласованы с текстами.

    Согласованность обязательна: ``score_run`` при заданных алиасах сверяет
    пересчитанные по основному ответу EM и F1 с теми, что лежат в файле, и
    падает при расхождении. Рассогласованная фикстура ловила бы не то.
    """
    record = {
        "id": "nq_test_0",
        "answer": "Roentgen",
        "prediction": "Wilhelm Conrad Rontgen",
        "pred_texts": ["chunk one", "chunk two"],
        "EM": 0,
        "F1": 0.0,
        "LLM_Judge_Score": 1,
    }
    record.update(overrides)
    return record


def matching_record(**overrides) -> dict:
    """Запись, где предсказание дословно совпало с основным ответом."""
    return judge_record(prediction="Roentgen", EM=1, F1=1.0, **overrides)


def write_judge(tmp_path, records) -> "object":
    path = tmp_path / "answer_judge.json"
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


def test_missing_gold_titles_give_null_metrics_not_a_hundred_percent(tmp_path) -> None:
    """Пустой голд у ``title_metrics`` даёт 1.0 — метрика обязана стать None.

    Это тот же капкан, из-за которого заведён build_musique_eval.py: ран без
    gold-титулов иначе отчитался бы стопроцентным попаданием по титулам и
    долей потолка больше единицы.
    """
    path = write_judge(tmp_path, [judge_record(), judge_record(id="nq_test_1")])
    scored = report_phase0.score_run(path, ceiling=None)
    for key in (
        "title_em", "title_recall", "title_em_exact",
        "share_of_ceiling", "em_given_hit", "em_given_miss", "hit_share",
    ):
        assert scored[key] is None, key
    assert scored["em"] == 0.0
    assert scored["samples"] == 2


def test_present_gold_titles_are_still_scored(tmp_path) -> None:
    path = write_judge(
        tmp_path,
        [
            judge_record(gold_titles=["Alpha"], retrieved_titles=["Alpha", "Beta"]),
            judge_record(id="x", gold_titles=["Gamma"], retrieved_titles=["Beta"]),
        ],
    )
    scored = report_phase0.score_run(path, ceiling=0.5)
    assert scored["title_em"] == 0.5
    assert scored["share_of_ceiling"] == 1.0
    assert scored["hit_share"] == 0.5


def test_aliases_add_the_official_definition_of_em(tmp_path) -> None:
    """EM официальных эвалов — максимум по алиасам, судейский — по одному ответу."""
    path = write_judge(tmp_path, [judge_record()])
    aliases = {"nq_test_0": ["Wilhelm Conrad Rontgen", "W. C. Roentgen"]}
    scored = report_phase0.score_run(path, ceiling=None, aliases=aliases)
    # Судья сравнивал с «Roentgen» и не засчитал; алиас совпадает дословно.
    assert scored["em"] == 0.0
    assert scored["em_alias"] == 1.0
    assert scored["with_aliases"] == 1
    assert scored["em_alias_per_sample"] == [1.0]


def test_without_aliases_there_is_no_alias_column(tmp_path) -> None:
    path = write_judge(tmp_path, [judge_record()])
    scored = report_phase0.score_run(path, ceiling=None)
    assert "em_alias" not in scored


def test_empty_alias_table_leaves_alias_em_equal_to_plain_em(tmp_path) -> None:
    path = write_judge(tmp_path, [matching_record()])
    scored = report_phase0.score_run(path, ceiling=None, aliases={})
    assert scored["em_alias"] == scored["em"] == 1.0
    assert scored["with_aliases"] == 0


def test_normalization_drift_from_the_judge_is_refused(tmp_path) -> None:
    """``em`` берётся у судьи, ``em_alias`` считается здесь — разъезд фатален.

    Если наша нормализация разойдётся с судейской, alias-версия окажется ниже
    основной, и главное число таблицы станет неверным беззвучно. Проверяется
    ровно этот случай: запись, где судья засчитал совпадение, а по текстам его
    нет.
    """
    path = write_judge(tmp_path, [judge_record(EM=1, F1=1.0)])
    with pytest.raises(ValueError, match="Нормализация разъехалась"):
        report_phase0.score_run(path, ceiling=None, aliases={})
    # Без алиасов сверять нечего: судейские числа берутся как есть.
    assert report_phase0.score_run(path, ceiling=None)["em"] == 1.0


def test_render_marks_absent_metrics_with_a_dash(tmp_path) -> None:
    path = write_judge(tmp_path, [judge_record()])
    scored = report_phase0.score_run(path, ceiling=None)
    table = report_phase0.render([("no retrieval", scored)], ceiling=None)
    header, _, body = table.partition("\n")
    # Заголовок доли потолка без процента: потолка у датасета нет.
    assert "доля потолка |" in header
    assert "EM по алиасам" not in header
    assert body.split("\n")[-1].count("—") >= 5


def test_render_puts_alias_em_in_bold_when_it_exists(tmp_path) -> None:
    path = write_judge(tmp_path, [judge_record()])
    scored = report_phase0.score_run(
        path, ceiling=None, aliases={"nq_test_0": ["Wilhelm Conrad Rontgen"]}
    )
    table = report_phase0.render([("обученная башня", scored)], ceiling=None)
    header, *rows = table.split("\n")
    assert "EM по алиасам" in header
    # Жирным ровно одна колонка, и это alias-версия: её читают первой.
    assert rows[-1].count("**") == 2
    assert "**100.00**" in rows[-1]


def test_percent_distinguishes_absent_from_not_a_number() -> None:
    assert report_phase0.percent(None) == "—"
    assert report_phase0.percent(float("nan")) == "n/a"
    assert report_phase0.percent(0.1234) == "12.34"


def test_paired_t_sees_a_shift_that_mcnemar_cannot() -> None:
    # Ровно тот случай, ради которого парный t добавлен: вариант лучше на
    # каждом вопросе по F1, но EM не переворачивается ни разу, поэтому
    # дискордантных пар нет и McNemar молчит.
    baseline_em = [0.0] * 40
    variant_em = [0.0] * 40
    baseline_f1 = [0.30] * 40
    variant_f1 = [0.42] * 40
    assert report_phase0.mcnemar(baseline_em, variant_em)["wins"] == 0
    # Постоянный сдвиг — нулевой разброс разностей, t не определён.
    assert report_phase0.paired_t(baseline_f1, variant_f1) != report_phase0.paired_t(
        baseline_f1, [value + 0.01 * index for index, value in enumerate(variant_f1)]
    )
    noisy = [0.42 + (0.01 if index % 2 else -0.01) for index in range(40)]
    assert report_phase0.paired_t(baseline_f1, noisy) > 10


def test_paired_t_separates_no_difference_from_a_perfectly_consistent_one() -> None:
    # Оба случая дают нулевой разброс разностей, но означают противоположное:
    # совпавшие раны — эффекта нет, ровный сдвиг — эффект на каждом вопросе.
    import math

    assert math.isnan(report_phase0.paired_t([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]))
    assert report_phase0.paired_t([0.0, 0.0, 0.0], [0.25, 0.25, 0.25]) == math.inf
    assert report_phase0.paired_t([0.25, 0.25, 0.25], [0.0, 0.0, 0.0]) == -math.inf


def test_paired_t_refuses_runs_of_different_length() -> None:
    with pytest.raises(ValueError):
        report_phase0.paired_t([0.1, 0.2], [0.1])


def test_score_run_keeps_f1_and_judge_per_sample(tmp_path) -> None:
    # Без покандидатных векторов парный t не из чего собрать, а таблица пар
    # молча осталась бы одной колонкой McNemar.
    path = tmp_path / "judge.json"
    path.write_text(
        json.dumps([matching_record(), matching_record()]), encoding="utf-8"
    )
    scored = report_phase0.score_run(path, None, None)
    assert len(scored["f1_per_sample"]) == 2
    assert len(scored["judge_per_sample"]) == 2
    assert scored["f1"] == pytest.approx(sum(scored["f1_per_sample"]) / 2)
