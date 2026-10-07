from __future__ import annotations

import json

import pytest

import report_phase0


def judge_record(**overrides) -> dict:
    """Judge record whose EM and F1 agree with the texts.

    Agreement is required: with aliases given, ``score_run`` checks EM and F1
    recomputed against the primary answer against those in the file and fails
    on a mismatch. An inconsistent fixture would test the wrong thing.
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
    """Record whose prediction matches the primary answer verbatim."""
    return judge_record(prediction="Roentgen", EM=1, F1=1.0, **overrides)


def write_judge(tmp_path, records) -> "object":
    path = tmp_path / "answer_judge.json"
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


def test_missing_gold_titles_give_null_metrics_not_a_hundred_percent(tmp_path) -> None:
    """Empty gold makes ``title_metrics`` return 1.0; the metric must be None.

    The same trap that build_musique_eval.py exists for: otherwise a run
    without gold titles would report a 100% title hit rate and a share of
    ceiling above one.
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
    """Official evals take EM as the max over aliases; the judge uses one answer."""
    path = write_judge(tmp_path, [judge_record()])
    aliases = {"nq_test_0": ["Wilhelm Conrad Rontgen", "W. C. Roentgen"]}
    scored = report_phase0.score_run(path, ceiling=None, aliases=aliases)
    # The judge compared against "Roentgen" and said no; an alias matches verbatim.
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
    """``em`` comes from the judge, ``em_alias`` is computed here; drift is fatal.

    If our normalization diverges from the judge's, the alias version ends up
    below the primary one and the headline number of the table becomes
    silently wrong. This tests exactly that case: a record where the judge
    counted a match that the texts do not support.
    """
    path = write_judge(tmp_path, [judge_record(EM=1, F1=1.0)])
    with pytest.raises(ValueError, match="Normalization has diverged"):
        report_phase0.score_run(path, ceiling=None, aliases={})
    # Without aliases there is nothing to cross-check: judge numbers are used as is.
    assert report_phase0.score_run(path, ceiling=None)["em"] == 1.0


def test_render_marks_absent_metrics_with_a_dash(tmp_path) -> None:
    path = write_judge(tmp_path, [judge_record()])
    scored = report_phase0.score_run(path, ceiling=None)
    table = report_phase0.render([("no retrieval", scored)], ceiling=None)
    header, _, body = table.partition("\n")
    # Share-of-ceiling header without a percentage: the dataset has no ceiling.
    assert "share of ceiling |" in header
    assert "alias EM" not in header
    assert body.split("\n")[-1].count("—") >= 5


def test_render_puts_alias_em_in_bold_when_it_exists(tmp_path) -> None:
    path = write_judge(tmp_path, [judge_record()])
    scored = report_phase0.score_run(
        path, ceiling=None, aliases={"nq_test_0": ["Wilhelm Conrad Rontgen"]}
    )
    table = report_phase0.render([("trained tower", scored)], ceiling=None)
    header, *rows = table.split("\n")
    assert "alias EM" in header
    # Exactly one column is bold, and it is the alias version: it is read first.
    assert rows[-1].count("**") == 2
    assert "**100.00**" in rows[-1]


def test_percent_distinguishes_absent_from_not_a_number() -> None:
    assert report_phase0.percent(None) == "—"
    assert report_phase0.percent(float("nan")) == "n/a"
    assert report_phase0.percent(0.1234) == "12.34"


def test_paired_t_sees_a_shift_that_mcnemar_cannot() -> None:
    # Exactly the case the paired t was added for: the variant has better F1
    # on every question, but EM never flips, so there are no discordant pairs
    # and McNemar is silent.
    baseline_em = [0.0] * 40
    variant_em = [0.0] * 40
    baseline_f1 = [0.30] * 40
    variant_f1 = [0.42] * 40
    assert report_phase0.mcnemar(baseline_em, variant_em)["wins"] == 0
    # A constant shift has zero spread of differences, so t is degenerate.
    assert report_phase0.paired_t(baseline_f1, variant_f1) != report_phase0.paired_t(
        baseline_f1, [value + 0.01 * index for index, value in enumerate(variant_f1)]
    )
    noisy = [0.42 + (0.01 if index % 2 else -0.01) for index in range(40)]
    assert report_phase0.paired_t(baseline_f1, noisy) > 10


def test_paired_t_separates_no_difference_from_a_perfectly_consistent_one() -> None:
    # Both cases give zero spread of differences but mean opposite things:
    # identical runs mean no effect, a constant shift an effect on every question.
    import math

    assert math.isnan(report_phase0.paired_t([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]))
    assert report_phase0.paired_t([0.0, 0.0, 0.0], [0.25, 0.25, 0.25]) == math.inf
    assert report_phase0.paired_t([0.25, 0.25, 0.25], [0.0, 0.0, 0.0]) == -math.inf


def test_paired_t_refuses_runs_of_different_length() -> None:
    with pytest.raises(ValueError):
        report_phase0.paired_t([0.1, 0.2], [0.1])


def test_score_run_keeps_f1_and_judge_per_sample(tmp_path) -> None:
    # Without per-sample vectors there is nothing to build a paired t from,
    # and the pairs table would silently keep only the McNemar column.
    path = tmp_path / "judge.json"
    path.write_text(
        json.dumps([matching_record(), matching_record()]), encoding="utf-8"
    )
    scored = report_phase0.score_run(path, None, None)
    assert len(scored["f1_per_sample"]) == 2
    assert len(scored["judge_per_sample"]) == 2
    assert scored["f1"] == pytest.approx(sum(scored["f1_per_sample"]) / 2)
