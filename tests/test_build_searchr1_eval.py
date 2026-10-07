from __future__ import annotations

import json

import pytest

import build_searchr1_eval as searchr1


def row(**overrides) -> dict:
    sample = {
        "id": "test_0",
        "question": "who got the first nobel prize in physics?",
        "golden_answers": ["Wilhelm Conrad Röntgen", "Roentgen"],
        "data_source": "nq",
    }
    sample.update(overrides)
    return sample


def test_first_answer_is_the_gold_and_the_rest_become_aliases() -> None:
    record = searchr1.convert_sample(row(), "nq")
    assert record["answer"] == "Wilhelm Conrad Röntgen"
    assert record["answer_aliases"] == ["Roentgen"]


def test_single_answer_leaves_aliases_empty() -> None:
    record = searchr1.convert_sample(row(golden_answers=["Paris"]), "nq")
    assert record["answer"] == "Paris"
    assert record["answer_aliases"] == []


def test_supporting_facts_is_absent_not_empty() -> None:
    """An empty list silently gives title EM = 100%; a missing field gives none.

    ``title_metrics`` returns 1.0 on an empty set of gold titles, and
    ``add_eval_fields`` computes title metrics only when the sample has
    ``supporting_facts``. The difference between "no field" and "empty field"
    is the difference between a missing metric and a perfect one.
    """
    record = searchr1.convert_sample(row(), "nq")
    for field in searchr1.FORBIDDEN_FIELDS:
        assert field not in record


def test_id_is_prefixed_by_source_because_ids_collide_between_datasets() -> None:
    # The first row of both nq and popqa is called test_0.
    assert searchr1.convert_sample(row(), "nq")["_id"] == "nq_test_0"
    assert searchr1.convert_sample(row(), "popqa")["_id"] == "popqa_test_0"


def test_blank_aliases_are_dropped_but_a_blank_gold_is_an_error() -> None:
    record = searchr1.convert_sample(row(golden_answers=["Paris", "  ", "paris"]), "nq")
    assert record["answer_aliases"] == ["paris"]
    with pytest.raises(ValueError, match="has no answers"):
        searchr1.convert_sample(row(golden_answers=["", "   "]), "nq")


def test_missing_question_is_an_error() -> None:
    with pytest.raises(ValueError, match="has no question"):
        searchr1.convert_sample(row(question="  "), "nq")


def test_composition_mismatch_is_refused() -> None:
    """Another dataset snapshot means other questions and breaks Search-R1 parity."""
    grouped = {source: [None] * count for source, count in searchr1.EXPECTED_ROWS.items()}
    searchr1.check_composition(grouped)
    grouped["bamboogle"] = [None] * 124
    with pytest.raises(ValueError, match="differs from the expected"):
        searchr1.check_composition(grouped)


def test_written_file_reads_back_through_the_pipeline_loader(tmp_path) -> None:
    rows = [row(), row(id="test_1", question="what?", golden_answers=["x"])]
    path, written = searchr1.write_source(rows, "nq", tmp_path)
    assert written == 2
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert [json.loads(line)["_id"] for line in lines] == ["nq_test_0", "nq_test_1"]
    # verify_output already ran inside write_source; here we check that it
    # really catches a mismatch rather than doing nothing.
    with pytest.raises(RuntimeError, match="re-read"):
        searchr1.verify_output(path, rows[:1])
