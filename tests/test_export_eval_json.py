from __future__ import annotations

import json

import pytest

import export_eval_json as export


def judged_record(**overrides) -> dict:
    record = {
        "id": "sample-1",
        "question": "who?",
        "answer": "Canada",
        "prediction": "Dominion of Canada",
        "pred_idx": [10, 20],
        "pred_texts": ['"Gold A"\nfirst chunk', '"Gold B"\nsecond chunk'],
        "retrieval_hops": [
            {"step": 0, "selected_idx": 10, "selected_score": 1.5},
            {"step": 1, "selected_idx": 20, "selected_score": 1.25},
        ],
        "gold_titles": ["Gold A", "Gold B"],
        "title_em": 1.0,
        "title_recall": 1.0,
        # F1 не выдуман: "dominion of canada" против "canada" даёт точность
        # 1/3 при полноте 1, то есть ровно 0.5. Судейские числа здесь такие
        # же, как посчитал бы answer_judge_llms.py.
        "EM": 0,
        "F1": 0.5,
        "LLM_Judge_Score": 1.0,
    }
    record.update(overrides)
    return record


def test_chunk_is_split_into_title_and_body() -> None:
    result = export.export_record(judged_record(), [], 0)
    assert result["chunks"][0] == {
        "step": 0,
        "title": "Gold A",
        "text": "first chunk",
        "score": 1.5,
        "row": 10,
    }


def test_alias_lifts_em_while_the_primary_metric_stays() -> None:
    result = export.export_record(judged_record(), ["Dominion of Canada"], 0)
    assert result["metrics"]["em"] == 0
    assert result["metrics"]["em_alias"] == 1
    assert result["metrics"]["f1_alias"] == 1.0
    assert result["gold_aliases"] == ["Dominion of Canada"]


def test_without_aliases_the_two_versions_agree() -> None:
    record = judged_record(prediction="Canada", EM=1, F1=1.0)
    result = export.export_record(record, [], 0)
    assert result["metrics"]["em"] == result["metrics"]["em_alias"] == 1
    assert result["metrics"]["f1"] == result["metrics"]["f1_alias"] == 1.0


def test_disagreement_with_the_judge_is_a_hard_error() -> None:
    # Если наша нормализация разъедется с судейской, alias-метрики станут
    # несравнимы с колонкой EM в таблице — тогда падать, а не публиковать.
    with pytest.raises(RuntimeError, match="EM разошёлся"):
        export.export_record(judged_record(EM=1), [], 0)


def test_f1_disagreement_is_a_hard_error() -> None:
    with pytest.raises(RuntimeError, match="F1 разошёлся"):
        export.export_record(judged_record(F1=0.1), [], 0)


def test_summary_averages_both_versions() -> None:
    records = [
        judged_record(id="a"),
        judged_record(id="b", prediction="Canada", EM=1, F1=1.0),
    ]
    exported = [
        export.export_record(records[0], ["Dominion of Canada"], 0),
        export.export_record(records[1], [], 1),
    ]
    summary = export.summarize(exported)
    assert summary["examples"] == 2
    assert summary["em"] == 0.5
    assert summary["em_alias"] == 1.0
    assert summary["with_aliases"] == 1


def test_id_aliases_table_is_read_by_entity(tmp_path) -> None:
    path = tmp_path / "id_aliases.json"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"Q_id": "Q16", "aliases": ["Canada", "CAN"], "demonyms": []},
                {"Q_id": "Q17", "aliases": [], "demonyms": []},
            ]
        ),
        encoding="utf-8",
    )
    table = export.load_id_aliases(path)
    assert table["Q16"] == ["Canada", "CAN"]
    assert table["Q17"] == []


def test_aliases_come_from_the_sample_when_it_carries_them(tmp_path) -> None:
    dataset = tmp_path / "musique.jsonl"
    dataset.write_text(
        json.dumps({"_id": "x", "question": "q", "answer": "a", "answer_aliases": ["b"]})
        + "\n",
        encoding="utf-8",
    )
    assert export.collect_aliases(dataset, None) == {"x": ["b"]}


def test_aliases_are_resolved_through_answer_id(tmp_path) -> None:
    (tmp_path / "id_aliases.json").write_text(
        json.dumps({"Q_id": "Q16", "aliases": ["Canada"], "demonyms": []}) + "\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "dev.json"
    dataset.write_text(
        json.dumps(
            [
                {"_id": "x", "question": "q", "answer": "a", "answer_id": "Q16"},
                {"_id": "y", "question": "q", "answer": "a"},
            ]
        ),
        encoding="utf-8",
    )
    # Таблица подхватывается по соседству с датасетом, без явного флага.
    assert export.collect_aliases(dataset, None) == {"x": ["Canada"]}
