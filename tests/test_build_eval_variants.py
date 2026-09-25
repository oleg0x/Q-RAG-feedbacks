from __future__ import annotations

import json
from pathlib import Path

import pytest

import build_eval_variants as variants


def gte_record(**overrides) -> dict:
    record = {
        "id": "sample-1",
        "question": "q",
        "answer": "a",
        "reranker": "none",
        "retrieval_mode": "fixed",
        "pred_idx": [10, 20, 30],
        "pred_texts": [
            '"Gold A"\nfirst',
            '"Other"\nsecond',
            '"Gold B"\nthird',
        ],
        "q_values": [0.9, 0.8, 0.7],
        "retrieval_hops": [{"step": 0}, {"step": 1}, {"step": 2}],
        "gold_titles": ["Gold A", "Gold B"],
    }
    record.update(overrides)
    return record


def test_truncate_keeps_a_prefix_and_rescores_titles() -> None:
    result = variants.build([gte_record()], "truncate", 2)[0]
    assert result["pred_idx"] == [10, 20]
    assert result["q_values"] == [0.9, 0.8]
    assert len(result["retrieval_hops"]) == 2
    # Gold B dropped with the third chunk, so the metrics must follow.
    assert result["retrieved_titles"] == ["Gold A", "Other"]
    assert result["title_recall"] == 0.5
    assert result["title_em"] == 0.0
    assert result["eval_variant"] == "truncate-2"


def test_truncate_refuses_a_qrag_run() -> None:
    # Q-RAG re-ranks against an updated state, so its 2-step selection is not
    # a prefix of its 6-step selection and truncation would fabricate a run.
    with pytest.raises(ValueError, match="prefix-consistent"):
        variants.build([gte_record(reranker="qrag")], "truncate", 2)


def test_truncate_refuses_a_refresh_run() -> None:
    with pytest.raises(ValueError, match="refresh-mode"):
        variants.build([gte_record(retrieval_mode="refresh")], "truncate", 2)


def test_truncate_refuses_to_extend() -> None:
    with pytest.raises(ValueError, match="record has 3"):
        variants.build([gte_record()], "truncate", 4)


def test_empty_context_scores_zero_without_touching_the_answer() -> None:
    result = variants.build([gte_record()], "none", None)[0]
    assert result["pred_texts"] == []
    assert result["retrieval_hops"] == []
    assert result["title_recall"] == 0.0
    assert result["title_em"] == 0.0
    assert result["answer"] == "a"


def test_oracle_uses_the_supplied_gold_source() -> None:
    gold = {"sample-1": ["Gold A first sentence", "Gold B third sentence"]}
    result = variants.build([gte_record()], "oracle", None, gold)[0]
    assert result["pred_texts"] == gold["sample-1"]
    assert result["title_em"] == 1.0
    # sf_texts are "Title sentence", not '"Title"\npassage', so titles come
    # from gold_titles rather than from parsing the context.
    assert result["retrieved_titles"] == ["Gold A", "Gold B"]


def test_oracle_rejects_a_missing_id() -> None:
    with pytest.raises(ValueError, match="no sample with id"):
        variants.build([gte_record()], "oracle", None, {})


def test_load_gold_sentences_rejects_a_file_without_recoverable_facts(
    tmp_path: Path,
) -> None:
    # hotpot_dev_fullwiki_v1.json stores retrieved paragraphs, not gold ones,
    # so most of its supporting facts cannot be resolved to a sentence.
    path = tmp_path / "fullwiki-like.json"
    path.write_text(
        json.dumps(
            [
                {
                    "_id": "x",
                    "question": "q",
                    "answer": "a",
                    "supporting_facts": [["Gold A", 0]],
                    "context": [["Unrelated", ["s0"]]],
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot serve as the oracle source"):
        variants.load_gold_sentences(path)


def test_keep_only_rescores() -> None:
    record = gte_record(gold_titles=["Gold A", "Gold B"])
    result = variants.build([record], "keep", None)[0]
    assert result["pred_idx"] == record["pred_idx"]
    assert result["title_em"] == 1.0
    assert result["eval_variant"] == "keep"
