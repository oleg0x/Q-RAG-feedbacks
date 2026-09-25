"""Юниты режима прямого поиска ``fullwiki_qrag.py search``.

Проверяется именно то, чем он отличается от ``retrieve``: пул задаёт сам
поиск, маска применяется до ``topk``, квота считается по таблице титулов, а
не по чтению корпуса, и запись остаётся того же формата, что у ``retrieve``.
Матрица здесь игрушечная — настоящая весит 60 ГиБ и живёт на GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import fullwiki_qrag as retrieval


# Три статьи: строки 0-1 — «Split», 2-3 — «Other», 4 — «Third». Wiki-18 режет
# статьи по 100 слов, поэтому соседние строки одного титула — обычное дело, и
# именно на них держится смысл квоты.
TITLE_IDS = torch.tensor([0, 0, 1, 1, 2], dtype=torch.int32)
MATRIX = torch.tensor(
    [
        [1.00, 0.00],
        [0.99, 0.10],
        [0.00, 1.00],
        [0.10, 0.99],
        [0.70, 0.70],
    ],
    dtype=torch.float32,
)


class ToyCorpus:
    TITLES = {0: "Split", 1: "Split", 2: "Other", 3: "Other", 4: "Third"}

    def read_rows(self, row_ids):
        return [
            {
                "id": str(row_id),
                "contents": f'"{self.TITLES[int(row_id)]}"\ndoc-{int(row_id)}',
            }
            for row_id in row_ids
        ]


def make_retriever(
    states,
    *,
    max_chunks_per_title: int | None = 2,
    log_candidates: str = "none",
) -> retrieval.DirectSearchRetriever:
    """Ретривер с подменённым кодировщиком: `states[step]` — вектор запроса."""
    runner = object.__new__(retrieval.DirectSearchRetriever)
    runner.matrix = MATRIX
    runner.title_ids = TITLE_IDS
    runner.corpus = ToyCorpus()
    runner.device = "cpu"
    runner.max_chunks_per_title = max_chunks_per_title
    runner.log_candidates = log_candidates
    runner.separator = " [SEP] "
    runner.max_state_segment_length = 256
    calls = {"step": 0}

    def encode(questions, selected_texts):
        vector = states[min(calls["step"], len(states) - 1)]
        calls["step"] += 1
        return np.asarray([vector] * len(questions), dtype=np.float32)

    runner.encode = encode
    return runner


def test_query_ranks_the_whole_matrix_by_inner_product() -> None:
    runner = make_retriever([[1.0, 0.0]])
    scores, ids = runner.search(
        np.asarray([[1.0, 0.0]], dtype=np.float32), 5, [[]], [[]]
    )

    expected = (MATRIX @ torch.tensor([1.0, 0.0])).numpy()
    assert ids[0].tolist() == list(np.argsort(-expected, kind="stable"))
    assert scores[0] == pytest.approx(sorted(expected, reverse=True), abs=1e-6)


def test_taken_rows_are_masked_before_topk() -> None:
    runner = make_retriever([[1.0, 0.0]])
    _, ids = runner.search(
        np.asarray([[1.0, 0.0]], dtype=np.float32), 3, [[0, 1]], [[]]
    )

    assert 0 not in ids[0].tolist()
    assert 1 not in ids[0].tolist()
    assert len(set(ids[0].tolist())) == 3


def test_masked_rows_never_pad_the_pool_up_to_k() -> None:
    """Пул из K строк обязан состоять из доступных, иначе лог кандидатов врёт."""
    runner = make_retriever([[1.0, 0.0]])

    with pytest.raises(RuntimeError, match="unmasked rows"):
        runner.search(np.asarray([[1.0, 0.0]], dtype=np.float32), 4, [[0, 1]], [[]])


def test_exhausted_titles_are_masked_before_topk() -> None:
    runner = make_retriever([[1.0, 0.0]])
    _, ids = runner.search(
        np.asarray([[1.0, 0.0]], dtype=np.float32), 3, [[]], [[0]]
    )

    assert TITLE_IDS[ids[0]].tolist() == [1, 1, 2] or 0 not in TITLE_IDS[ids[0]].tolist()


def test_quota_two_lets_the_second_chunk_of_an_article_through() -> None:
    """Квота N=2, а не «титул уже взят»: gold-предложение часто во втором чанке."""
    runner = make_retriever([[1.0, 0.0]])

    result = runner.retrieve(["question"], top_k=3, steps=3)

    assert result[0]["pred_idx"] == [0, 1, 4]
    titles = [text.split("\n")[0].strip('"') for text in result[0]["pred_texts"]]
    assert titles == ["Split", "Split", "Third"]


def test_quota_one_closes_the_article_on_the_first_chunk() -> None:
    runner = make_retriever([[1.0, 0.0]], max_chunks_per_title=1)

    result = runner.retrieve(["question"], top_k=2, steps=3)

    titles = [text.split("\n")[0].strip('"') for text in result[0]["pred_texts"]]
    assert titles == ["Split", "Third", "Other"]


def test_episode_never_repeats_a_row() -> None:
    runner = make_retriever([[1.0, 0.0]], max_chunks_per_title=None)

    # Пул из одной строки: матрица игрушечная, а маска съедает её за пять шагов.
    result = runner.retrieve(["question"], top_k=1, steps=5)

    assert sorted(result[0]["pred_idx"]) == [0, 1, 2, 3, 4]


def test_the_pool_follows_the_state_between_hops() -> None:
    """Пул на каждом шаге считается заново — это и отличает прямой поиск."""
    runner = make_retriever([[1.0, 0.0], [0.0, 1.0]], max_chunks_per_title=None)

    result = runner.retrieve(["question"], top_k=3, steps=2)

    assert result[0]["pred_idx"] == [0, 2]


def test_record_matches_the_retrieve_format() -> None:
    runner = make_retriever([[1.0, 0.0]], log_candidates="full")

    result = runner.retrieve(["question"], top_k=3, steps=2)[0]
    record = retrieval.add_eval_fields(
        {
            "_id": "id-0",
            "question": "question",
            "answer": "answer",
            "supporting_facts": [["Split", 0], ["Third", 1]],
        },
        result,
        0,
        "direct",
        3,
    )

    assert record["retrieval_mode"] == "direct"
    assert record["candidate_pool_size"] == 3
    assert record["pred_idx"] == result["pred_idx"]
    assert record["retrieved_titles"] == ["Split", "Split"]
    assert record["gold_titles"] == ["Split", "Third"]
    assert record["title_recall"] == pytest.approx(0.5)
    hop = record["retrieval_hops"][0]
    assert list(hop) == ["step", "candidate_idx", "candidate_scores", "selected_idx", "selected_score"]
    assert len(hop["candidate_idx"]) == 3


def test_direct_run_is_not_prefix_consistent_and_says_so() -> None:
    """`build_eval_variants --context truncate` обязан отказаться от такого рана.

    Каждый шаг переранжирует по обновлённому состоянию, поэтому двухшаговый
    выбор не является префиксом шестишагового.
    """
    import build_eval_variants as variants

    runner = make_retriever([[1.0, 0.0]])
    result = runner.retrieve(["question"], top_k=3, steps=2)[0]
    record = {"retrieval_mode": "direct", **result}

    with pytest.raises(ValueError, match="reranker"):
        variants.build([record], "truncate", 1)


def test_matrix_manifest_validation_rejects_the_qrag_index(tmp_path) -> None:
    """Матрицей действий служит нормированный GTE, а не Q-RAG-индекс."""
    import json

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "index": {"metric": "raw_inner_product", "dimension": 768},
                "corpus": {"faiss_id_mapping": "zero-based JSONL row number"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="index.metric"):
        retrieval.load_and_validate_matrix_manifest(tmp_path)


def test_title_table_length_must_match_the_matrix(tmp_path) -> None:
    path = tmp_path / "corpus-title-ids.npy"
    np.save(path, np.arange(4, dtype=np.int32), allow_pickle=False)

    with pytest.raises(ValueError, match="rows"):
        retrieval.load_title_ids(path, 5, "cpu")

    assert retrieval.load_title_ids(path, 4, "cpu").tolist() == [0, 1, 2, 3]
