from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import fullwiki_qrag as retrieval


def make_manifest(corpus: Path, rows: int) -> dict:
    return {
        "corpus": {
            "source_path": str(corpus),
            "rows": rows,
            "faiss_id_mapping": "zero-based JSONL row number",
        },
        "encoder": {
            "positions_processor": "none",
        },
        "index": {
            "type": "faiss.IndexFlatIP",
            "metric": "raw_inner_product",
            "normalization": "none",
            "dimension": 768,
        },
    }


def write_tiny_corpus(path: Path) -> list[dict]:
    rows = [
        {"id": "0", "contents": '"First title"\nFirst passage.'},
        {"id": "1", "contents": '"Quoted \\"title\\""\nSecond passage.'},
        {"id": "2", "contents": '"Third title"\nUnicode: Привет.'},
    ]
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def test_corpus_offsets_support_random_duplicate_reads(tmp_path: Path) -> None:
    corpus_path = tmp_path / "wiki.jsonl"
    expected = write_tiny_corpus(corpus_path)
    offsets_path = tmp_path / "offsets.npy"

    offsets = retrieval.build_corpus_offsets(
        corpus_path,
        offsets_path,
        len(expected),
        progress_every=0,
    )
    assert offsets.shape == (len(expected) + 1,)
    assert int(offsets[-1]) == corpus_path.stat().st_size

    corpus = retrieval.WikiCorpus(
        corpus_path,
        offsets_path,
        len(expected),
        auto_prepare=False,
    )
    assert corpus.read_rows([2, 0, 2]) == [
        expected[2],
        expected[0],
        expected[2],
    ]


def test_offsets_reject_changed_corpus(tmp_path: Path) -> None:
    corpus_path = tmp_path / "wiki.jsonl"
    expected = write_tiny_corpus(corpus_path)
    offsets_path = tmp_path / "offsets.npy"
    retrieval.build_corpus_offsets(
        corpus_path,
        offsets_path,
        len(expected),
        progress_every=0,
    )
    with corpus_path.open("a", encoding="utf-8") as destination:
        destination.write('{"id":"3","contents":"new"}\n')

    with pytest.raises(RuntimeError, match="Corpus changed"):
        retrieval.validate_offsets(corpus_path, offsets_path, len(expected))


def test_wiki_title_decodes_quoted_json_title() -> None:
    assert retrieval.wiki_title('"A \\"quoted\\" title"\nPassage') == (
        'A "quoted" title'
    )


def test_extract_state_dict_supports_critic_and_policy() -> None:
    checkpoint = {
        "critic": {
            "state_embed.model.weight": torch.tensor([1.0]),
            "action_embed.model.weight": torch.tensor([2.0]),
        },
        "policy": {
            "state_embed.model.weight": torch.tensor([3.0]),
        },
    }
    critic = retrieval.extract_state_dict(checkpoint, "critic")
    policy = retrieval.extract_state_dict(checkpoint, "policy")
    assert set(critic) == {"model.weight"}
    assert critic["model.weight"].item() == 1.0
    assert policy["model.weight"].item() == 3.0


def test_manifest_validation_rejects_cosine_index(tmp_path: Path) -> None:
    corpus = tmp_path / "wiki.jsonl"
    corpus.write_text("", encoding="utf-8")
    manifest = make_manifest(corpus, 0)
    manifest["index"]["normalization"] = "l2"
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="normalization"):
        retrieval.load_and_validate_manifest(tmp_path)


def test_supporting_fact_texts_and_title_metrics() -> None:
    sample = {
        "_id": "sample",
        "question": "q",
        "answer": "a",
        "supporting_facts": [["Gold A", 1], ["Gold B", 0]],
        "context": [
            ["Gold A", ["a0", "a1"]],
            ["Gold B", ["b0"]],
        ],
    }
    result = retrieval.add_eval_fields(
        sample,
        {
            "pred_idx": [10, 20],
            "pred_texts": [
                '"Gold A"\npassage',
                '"Other"\npassage',
            ],
            "q_values": [0.2, 0.1],
            "retrieval_hops": [],
        },
        0,
        "refresh",
        100,
    )
    assert result["sf_texts"] == ["Gold A a1", "Gold B b0"]
    assert result["sf_idx"] == []
    assert result["supporting_facts"] == sample["supporting_facts"]
    assert result["gold_titles"] == ["Gold A", "Gold B"]
    assert result["retrieved_titles"] == ["Gold A", "Other"]
    assert result["title_recall"] == 0.5
    assert result["title_em"] == 0.0


def test_load_input_samples_supports_json_array_and_jsonl(
    tmp_path: Path,
) -> None:
    expected = [{"question": "q1"}, {"question": "q2"}]
    array_path = tmp_path / "array.json"
    array_path.write_text(json.dumps(expected), encoding="utf-8")
    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text(
        "\n".join(json.dumps(item) for item in expected) + "\n",
        encoding="utf-8",
    )
    assert retrieval.load_input_samples(array_path) == expected
    assert retrieval.load_input_samples(jsonl_path) == expected


def test_fixed_dot_product_can_change_second_hop_choice() -> None:
    candidates = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    first_state = np.array([1.0, 0.0], dtype=np.float32)
    second_state = np.array([0.0, 1.0], dtype=np.float32)
    first_scores = first_state @ candidates.T
    first = int(np.argmax(first_scores))
    second_scores = second_state @ candidates.T
    second_scores[first] = -np.inf
    second = int(np.argmax(second_scores))
    assert (first, second) == (0, 1)
def test_text_memory_has_nonempty_padding_sentinel() -> None:
    # Путь строится от файла теста, а не от cwd: иначе тест проходит только
    # при запуске из /home/a.anokhin/Judge и падает из каталога full-wiki.
    # Берётся Q-RAG_for_full-wiki, а не read-only оригинал: ровно этот `envs`
    # инференс линии A получает через `qrag_repo` конфига, и ровно он едет в
    # публикацию — тест обязан проходить и без соседнего Q-RAG-feedback.
    qrag_repo = Path(__file__).resolve().parent.parent / "Q-RAG_for_full-wiki"
    retrieval.add_qrag_repo_to_path(qrag_repo)
    memories = retrieval.make_text_memories(["question"], [[]], " [SEP] ")
    assert memories[0].available_mask.tolist() == [True]
    assert memories[0].text == "question"
class FakeFirstStage:
    def search(self, texts, top_k):
        return (
            np.array([[0.9, 0.8, 0.7]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int64),
        )


class FakeShards:
    vectors = np.pad(
        np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float32),
        ((0, 0), (0, 766)),
    )

    def rows(self, row_ids):
        return self.vectors[np.asarray(row_ids)]


class ExplodingShards:
    def rows(self, row_ids):
        raise AssertionError("--reranker none must not touch the action shards")


class FakeCorpus:
    def read_rows(self, row_ids):
        return [
            {"id": str(row_id), "contents": f'"Title {row_id}"\ndoc-{row_id}'}
            for row_id in row_ids
        ]


class SplitArticleCorpus:
    """Rows 0 and 1 are two chunks of one article, row 2 is another article.

    Wiki-18 cuts every article into 100-word chunks, so a gold sentence often
    sits in the neighbouring chunk of an article the selection has already
    taken. This corpus is the smallest shape where that matters.
    """

    def read_rows(self, row_ids):
        titles = {0: "Split", 1: "Split", 2: "Other"}
        return [
            {
                "id": str(row_id),
                "contents": f'"{titles[int(row_id)]}"\ndoc-{row_id}',
            }
            for row_id in row_ids
        ]


def make_two_stage(
    *,
    reranker: str = "qrag",
    log_candidates: str = "full",
    shards=None,
    corpus=None,
    max_chunks_per_title: int | None = 1,
) -> retrieval.TwoStageFullWikiQrag:
    runner = object.__new__(retrieval.TwoStageFullWikiQrag)
    runner.first_stage = FakeFirstStage()
    runner._action_shards = FakeShards() if shards is None else shards
    runner.corpus = FakeCorpus() if corpus is None else corpus
    runner.separator = " [SEP] "
    runner.max_chunks_per_title = max_chunks_per_title
    runner.reranker = reranker
    runner.log_candidates = log_candidates
    runner.encode = lambda questions, selected: np.pad(
        np.array([[1.0, 0.0] if not selected[0] else [0.0, 1.0]], dtype=np.float32),
        ((0, 0), (0, 766)),
    )
    return runner


def test_two_stage_fixed_pool_is_reranked_after_state_update() -> None:
    runner = make_two_stage()
    result = runner._retrieve_two_stage(["question"], top_k=3, steps=2, refresh=False)
    assert result[0]["pred_idx"] == [0, 1]
    assert result[0]["pred_texts"] == ['"Title 0"\ndoc-0', '"Title 1"\ndoc-1']
    assert result[0]["reranker"] == "qrag"


def test_reranker_none_keeps_first_stage_order_without_action_shards() -> None:
    # The state encoder would flip the second pick to row 1 under Q-RAG
    # scoring; the baseline must stay on the GTE order instead.
    runner = make_two_stage(reranker="none", shards=ExplodingShards())
    result = runner._retrieve_two_stage(["question"], top_k=3, steps=3, refresh=False)
    assert result[0]["pred_idx"] == [0, 1, 2]
    assert result[0]["q_values"] == pytest.approx([0.9, 0.8, 0.7])
    assert result[0]["reranker"] == "none"


def test_reranker_none_is_prefix_consistent_across_step_budgets() -> None:
    short = make_two_stage(reranker="none", shards=ExplodingShards())._retrieve_two_stage(
        ["question"], top_k=3, steps=1, refresh=False
    )
    long = make_two_stage(reranker="none", shards=ExplodingShards())._retrieve_two_stage(
        ["question"], top_k=3, steps=3, refresh=False
    )
    assert long[0]["pred_idx"][:1] == short[0]["pred_idx"]
    assert long[0]["pred_texts"][:1] == short[0]["pred_texts"]


def test_one_chunk_per_title_skips_the_neighbouring_chunk() -> None:
    # Прежнее поведение --dedupe-titles: титул закрыт первым взятым чанком,
    # поэтому второй хоп уходит на другую статью.
    runner = make_two_stage(
        reranker="none", shards=ExplodingShards(), corpus=SplitArticleCorpus()
    )
    result = runner._retrieve_two_stage(["question"], top_k=3, steps=2, refresh=False)
    assert result[0]["pred_idx"] == [0, 2]


def test_two_chunks_per_title_reach_the_neighbouring_chunk() -> None:
    runner = make_two_stage(
        reranker="none",
        shards=ExplodingShards(),
        corpus=SplitArticleCorpus(),
        max_chunks_per_title=2,
    )
    result = runner._retrieve_two_stage(["question"], top_k=3, steps=3, refresh=False)
    # Квота 2 пускает второй чанк той же статьи, но не третий: после двух
    # чанков «Split» титул закрывается и отбор уходит на «Other».
    assert result[0]["pred_idx"] == [0, 1, 2]


def test_no_limit_per_title_is_pure_first_stage_order() -> None:
    runner = make_two_stage(
        reranker="none",
        shards=ExplodingShards(),
        corpus=SplitArticleCorpus(),
        max_chunks_per_title=None,
    )
    result = runner._retrieve_two_stage(["question"], top_k=3, steps=3, refresh=False)
    assert result[0]["pred_idx"] == [0, 1, 2]


def cli_args(*extra: str):
    return retrieval.parse_args(
        ["retrieve", "--index-dir", "idx", "--qrag-repo", "repo", "--query", "q", *extra]
    )


def test_chunk_quota_defaults_to_the_published_deduplication() -> None:
    # Умолчание обязано воспроизводить опубликованные раны: один чанк на
    # титул, то есть прежнее --dedupe-titles.
    assert cli_args().max_chunks_per_title == 1
    assert retrieval.chunks_per_title(cli_args()) == 1
    assert retrieval.chunks_per_title(cli_args("--max-chunks-per-title", "2")) == 2
    # Выключенная дедупликация — это отсутствие квоты, а не квота в единицу.
    assert retrieval.chunks_per_title(cli_args("--no-dedupe-titles")) is None


def test_reranker_none_does_not_mutate_logged_first_stage_scores() -> None:
    runner = make_two_stage(reranker="none", shards=ExplodingShards())
    result = runner._retrieve_two_stage(["question"], top_k=3, steps=3, refresh=False)
    for hop in result[0]["retrieval_hops"]:
        assert hop["first_stage_scores"] == pytest.approx([0.9, 0.8, 0.7])


def test_log_candidates_controls_hop_payload() -> None:
    dumped = ("candidate_idx", "candidate_scores", "first_stage_candidate_idx")
    full = make_two_stage()._retrieve_two_stage(
        ["question"], top_k=3, steps=2, refresh=False
    )[0]["retrieval_hops"][0]
    assert all(key in full for key in dumped)
    assert list(full) == [
        "step",
        "candidate_idx",
        "candidate_scores",
        "first_stage_candidate_idx",
        "first_stage_scores",
        "selected_idx",
        "selected_score",
    ]

    quiet = make_two_stage(log_candidates="none")._retrieve_two_stage(
        ["question"], top_k=3, steps=2, refresh=False
    )[0]["retrieval_hops"][0]
    assert not any(key in quiet for key in dumped)
    assert quiet["selected_idx"] == 0

    trimmed = retrieval.trim_hop(dict(full), "topk")
    assert len(trimmed["candidate_idx"]) == min(
        retrieval.TOPK_CANDIDATE_LOG, len(full["candidate_idx"])
    )
    assert trimmed["selected_idx"] == full["selected_idx"]


def test_normalize_title_bridges_the_two_wikipedia_dumps() -> None:
    # HotpotQA ships 2017 titles; Wiki-18 stores the decoded 2018 form.
    assert retrieval.normalize_title("Procter &amp; Gamble") == retrieval.normalize_title(
        "Procter & Gamble"
    )
    assert retrieval.normalize_title("The_Beatles") == retrieval.normalize_title(
        "the beatles"
    )
    assert retrieval.normalize_title("  Kiss   &amp;  Tell ") == "kiss & tell"


def test_title_metrics_report_both_exact_and_normalized() -> None:
    sample = {
        "_id": "sample",
        "question": "q",
        "answer": "a",
        "supporting_facts": [["Procter &amp; Gamble", 0], ["Gold B", 0]],
        "context": [["Procter &amp; Gamble", ["p0"]], ["Gold B", ["b0"]]],
    }
    result = retrieval.add_eval_fields(
        sample,
        {
            "pred_idx": [10, 20],
            "pred_texts": ['"Procter & Gamble"\npassage', '"Gold B"\npassage'],
            "q_values": [0.2, 0.1],
            "retrieval_hops": [],
        },
        0,
        "fixed",
        100,
    )
    # Exact matching sees a miss that never happened.
    assert result["title_em_exact"] == 0.0
    assert result["title_recall_exact"] == 0.5
    assert result["title_em"] == 1.0
    assert result["title_recall"] == 1.0
def test_query_length_defaults_to_the_corpus_encoding_length() -> None:
    # Умолчание обязано повторять поведение опубликованных ранов: длина
    # запроса берётся из манифеста индекса, а не задаётся числом в коде.
    assert retrieval.resolve_query_length(256, None) == 256
    assert retrieval.resolve_query_length(256, 1024) == 1024


def test_query_length_rejects_a_nonpositive_budget() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        retrieval.resolve_query_length(256, 0)


def test_retrieve_cli_leaves_query_length_unset_by_default() -> None:
    args = retrieval.parse_args(
        [
            "retrieve",
            "--index-dir", "idx",
            "--qrag-repo", "repo",
            "--query", "q",
        ]
    )
    assert args.first_stage_query_length is None
    args = retrieval.parse_args(
        [
            "retrieve",
            "--index-dir", "idx",
            "--qrag-repo", "repo",
            "--query", "q",
            "--first-stage-query-length", "1024",
        ]
    )
    assert args.first_stage_query_length == 1024


def test_refresh_query_grows_with_every_selected_chunk() -> None:
    # Ровно та строка, которую обрезает лимит в 256 токенов; тест фиксирует,
    # что обрезать действительно есть что.
    runner = make_two_stage(reranker="none", shards=ExplodingShards())
    assert runner._state_texts(["question"], [[]]) == ["question"]
    assert runner._state_texts(["question"], [["chunk-1", "chunk-2"]]) == [
        "question [SEP] chunk-1 [SEP] chunk-2"
    ]


def test_torch_candidate_backend_returns_exact_topk() -> None:
    stage = object.__new__(retrieval.GteFirstStage)
    stage.backend = "torch-cuda"
    stage.ntotal = 3
    stage.device = "cpu"
    stage.index = None
    stage.torch_vectors = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float32
    )
    stage.encode = lambda texts: np.array([[1.0, 0.25]], dtype=np.float32)
    scores, row_ids = stage.search(["q"], top_k=2)
    assert row_ids.tolist() == [[0, 2]]
    np.testing.assert_allclose(scores, [[1.0, 0.625]])
