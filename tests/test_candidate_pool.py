from __future__ import annotations

import json
from pathlib import Path

import pytest

import candidate_pool as pool


def chunk(title: str, body: str = "body") -> str:
    return f'"{title}"\n{body}'


def diag_record(**overrides) -> dict:
    """A run A record: the whole pool is logged in the first hop."""
    record = {
        "id": "sample-1",
        "question": "q",
        "answer": "a",
        "reranker": "none",
        "retrieval_mode": "fixed",
        "gold_titles": ["Gold A", "Gold B"],
        "pred_idx": [10, 11],
        "pred_texts": [chunk("Noise 0"), chunk("Noise 1")],
        "retrieval_hops": [
            {
                "step": 0,
                "first_stage_candidate_idx": [10, 11, 12, 13],
                "first_stage_scores": [0.9, 0.8, 0.7, 0.6],
            },
            {"step": 1, "first_stage_candidate_idx": [10, 11, 12, 13]},
        ],
    }
    record.update(overrides)
    return record


class FakeCorpus:
    """Row ID -> chunk, like WikiCorpus but without a corpus on disk."""

    def __init__(self, rows: dict[int, str]) -> None:
        self.rows = rows

    def read_rows(self, row_ids):
        return [{"id": str(row), "contents": self.rows[int(row)]} for row in row_ids]


# Pool: two noise chunks on top, both gold chunks below. Exactly the case the
# oracle exists for: a reranker has to reach down the ranking.
POOL = FakeCorpus(
    {
        10: chunk("Noise 0"),
        11: chunk("Noise 1"),
        12: chunk("Gold A", "Alpha lives in Paris."),
        13: chunk("Gold B", "Beta was born in Rome."),
    }
)


# --------------------------------------------------------------------------
# pool


def test_pool_comes_from_the_first_stage_dump_of_the_first_hop() -> None:
    assert pool.pool_row_ids(diag_record()) == [10, 11, 12, 13]


def test_pool_refuses_a_run_logged_without_candidates() -> None:
    # The canonical runs were logged with --log-candidates none, so the error
    # must name the flag instead of failing with a KeyError.
    record = diag_record(retrieval_hops=[{"step": 0, "selected_idx": 10}])
    with pytest.raises(ValueError, match="--log-candidates full"):
        pool.pool_row_ids(record)


# --------------------------------------------------------------------------
# selection


def test_oracle_pulls_both_gold_titles_above_higher_ranked_noise() -> None:
    titles = ["Noise 0", "Noise 1", "Gold A", "Gold B"]
    assert pool.select_positions(titles, ["Gold A", "Gold B"], 2) == [2, 3]


def test_oracle_fills_the_remaining_budget_by_gte_rank() -> None:
    titles = ["Noise 0", "Noise 1", "Gold A", "Gold B"]
    assert pool.select_positions(titles, ["Gold A", "Gold B"], 4) == [2, 3, 0, 1]


def test_oracle_selection_is_prefix_consistent_across_budgets() -> None:
    titles = ["Noise 0", "Noise 1", "Gold A", "Gold B"]
    wide = pool.select_positions(titles, ["Gold A", "Gold B"], 4)
    assert wide[:2] == pool.select_positions(titles, ["Gold A", "Gold B"], 2)


def test_oracle_deduplicates_titles_when_filling() -> None:
    # Wiki-18 holds several chunks per article; deduplication is on in all
    # published runs, so filling the budget must respect it.
    titles = ["Noise 0", "Noise 0", "Noise 1", "Gold A"]
    assert pool.select_positions(titles, ["Gold A"], 3) == [3, 0, 2]


def test_oracle_matches_titles_across_the_two_wikipedia_dumps() -> None:
    titles = ["Noise", "Procter & Gamble"]
    assert pool.select_positions(titles, ["Procter &amp; Gamble"], 1) == [1]


def test_oracle_refuses_a_budget_larger_than_the_pool_can_offer() -> None:
    with pytest.raises(ValueError, match="only 2 chunks at 1 per title"):
        pool.select_positions(["A", "A", "B"], ["A"], 3)
    # With a quota of 2 the same three chunks do fill the budget.
    assert pool.select_positions(["A", "A", "B"], ["A"], 3, max_per_title=2) == [
        0, 1, 2
    ]


def test_missing_gold_falls_back_to_the_top_of_the_ranking() -> None:
    titles = ["Noise 0", "Noise 1"]
    assert pool.select_positions(titles, ["Absent"], 2) == [0, 1]


# --------------------------------------------------------------------------
# chunk-aware selection

# A Wiki-18 article is cut every 100 words, so its gold sentences spread over
# neighbouring chunks: "Gold A" has the first in chunk 1 and the second in
# chunk 2, while "Gold B" is entirely in one chunk and has another without
# gold text.
SPLIT_TITLES = ["Noise", "Gold A", "Gold A", "Gold B", "Gold B"]
SPLIT_TEXTS = [
    chunk("Noise", "Nothing relevant here."),
    chunk("Gold A", "Alpha lives in Paris."),
    chunk("Gold A", "Alpha later moved to Rome."),
    chunk("Gold B", "Beta was born in Rome."),
    chunk("Gold B", "Beta likes cats."),
]
SPLIT_SENTENCES = {
    "gold a": ["Alpha lives in Paris.", "Alpha later moved to Rome."],
    "gold b": ["Beta was born in Rome."],
}


def split_penalties() -> list[int]:
    return pool.sentence_penalties(
        [pool.normalize_title(title) for title in SPLIT_TITLES],
        SPLIT_TEXTS,
        SPLIT_SENTENCES,
    )


def test_penalties_rank_chunks_by_how_much_gold_text_they_hold() -> None:
    # "Gold A" has two gold sentences, one per chunk: both chunks are partial.
    # "Gold B" has a full first chunk and an empty second one.
    assert split_penalties() == [
        pool.VERDICT_PENALTY["none"],
        pool.VERDICT_PENALTY["partial"],
        pool.VERDICT_PENALTY["partial"],
        pool.VERDICT_PENALTY["full"],
        pool.VERDICT_PENALTY["none"],
    ]


def test_verdict_needs_every_gold_sentence_of_the_title() -> None:
    # One of the two facts stayed in the neighbouring chunk, so the reader
    # answers from incomplete context and this is not 'full'.
    assert pool.chunk_verdict(SPLIT_SENTENCES["gold a"], SPLIT_TEXTS[1]) == "partial"
    assert pool.chunk_verdict(SPLIT_SENTENCES["gold b"], SPLIT_TEXTS[3]) == "full"
    assert pool.chunk_verdict(SPLIT_SENTENCES["gold b"], SPLIT_TEXTS[0]) == "none"


def test_chunk_aware_oracle_skips_a_higher_ranked_chunk_without_the_sentence() -> None:
    titles = ["Noise", "Gold B", "Gold B"]
    texts = [
        chunk("Noise", "Nothing relevant here."),
        chunk("Gold B", "Beta likes cats."),
        chunk("Gold B", "Beta was born in Rome."),
    ]
    penalties = pool.sentence_penalties(
        [pool.normalize_title(title) for title in titles], texts, SPLIT_SENTENCES
    )
    # The title oracle would take chunk 1, which ranks higher in GTE. The
    # chunk-aware one goes down to 2 because the gold sentence is there.
    assert pool.select_positions(titles, ["Gold B"], 1) == [1]
    assert pool.select_positions(titles, ["Gold B"], 1, penalties=penalties) == [2]


def test_chunk_aware_oracle_falls_back_to_rank_when_no_chunk_has_the_sentence() -> None:
    titles = ["Gold B", "Gold B"]
    texts = [chunk("Gold B", "Nothing"), chunk("Gold B", "Also nothing")]
    penalties = pool.sentence_penalties(
        [pool.normalize_title(title) for title in titles], texts, SPLIT_SENTENCES
    )
    assert pool.select_positions(titles, ["Gold B"], 1, penalties=penalties) == [0]


def test_second_chunk_is_taken_only_when_it_carries_gold_text() -> None:
    penalties = split_penalties()
    # Budget 4, quota 2: one chunk per title, then the second chunk of "Gold A"
    # (it holds the second gold sentence), and only then filling by rank.
    # The second chunk of "Gold B" carries no gold text and is not in a round.
    assert pool.select_positions(
        SPLIT_TITLES, ["Gold A", "Gold B"], 4, penalties=penalties, max_per_title=2
    ) == [1, 3, 2, 0]


def test_chunk_aware_selection_stays_prefix_consistent() -> None:
    penalties = split_penalties()
    wide = pool.select_positions(
        SPLIT_TITLES, ["Gold A", "Gold B"], 4, penalties=penalties, max_per_title=2
    )
    assert wide[:2] == pool.select_positions(
        SPLIT_TITLES, ["Gold A", "Gold B"], 2, penalties=penalties, max_per_title=2
    )


# --------------------------------------------------------------------------
# gold ranks


def test_gold_ranks_are_sorted_and_flag_the_absent_title() -> None:
    titles = ["Noise 0", "Gold B", "Noise 1", "Gold A"]
    assert pool.gold_ranks(titles, ["Gold A", "Gold B"]) == [1, 3]
    # An absent title goes last: the "second gold" is the worst of the found
    # ones, not the second in the annotation.
    assert pool.gold_ranks(titles, ["Gold B", "Absent"]) == [1, pool.MISSING_RANK]


def test_histogram_buckets_are_one_based_and_count_every_rank() -> None:
    ranks = [0, 1, 4, 9, 99, pool.MISSING_RANK]
    counts = pool.histogram(ranks, (1, 5, 10, 100))
    assert counts["1-1"] == 1
    assert counts["2-5"] == 2
    assert counts["6-10"] == 1
    assert counts["11-100"] == 1
    assert counts["missing"] == 1
    assert sum(value for key, value in counts.items()) == len(ranks)


# --------------------------------------------------------------------------
# sentences in a chunk


def test_sentence_containment_reports_full_partial_and_none() -> None:
    sentence = "Alpha was born in Paris in 1900."
    assert pool.sentence_in_chunk(sentence, f"text {sentence} more") == "full"
    # Cutting every 100 words splits sentences across chunks: half a sentence
    # counts as a partial hit, not a miss.
    assert pool.sentence_in_chunk(sentence, "text Alpha was born") == "partial"
    assert pool.sentence_in_chunk(sentence, "unrelated text") == "none"


def test_sentence_containment_normalizes_whitespace_and_case() -> None:
    assert pool.sentence_in_chunk("Alpha  lives\nhere", "ALPHA LIVES HERE") == "full"


def test_gold_sentences_are_grouped_by_normalized_title() -> None:
    sample = {
        "_id": "s",
        "supporting_facts": [["Procter &amp; Gamble", 1], ["Other", 0]],
        "context": [
            ["Procter &amp; Gamble", ["zero", "one"]],
            ["Other", ["only"]],
        ],
    }
    grouped = pool.gold_sentences_by_title(sample)
    assert grouped["procter & gamble"] == ["one"]
    assert grouped["other"] == ["only"]


# --------------------------------------------------------------------------
# report and select end to end


def coverage_file(tmp_path: Path) -> Path:
    path = tmp_path / "coverage.json"
    path.write_text(
        json.dumps({"title_em_ceiling_normalized": 0.5}), encoding="utf-8"
    )
    return path


def test_report_measures_the_pool_and_the_oracle_over_it() -> None:
    gold = {
        "sample-1": {
            "gold a": ["Alpha lives in Paris."],
            "gold b": ["Beta was born in Rome."],
        }
    }
    result = pool.report([diag_record()], POOL, gold, (2, 4), ceiling=0.5)
    assert result["examples"] == 1
    assert result["pool"]["title_em_at_pool"] == 1.0
    assert result["pool"]["examples_with_all_gold"] == 1.0
    # Both gold titles are in the pool but GTE top-2 misses them: the whole
    # point of the oracle.
    assert result["gold_rank"]["best_found"]["median"] == 2.0
    assert result["gold_rank"]["second_found"]["median"] == 3.0
    assert result["oracle_by_titles"]["2"]["title_em"] == 1.0
    # The fixture ceiling is 0.5, so the share of the ceiling doubles title EM.
    assert result["pool"]["share_of_ceiling"] == 2.0
    assert result["chunk_granularity"]["selected_chunk"]["full"] == 2
    assert result["chunk_granularity"]["examples_losing_a_sentence"] == 0.0


def test_report_counts_a_title_found_without_its_sentence() -> None:
    # The title is in the pool but the needed sentence is in another chunk of
    # the article: a ceiling that reranking cannot fix.
    gold = {"sample-1": {"gold a": ["A sentence that is not in the chunk at all."]}}
    result = pool.report([diag_record()], POOL, gold, (2,), ceiling=0.5)
    assert result["chunk_granularity"]["selected_chunk"]["none"] == 1
    assert result["chunk_granularity"]["titles_missing_sentence"] == 1.0
    assert result["chunk_granularity"]["examples_losing_a_sentence"] == 1.0


def test_report_counts_examples_missing_the_second_gold() -> None:
    record = diag_record(gold_titles=["Gold A", "Absent"])
    result = pool.report([record], POOL, {}, (2,), ceiling=0.5)
    assert result["pool"]["examples_with_one_gold"] == 1.0
    assert result["pool"]["title_em_at_pool"] == 0.0
    assert result["gold_rank"]["second_missing_given_first_found"] == 1.0


def test_select_writes_a_scored_variant_the_reader_can_consume(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "retrieval.jsonl"
    summary = pool.select([diag_record()], POOL, 2, destination)
    assert summary == {
        "samples": 1,
        "budget": 2,
        "max_per_title": 1,
        "prefer": "rank",
        "title_em": 1.0,
        "title_recall": 1.0,
    }
    written = json.loads(destination.read_text(encoding="utf-8").strip())
    assert written["pred_idx"] == [12, 13]
    assert written["retrieved_titles"] == ["Gold A", "Gold B"]
    assert written["q_values"] == pytest.approx([0.7, 0.6])
    assert written["title_em"] == 1.0
    assert written["eval_variant"] == "oracle-titles-2"
    # The answer and the question must reach the reader untouched.
    assert written["question"] == "q"
    assert written["answer"] == "a"
    # The large candidate dump is not carried over into the derived run.
    assert len(written["retrieval_hops"]) == 2
    assert "first_stage_candidate_idx" not in written["retrieval_hops"][0]


def test_select_records_the_chunk_aware_mode_in_the_variant_name(
    tmp_path: Path,
) -> None:
    # The run must name itself: its results row is compared with the title
    # oracle, and the two must not be confused.
    destination = tmp_path / "retrieval.jsonl"
    gold = {"sample-1": {"gold a": ["Alpha lives in Paris."]}}
    summary = pool.select([diag_record()], POOL, 2, destination, gold, 2)
    assert summary["prefer"] == "gold-sentence"
    assert summary["max_per_title"] == 2
    written = json.loads(destination.read_text(encoding="utf-8").strip())
    assert written["reranker"] == "oracle-chunks"
    assert written["eval_variant"] == "oracle-chunks-2-n2"


def test_select_marks_the_variant_so_it_cannot_be_truncated() -> None:
    import build_eval_variants as variants

    record = pool.oracle_record(diag_record(), [10, 11, 12, 13], list(POOL.rows.values()), 4)
    # An oracle selection is not a first-stage baseline, so it must not be
    # truncated: k=2 is rebuilt from the pool, not by cutting the tail.
    assert record["reranker"] == "oracle-titles"
    with pytest.raises(ValueError, match="prefix-consistent"):
        variants.build([record], "truncate", 2)
