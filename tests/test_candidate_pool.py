from __future__ import annotations

import json
from pathlib import Path

import pytest

import candidate_pool as pool


def chunk(title: str, body: str = "body") -> str:
    return f'"{title}"\n{body}'


def diag_record(**overrides) -> dict:
    """Запись рана A: пул логируется целиком в первом хопе."""
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
    """Row ID → чанк, как WikiCorpus, но без корпуса на диске."""

    def __init__(self, rows: dict[int, str]) -> None:
        self.rows = rows

    def read_rows(self, row_ids):
        return [{"id": str(row), "contents": self.rows[int(row)]} for row in row_ids]


# Пул: два шумовых чанка сверху, оба gold ниже. Ровно та ситуация, ради
# которой считается oracle: реранкер обязан спуститься вниз по ранжированию.
POOL = FakeCorpus(
    {
        10: chunk("Noise 0"),
        11: chunk("Noise 1"),
        12: chunk("Gold A", "Alpha lives in Paris."),
        13: chunk("Gold B", "Beta was born in Rome."),
    }
)


# --------------------------------------------------------------------------
# пул


def test_pool_comes_from_the_first_stage_dump_of_the_first_hop() -> None:
    assert pool.pool_row_ids(diag_record()) == [10, 11, 12, 13]


def test_pool_refuses_a_run_logged_without_candidates() -> None:
    # Все восемь канонических ранов сделаны с --log-candidates none, поэтому
    # ошибка должна называть флаг, а не падать по KeyError.
    record = diag_record(retrieval_hops=[{"step": 0, "selected_idx": 10}])
    with pytest.raises(ValueError, match="--log-candidates full"):
        pool.pool_row_ids(record)


# --------------------------------------------------------------------------
# выбор


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
    # Wiki-18 держит несколько чанков одной статьи; дедупликация включена во
    # всех опубликованных ранах, поэтому добор обязан её соблюдать.
    titles = ["Noise 0", "Noise 0", "Noise 1", "Gold A"]
    assert pool.select_positions(titles, ["Gold A"], 3) == [3, 0, 2]


def test_oracle_matches_titles_across_the_two_wikipedia_dumps() -> None:
    titles = ["Noise", "Procter & Gamble"]
    assert pool.select_positions(titles, ["Procter &amp; Gamble"], 1) == [1]


def test_oracle_refuses_a_budget_larger_than_the_pool_can_offer() -> None:
    with pytest.raises(ValueError, match="only 2 chunks at 1 per title"):
        pool.select_positions(["A", "A", "B"], ["A"], 3)
    # Та же тройка чанков при квоте 2 бюджет уже закрывает.
    assert pool.select_positions(["A", "A", "B"], ["A"], 3, max_per_title=2) == [
        0, 1, 2
    ]


def test_missing_gold_falls_back_to_the_top_of_the_ranking() -> None:
    titles = ["Noise 0", "Noise 1"]
    assert pool.select_positions(titles, ["Absent"], 2) == [0, 1]


# --------------------------------------------------------------------------
# чанк-осознанный выбор

# Статья Wiki-18 нарезана по 100 слов, поэтому её gold-предложения
# расходятся по соседним чанкам: у «Gold A» первое в чанке 1, второе в
# чанке 2, а «Gold B» целиком в одном чанке и имеет ещё один без gold-текста.
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
    # У «Gold A» два gold-предложения, в каждом чанке лежит одно: оба чанка
    # частичные. У «Gold B» первый чанк полный, второй пустой.
    assert split_penalties() == [
        pool.VERDICT_PENALTY["none"],
        pool.VERDICT_PENALTY["partial"],
        pool.VERDICT_PENALTY["partial"],
        pool.VERDICT_PENALTY["full"],
        pool.VERDICT_PENALTY["none"],
    ]


def test_verdict_needs_every_gold_sentence_of_the_title() -> None:
    # Один из двух фактов остался в соседнем чанке — ридер отвечает по
    # неполному контексту, поэтому это не 'full'.
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
    # Оракул по титулам взял бы чанк 1 — он выше по рангу GTE. Чанковый
    # спускается на 2, потому что gold-предложение лежит там.
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
    # Бюджет 4, квота 2: по одному чанку на титул, затем второй чанк «Gold A»
    # (в нём второе gold-предложение) — и только потом добор по рангу.
    # Второй чанк «Gold B» gold-текста не несёт и в раунд не попадает.
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
# позиции gold


def test_gold_ranks_are_sorted_and_flag_the_absent_title() -> None:
    titles = ["Noise 0", "Gold B", "Noise 1", "Gold A"]
    assert pool.gold_ranks(titles, ["Gold A", "Gold B"]) == [1, 3]
    # Отсутствующий титул уходит в конец: «второй gold» — это худший из
    # найденных, а не второй по разметке.
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
# предложения в чанке


def test_sentence_containment_reports_full_partial_and_none() -> None:
    sentence = "Alpha was born in Paris in 1900."
    assert pool.sentence_in_chunk(sentence, f"text {sentence} more") == "full"
    # Нарезка по 100 слов рвёт предложения между чанками: половина считается
    # частичным попаданием, а не промахом.
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
# report и select целиком


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
    # Оба gold лежат в пуле, но GTE top-2 их не достаёт — весь смысл oracle.
    assert result["gold_rank"]["best_found"]["median"] == 2.0
    assert result["gold_rank"]["second_found"]["median"] == 3.0
    assert result["oracle_by_titles"]["2"]["title_em"] == 1.0
    # Потолок 0.5 в фикстуре, поэтому доля потолка удваивает title EM.
    assert result["pool"]["share_of_ceiling"] == 2.0
    assert result["chunk_granularity"]["selected_chunk"]["full"] == 2
    assert result["chunk_granularity"]["examples_losing_a_sentence"] == 0.0


def test_report_counts_a_title_found_without_its_sentence() -> None:
    # Титул в пуле есть, но нужное предложение лежит в другом чанке статьи:
    # это потолок, который реранкингом не лечится.
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
    # Ответ и вопрос обязаны дойти до ридера нетронутыми.
    assert written["question"] == "q"
    assert written["answer"] == "a"
    # Стотысячный дамп кандидатов в производный ран не переносится.
    assert len(written["retrieval_hops"]) == 2
    assert "first_stage_candidate_idx" not in written["retrieval_hops"][0]


def test_select_records_the_chunk_aware_mode_in_the_variant_name(
    tmp_path: Path,
) -> None:
    # Ран должен сам себя называть: строка в RESULTS.md сравнивается с
    # оракулом по титулам, и перепутать их нельзя.
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
    # Oracle-выбор не является first-stage baseline, поэтому усекать его
    # нельзя: k=2 строится из пула заново, а не отрезанием хвоста.
    assert record["reranker"] == "oracle-titles"
    with pytest.raises(ValueError, match="prefix-consistent"):
        variants.build([record], "truncate", 2)
