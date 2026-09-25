#!/usr/bin/env python3
"""Что вообще лежит в top-100 кандидатов GTE и сколько из этого достижимо.

Реранкер не может выбрать то, чего нет в пуле. Прежде чем вкладываться в
обучение реранкера, нужно знать две вещи: какая доля вопросов в принципе
решаема из этого пула и сколько ответов даёт идеальный выбор из него. Обе
считаются здесь, из ранa с ``--log-candidates full``.

``report``
    Диагностика пула в JSON: title recall/EM@100, распределение позиций
    gold-титулов, oracle-выбор по титулам, доля примеров, где титул найден,
    а gold-предложение в чанк не попало.

``select``
    Собирает вариант retrieval JSONL, где из тех же 100 кандидатов взяты k
    чанков, максимизирующих покрытие gold-титулов. Это **oracle по титулам**,
    а не по ответу: он показывает потолок реранкинга над данным пулом, а не
    абсолютный потолок задачи.

Пул берётся из ``retrieval_hops[0]["first_stage_candidate_idx"]`` — сырого
ранжирования GTE до каких-либо исключений. Титулы и тексты чанков
резолвятся по row ID через ту же таблицу байтовых смещений, что и сам
ретривал, поэтому «строка 12345» здесь и в ране означает одно и то же.

Примеры:

    python src/candidate_pool.py report \
      --input runs/2026-08-02-gte-pool-diag/retrieval.jsonl \
      --index-dir .../wiki18-qrag-jul21-best-raw \
      --gold-source .../hotpot_dev_distractor_v1.json \
      --coverage runs/shared/hotpotqa_dev_title_coverage.json \
      --output runs/shared/hotpotqa_dev_pool_diagnostics.json

    python src/candidate_pool.py select --steps 6 \
      --input runs/2026-08-02-gte-pool-diag/retrieval.jsonl \
      --index-dir .../wiki18-qrag-jul21-best-raw \
      --output runs/2026-08-02-pool-oracle-titles-k6/retrieval.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import statistics
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence
import unicodedata

import numpy as np

from build_index_wiki_gte import read_json
from fullwiki_qrag import (
    DEFAULT_OFFSETS_FILE,
    MANIFEST_FILE,
    WikiCorpus,
    iter_jsonl,
    load_input_samples,
    normalize_title,
    ordered_unique,
    resolve_corpus_path,
    sample_id,
    title_metrics,
    wiki_title,
)


LOG = logging.getLogger("candidate-pool")

# Бюджеты чанков, для которых считается oracle-выбор. Те же, что у
# канонических строк GTE top-2/4/6 в RESULTS.md, иначе сравнивать не с чем.
DEFAULT_BUDGETS = (2, 4, 6)

# Позиция в ранжировании, если gold-титула в пуле нет. Отдельным значением, а
# не None, чтобы гистограмма и медиана считались по одному и тому же полю.
MISSING_RANK = -1


# --------------------------------------------------------------------------
# пул и корпус


def pool_row_ids(record: dict[str, Any]) -> list[int]:
    """Сырое top-k ранжирование GTE из первого шага рана.

    В режиме ``fixed`` пул считается один раз и переиспользуется каждым
    шагом, поэтому первого шага достаточно. ``first_stage_candidate_idx``, а
    не ``candidate_idx``: второй у reranker=none совпадает с первым только на
    нулевом шаге, дальше в нём уже стоят исключения по дедупликации.
    """
    hops = record.get("retrieval_hops")
    if not hops:
        raise ValueError(f"Record {record.get('id')!r} has no retrieval_hops")
    hop = hops[0]
    pool = hop.get("first_stage_candidate_idx")
    if pool is None:
        raise ValueError(
            f"Record {record.get('id')!r} has no first_stage_candidate_idx: "
            "the run needs --log-candidates full"
        )
    return [int(value) for value in pool]


def open_corpus(index_dir: Path, corpus: Path | None, offsets: Path | None) -> WikiCorpus:
    manifest = read_json(index_dir / MANIFEST_FILE)
    return WikiCorpus(
        resolve_corpus_path(manifest, corpus),
        offsets if offsets is not None else index_dir / DEFAULT_OFFSETS_FILE,
        int(manifest["corpus"]["rows"]),
        auto_prepare=False,
    )


def iter_pools(
    records: Iterable[dict[str, Any]],
    corpus: WikiCorpus,
    progress_every: int = 500,
) -> Iterator[tuple[dict[str, Any], list[int], list[str]]]:
    """Записи вместе с текстами их пула, по одному чтению корпуса на запись."""
    started = time.monotonic()
    for number, record in enumerate(records, start=1):
        row_ids = pool_row_ids(record)
        texts = [item["contents"] for item in corpus.read_rows(row_ids)]
        yield record, row_ids, texts
        if progress_every and number % progress_every == 0:
            LOG.info(
                "%d записей, %.0f c (%.1f зап/с)",
                number,
                time.monotonic() - started,
                number / max(time.monotonic() - started, 1e-9),
            )


# --------------------------------------------------------------------------
# сопоставление предложений


def normalize_text(text: str) -> str:
    """Одна форма для gold-предложения и для чанка Wiki-18.

    Дампы 2017 и 2018 годов различаются HTML-сущностями и пробелами, как и
    титулы (см. ``normalize_title``); сравнивать сырые строки бессмысленно.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(folded.split())


def sentence_in_chunk(sentence: str, chunk: str) -> str:
    """Насколько gold-предложение попало в чанк: full, partial или none.

    Чанк Wiki-18 — это ровно 100 слов, нарезка не уважает границы
    предложений, поэтому gold-предложение регулярно оказывается разорванным
    между двумя чанками. Строгое вхождение занижает долю попаданий, половинное
    её завышает: настоящее значение лежит между ними, и обе границы полезнее
    одного числа с невнятной погрешностью.
    """
    needle = normalize_text(sentence)
    haystack = normalize_text(chunk)
    if not needle:
        return "none"
    if needle in haystack:
        return "full"
    words = needle.split()
    if len(words) >= 4:
        middle = len(words) // 2
        halves = (" ".join(words[:middle]), " ".join(words[middle:]))
        if any(half and half in haystack for half in halves):
            return "partial"
    return "none"


def chunk_verdict(sentences: Sequence[str], chunk: str) -> str:
    """Вердикт по всем gold-предложениям титула сразу.

    ``full`` только если в чанк попали все предложения: если у титула их два и
    один остался в соседнем чанке, ридер всё равно отвечает по неполному
    контексту. ``none`` — не попало ни одного, остальное ``partial``.
    """
    verdicts = [sentence_in_chunk(sentence, chunk) for sentence in sentences]
    if verdicts and all(value == "full" for value in verdicts):
        return "full"
    if not verdicts or all(value == "none" for value in verdicts):
        return "none"
    return "partial"


# Чем меньше, тем предпочтительнее чанк. Порядок «целиком > половина > нет»,
# а не «есть > нет»: разорванное по границе чанка предложение всё-таки лучше
# чанка, где нужного текста нет вовсе.
VERDICT_PENALTY = {"full": 0, "partial": 1, "none": 2}


def gold_sentences_by_title(sample: dict[str, Any]) -> dict[str, list[str]]:
    """Gold-предложения из distractor-файла, сгруппированные по титулу.

    Только distractor-файл содержит gold-абзацы: в fullwiki-файле поле
    ``context`` — это выдача исходного ретривера (docs/gotchas.md).
    """
    supporting = sample.get("supporting_facts")
    context = sample.get("context")
    if not isinstance(supporting, list) or not isinstance(context, list):
        return {}
    by_title = {
        str(title): sentences
        for title, sentences in context
        if isinstance(sentences, list)
    }
    result: dict[str, list[str]] = {}
    for fact in supporting:
        if not isinstance(fact, list) or len(fact) != 2:
            continue
        title, position = str(fact[0]), int(fact[1])
        sentences = by_title.get(title, [])
        if 0 <= position < len(sentences):
            result.setdefault(normalize_title(title), []).append(sentences[position])
    return result


# --------------------------------------------------------------------------
# выбор чанков


def first_positions(normalized_pool: Sequence[str]) -> dict[str, int]:
    """Титул → его лучшая позиция в пуле. Считается один раз на запись."""
    best_position: dict[str, int] = {}
    for position, title in enumerate(normalized_pool):
        best_position.setdefault(title, position)
    return best_position


def sentence_penalties(
    normalized_pool: Sequence[str],
    texts: Sequence[str],
    gold_sentences: Mapping[str, Sequence[str]],
) -> list[int]:
    """Штраф каждого чанка пула по ``VERDICT_PENALTY``.

    Считается только для титулов, чьи gold-предложения известны: у остальных
    все чанки получают одинаковый штраф, и сортировка внутри титула
    вырождается в ранг GTE — ровно то, что делал оракул по титулам.
    """
    penalties = [VERDICT_PENALTY["none"]] * len(texts)
    for position, title in enumerate(normalized_pool):
        sentences = gold_sentences.get(title)
        if sentences:
            penalties[position] = VERDICT_PENALTY[
                chunk_verdict(sentences, texts[position])
            ]
    return penalties


def chunks_by_title(
    normalized_pool: Sequence[str],
    penalties: Sequence[int] | None,
) -> dict[str, list[int]]:
    """Титул → его позиции в пуле, лучшая первой.

    Порядок внутри титула — ``(penalty, ранг GTE)``. Без штрафов это просто
    ранг, то есть прежний «лучший по рангу чанк статьи».
    """
    grouped: dict[str, list[int]] = {}
    for position, title in enumerate(normalized_pool):
        grouped.setdefault(title, []).append(position)
    if penalties is not None:
        for positions in grouped.values():
            positions.sort(key=lambda position: (penalties[position], position))
    return grouped


def select_positions(
    pool_titles: Sequence[str],
    gold_titles: Sequence[str],
    budget: int,
    normalized_pool: Sequence[str] | None = None,
    penalties: Sequence[int] | None = None,
    max_per_title: int = 1,
) -> list[int]:
    """Позиции k чанков, максимизирующих покрытие gold-титулов.

    Сначала по одному чанку на каждый найденный gold-титул, затем — если
    ``max_per_title`` больше единицы — вторые чанки тех же титулов, и только
    потом добор сверху ранжирования. Квота на титул соблюдается и в доборе:
    она инвариант ранов, с которыми этот выбор сравнивается.

    ``penalties`` задаёт предпочтение между чанками одного титула: 0 лучше,
    чем 1. Оракул по титулам их не передаёт и берёт лучший по рангу GTE
    чанк; чанк-осознанный оракул ставит вперёд тот чанк, в котором лежит
    gold-предложение (``VERDICT_PENALTY``). Между титулами штраф ничего не
    решает — титул берётся в любом случае, вопрос только каким чанком.

    Вторые чанки берутся не подряд, а раундами и только если чанк вообще
    несёт gold-текст (штраф меньше ``VERDICT_PENALTY["none"]``): иначе бюджет
    уходил бы на второй чанк уже покрытой статьи вместо нового титула.

    Порядок выбора — раунд за раундом, внутри раунда по рангу — делает
    результат префикс-согласованным по бюджету: выбор на 2 чанка есть префикс
    выбора на 6. Поэтому строки k=2/4/6 сравнимы между собой как один и тот
    же алгоритм с разным бюджетом.
    """
    if budget <= 0:
        raise ValueError(f"Budget must be positive: {budget}")
    if max_per_title < 1:
        raise ValueError(f"max_per_title must be positive: {max_per_title}")
    if normalized_pool is None:
        normalized_pool = [normalize_title(title) for title in pool_titles]
    wanted = {normalize_title(title) for title in gold_titles}
    grouped = chunks_by_title(normalized_pool, penalties)
    found = [title for title in wanted if title in grouped]

    chosen: list[int] = []
    taken: dict[str, int] = {}
    for round_number in range(max_per_title):
        candidates = []
        for title in found:
            positions = grouped[title]
            if len(positions) <= round_number:
                continue
            position = positions[round_number]
            if round_number and penalties is not None:
                if penalties[position] >= VERDICT_PENALTY["none"]:
                    continue
            candidates.append(position)
        for position in sorted(candidates):
            if len(chosen) >= budget:
                break
            chosen.append(position)
            taken[normalized_pool[position]] = taken.get(normalized_pool[position], 0) + 1

    for position, title in enumerate(normalized_pool):
        if len(chosen) >= budget:
            break
        if position in chosen or taken.get(title, 0) >= max_per_title:
            continue
        chosen.append(position)
        taken[title] = taken.get(title, 0) + 1

    if len(chosen) < budget:
        raise ValueError(
            f"Pool of {len(pool_titles)} chunks offers only {len(chosen)} "
            f"chunks at {max_per_title} per title, cannot fill a budget of "
            f"{budget}"
        )
    return chosen


# --------------------------------------------------------------------------
# report


def gold_ranks(
    pool_titles: Sequence[str],
    gold_titles: Sequence[str],
    normalized_pool: Sequence[str] | None = None,
) -> list[int]:
    """Позиция каждого gold-титула в ранжировании GTE, по возрастанию.

    Позиции сортируются, а не берутся в порядке gold: для bridge-вопроса
    важно не «какой титул первый в разметке», а насколько глубоко лежит
    худший из тех, что нужно достать.
    """
    if normalized_pool is None:
        normalized_pool = [normalize_title(title) for title in pool_titles]
    best_position = first_positions(normalized_pool)
    ranks = [
        best_position.get(title, MISSING_RANK)
        for title in ordered_unique(normalize_title(name) for name in gold_titles)
    ]
    found = sorted(rank for rank in ranks if rank != MISSING_RANK)
    return found + [MISSING_RANK] * (len(ranks) - len(found))


def histogram(ranks: Sequence[int], edges: Sequence[int]) -> dict[str, int]:
    """Гистограмма позиций по границам вида 1, 5, 10, ... плюс «не найден»."""
    counts = {"missing": sum(1 for rank in ranks if rank == MISSING_RANK)}
    previous = 0
    for edge in edges:
        counts[f"{previous + 1}-{edge}"] = sum(
            1 for rank in ranks if previous <= rank < edge
        )
        previous = edge
    counts[f"{previous + 1}+"] = sum(1 for rank in ranks if rank >= previous)
    return counts


def summarize(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "median": float(statistics.median(values)),
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(max(values)),
    }


def report(
    records: Iterable[dict[str, Any]],
    corpus: WikiCorpus,
    gold_by_id: dict[str, dict[str, list[str]]],
    budgets: Sequence[int],
    ceiling: float,
) -> dict[str, Any]:
    total = 0
    pool_sizes: set[int] = set()
    recall_at_pool: list[float] = []
    em_at_pool: list[float] = []
    found_counts: list[int] = []
    gold_counts: list[int] = []
    best_ranks: list[int] = []
    second_ranks: list[int] = []
    second_missing_given_first = 0
    first_found = 0
    oracle_em: dict[int, list[float]] = {budget: [] for budget in budgets}
    oracle_recall: dict[int, list[float]] = {budget: [] for budget in budgets}

    # Титул нашли, а нужного предложения в его чанке нет. Считается по
    # титулам (у одного примера их два) и отдельно по примерам.
    sentence_titles = {"full": 0, "partial": 0, "none": 0, "total": 0}
    sentence_titles_any = {"full": 0, "partial": 0, "none": 0, "total": 0}
    examples_with_lost_sentence = 0
    examples_with_gold_in_pool = 0

    for record, row_ids, texts in iter_pools(records, corpus):
        total += 1
        pool_sizes.add(len(row_ids))
        pool_titles = [wiki_title(text) for text in texts]
        normalized_pool = [normalize_title(title) for title in pool_titles]
        gold = ordered_unique(str(title) for title in record.get("gold_titles", []))
        metrics = title_metrics(gold, pool_titles)
        recall_at_pool.append(metrics["title_recall"])
        em_at_pool.append(metrics["title_em"])
        gold_counts.append(len(gold))

        ranks = gold_ranks(pool_titles, gold, normalized_pool)
        found_counts.append(sum(1 for rank in ranks if rank != MISSING_RANK))
        if ranks:
            best_ranks.append(ranks[0])
        if len(ranks) > 1:
            second_ranks.append(ranks[1])
            if ranks[0] != MISSING_RANK:
                first_found += 1
                if ranks[1] == MISSING_RANK:
                    second_missing_given_first += 1

        for budget in budgets:
            positions = select_positions(pool_titles, gold, budget, normalized_pool)
            picked = [pool_titles[position] for position in positions]
            picked_metrics = title_metrics(gold, picked)
            oracle_em[budget].append(picked_metrics["title_em"])
            oracle_recall[budget].append(picked_metrics["title_recall"])

        # Гранулярность чанка: титул в пуле есть — а предложение в нём?
        gold_sentences = gold_by_id.get(str(record.get("id")), {})
        lost_here = False
        gold_present_here = False
        for title, sentences in gold_sentences.items():
            positions = [
                position
                for position, name in enumerate(normalized_pool)
                if name == title
            ]
            if not positions:
                continue
            gold_present_here = True
            # Тот чанк, который реально возьмёт отбор: лучший по рангу.
            verdict = chunk_verdict(sentences, texts[positions[0]])
            sentence_titles[verdict] += 1
            sentence_titles["total"] += 1
            if verdict != "full":
                lost_here = True
            # Тот же вопрос, но если бы отбор мог взять любой чанк статьи из
            # пула: это верхняя граница для гранулярности, и ровно её берёт
            # чанк-осознанный выбор в `select --prefer gold-sentence`.
            best = min(
                (chunk_verdict(sentences, texts[position]) for position in positions),
                key=lambda value: VERDICT_PENALTY[value],
            )
            sentence_titles_any[best] += 1
            sentence_titles_any["total"] += 1
        if gold_present_here:
            examples_with_gold_in_pool += 1
            examples_with_lost_sentence += int(lost_here)

    if not total:
        raise ValueError("No records to report on")

    share = lambda count: count / total  # noqa: E731
    title_em = float(np.mean(em_at_pool))
    return {
        "examples": total,
        "pool_size": sorted(pool_sizes),
        "gold_titles_per_example": sorted(set(gold_counts)),
        "ceiling_normalized": ceiling,
        "pool": {
            "title_recall_at_pool": float(np.mean(recall_at_pool)),
            "title_em_at_pool": title_em,
            "share_of_ceiling": title_em / ceiling if ceiling else float("nan"),
            "examples_with_no_gold": share(
                sum(1 for count in found_counts if count == 0)
            ),
            "examples_with_one_gold": share(
                sum(1 for count in found_counts if count == 1)
            ),
            "examples_with_all_gold": share(
                sum(
                    1
                    for count, wanted in zip(found_counts, gold_counts)
                    if count == wanted
                )
            ),
        },
        "gold_rank": {
            "comment": (
                "Позиции gold-титулов в ранжировании GTE, 0-based, "
                "отсортированные по возрастанию: 'best' — легче найденный "
                f"gold, 'second' — следующий за ним. {MISSING_RANK} означает "
                "«в пуле нет»."
            ),
            "best_found": summarize(
                [rank for rank in best_ranks if rank != MISSING_RANK]
            ),
            "second_found": summarize(
                [rank for rank in second_ranks if rank != MISSING_RANK]
            ),
            "best_histogram": histogram(best_ranks, (1, 5, 10, 20, 50, 100)),
            "second_histogram": histogram(second_ranks, (1, 5, 10, 20, 50, 100)),
            "second_missing_given_first_found": (
                second_missing_given_first / first_found if first_found else 0.0
            ),
            "examples_with_first_found": first_found,
        },
        "oracle_by_titles": {
            str(budget): {
                "title_em": float(np.mean(oracle_em[budget])),
                "title_recall": float(np.mean(oracle_recall[budget])),
                "share_of_ceiling": (
                    float(np.mean(oracle_em[budget])) / ceiling
                    if ceiling
                    else float("nan")
                ),
            }
            for budget in budgets
        },
        "chunk_granularity": {
            "comment": (
                "Титул в пуле есть — лежит ли в его чанке gold-предложение. "
                "'full' — предложение целиком внутри чанка, 'partial' — "
                "половина (чанк Wiki-18 режется по 100 слов и рвёт "
                "предложения), 'none' — не найдено. 'selected_chunk' — "
                "лучший по рангу чанк титула, то есть тот, который возьмёт "
                "отбор при включённой дедупликации; 'any_chunk_in_pool' — "
                "верхняя граница, если бы разрешалось взять любой чанк "
                "статьи из пула."
            ),
            "selected_chunk": sentence_titles,
            "any_chunk_in_pool": sentence_titles_any,
            "titles_missing_sentence": (
                sentence_titles["none"] / sentence_titles["total"]
                if sentence_titles["total"]
                else 0.0
            ),
            "examples_with_gold_in_pool": examples_with_gold_in_pool,
            "examples_losing_a_sentence": (
                examples_with_lost_sentence / examples_with_gold_in_pool
                if examples_with_gold_in_pool
                else 0.0
            ),
        },
    }


# --------------------------------------------------------------------------
# select


def oracle_record(
    record: dict[str, Any],
    row_ids: Sequence[int],
    texts: Sequence[str],
    budget: int,
    gold_sentences: Mapping[str, Sequence[str]] | None = None,
    max_per_title: int = 1,
) -> dict[str, Any]:
    pool_titles = [wiki_title(text) for text in texts]
    normalized_pool = [normalize_title(title) for title in pool_titles]
    gold = ordered_unique(str(title) for title in record.get("gold_titles", []))
    penalties = (
        None
        if gold_sentences is None
        else sentence_penalties(normalized_pool, texts, gold_sentences)
    )
    positions = select_positions(
        pool_titles, gold, budget, normalized_pool, penalties, max_per_title
    )
    scores = record["retrieval_hops"][0].get("first_stage_scores")

    result = dict(record)
    result["pred_idx"] = [int(row_ids[position]) for position in positions]
    result["pred_texts"] = [texts[position] for position in positions]
    result["q_values"] = (
        [float(scores[position]) for position in positions] if scores else []
    )
    # Пул один и тот же на всех шагах, отдельных хопов у этого варианта нет:
    # логировать сто кандидатов повторно незачем, ран A их уже сохранил.
    result["retrieval_hops"] = [
        {"step": step, "selected_idx": int(row_ids[position]), "pool_rank": position}
        for step, position in enumerate(positions)
    ]
    result["retrieved_titles"] = [pool_titles[position] for position in positions]
    result.update(title_metrics(gold, result["retrieved_titles"]))
    # Не 'none': build_eval_variants.py обязан отказаться усекать этот ран,
    # он не является first-stage baseline.
    mode = "oracle-titles" if gold_sentences is None else "oracle-chunks"
    result["reranker"] = mode
    result["eval_variant"] = f"{mode}-{budget}" + (
        f"-n{max_per_title}" if max_per_title > 1 else ""
    )
    return result


def select(
    records: Iterable[dict[str, Any]],
    corpus: WikiCorpus,
    budget: int,
    destination: Path,
    gold_by_id: dict[str, dict[str, list[str]]] | None = None,
    max_per_title: int = 1,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    title_em: list[float] = []
    title_recall: list[float] = []
    with destination.open("w", encoding="utf-8") as stream:
        for record, row_ids, texts in iter_pools(records, corpus):
            sentences = (
                None
                if gold_by_id is None
                else gold_by_id.get(str(record.get("id")), {})
            )
            result = oracle_record(
                record, row_ids, texts, budget, sentences, max_per_title
            )
            title_em.append(result["title_em"])
            title_recall.append(result["title_recall"])
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            written += 1
    return {
        "samples": written,
        "budget": budget,
        "max_per_title": max_per_title,
        "prefer": "rank" if gold_by_id is None else "gold-sentence",
        "title_em": float(np.mean(title_em)) if title_em else 0.0,
        "title_recall": float(np.mean(title_recall)) if title_recall else 0.0,
    }


# --------------------------------------------------------------------------
# CLI


def add_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--offsets", type=Path, default=None)


def command_report(args: argparse.Namespace) -> int:
    corpus = open_corpus(
        args.index_dir.expanduser().resolve(), args.corpus, args.offsets
    )
    gold_by_id: dict[str, dict[str, list[str]]] = {}
    if args.gold_source is not None:
        gold_by_id = load_gold_sentences(args.gold_source.expanduser().resolve())
    ceiling = float(
        read_json(args.coverage.expanduser().resolve())["title_em_ceiling_normalized"]
    )
    result = report(
        iter_jsonl(args.input.expanduser().resolve()),
        corpus,
        gold_by_id,
        args.budget or list(DEFAULT_BUDGETS),
        ceiling,
    )
    result["input"] = str(args.input.expanduser().resolve())
    result["gold_source"] = (
        str(args.gold_source.expanduser().resolve()) if args.gold_source else None
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
        LOG.info("Записан %s", destination)
    return 0


def load_gold_sentences(path: Path) -> dict[str, dict[str, list[str]]]:
    gold_by_id: dict[str, dict[str, list[str]]] = {}
    for index, sample in enumerate(load_input_samples(path)):
        gold_by_id[sample_id(sample, index)] = gold_sentences_by_title(sample)
    if not any(gold_by_id.values()):
        raise ValueError(
            f"{path} has no recoverable gold sentences; the sentence-level "
            "diagnostic needs hotpot_dev_distractor_v1.json"
        )
    return gold_by_id


def command_select(args: argparse.Namespace) -> int:
    source = args.input.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if source == destination:
        raise ValueError("Refusing to overwrite the input file in place")
    if args.prefer == "gold-sentence" and args.gold_source is None:
        raise ValueError("--prefer gold-sentence requires --gold-source")
    corpus = open_corpus(
        args.index_dir.expanduser().resolve(), args.corpus, args.offsets
    )
    gold_by_id = (
        None
        if args.prefer == "rank"
        else load_gold_sentences(args.gold_source.expanduser().resolve())
    )
    summary = select(
        iter_jsonl(source),
        corpus,
        args.steps,
        destination,
        gold_by_id,
        args.max_per_title,
    )
    LOG.info("%s", json.dumps(summary, indent=2))
    LOG.info("Записан %s", destination)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser("report", help="диагностика пула в JSON")
    add_corpus_arguments(report_parser)
    report_parser.add_argument(
        "--gold-source",
        type=Path,
        default=None,
        help=(
            "hotpot_dev_distractor_v1.json: единственный dev-файл, где "
            "context содержит gold-абзацы. Без него не считается только "
            "диагностика по предложениям"
        ),
    )
    report_parser.add_argument("--coverage", type=Path, required=True)
    report_parser.add_argument(
        "--budget", type=int, action="append", default=None,
        help=f"бюджеты чанков для oracle-выбора; по умолчанию {DEFAULT_BUDGETS}",
    )
    report_parser.add_argument("--output", type=Path, default=None)
    report_parser.set_defaults(handler=command_report)

    select_parser = subparsers.add_parser(
        "select", help="вариант JSONL с oracle-выбором по титулам"
    )
    add_corpus_arguments(select_parser)
    select_parser.add_argument("--steps", type=int, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument(
        "--prefer",
        choices=("rank", "gold-sentence"),
        default="rank",
        help=(
            "какой чанк gold-титула брать: 'rank' — лучший по рангу GTE "
            "(оракул по титулам), 'gold-sentence' — тот, в котором лежит "
            "gold-предложение (целиком > половина > нет), тай-брейк — ранг. "
            "Второй режим и есть честная верхняя граница чанкового реранкера: "
            "он выбирает из тех же чанков, что и реранкер"
        ),
    )
    select_parser.add_argument(
        "--gold-source",
        type=Path,
        default=None,
        help=(
            "hotpot_dev_distractor_v1.json: обязателен для "
            "--prefer gold-sentence, только в нём context содержит "
            "gold-абзацы"
        ),
    )
    select_parser.add_argument(
        "--max-per-title",
        type=int,
        default=1,
        help=(
            "сколько чанков одной статьи разрешено взять; 1 — дедупликация "
            "титулов, как во всех опубликованных ранах"
        ),
    )
    select_parser.set_defaults(handler=command_select)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    return int(args.handler(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOG.error("Прервано")
        raise SystemExit(130)
    except Exception as error:
        LOG.error("Не удалось: %s", error)
        raise
