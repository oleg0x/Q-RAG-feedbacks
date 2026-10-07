#!/usr/bin/env python3
"""What the GTE top-100 candidates contain and how much of it is reachable.

A reranker cannot pick what is not in the pool. Before investing in reranker
training we need two numbers: the share of questions solvable from this pool
at all, and how many answers a perfect selection from it would yield. Both
are computed here from a run logged with ``--log-candidates full``.

``report``
    Pool diagnostics as JSON: title recall/EM@100, the rank distribution of
    gold titles, the title oracle, and the share of examples where the title
    is found but the gold sentence did not land in the chunk.

``select``
    Builds a retrieval JSONL variant that takes, from the same 100
    candidates, the k chunks maximizing gold-title coverage. This is a
    **title oracle**, not an answer oracle: it shows the reranking ceiling
    over this pool, not the absolute ceiling of the task.

The pool comes from ``retrieval_hops[0]["first_stage_candidate_idx"]``, the
raw GTE ranking before any exclusions. Titles and chunk texts are resolved
by row ID through the same byte-offset table as retrieval itself, so
"row 12345" means the same thing here and in the run.

Examples:

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

# Chunk budgets for the oracle selection. They match the canonical GTE
# top-2/4/6 rows, otherwise there would be nothing to compare against.
DEFAULT_BUDGETS = (2, 4, 6)

# Rank of a gold title absent from the pool. A sentinel value rather than
# None, so that the histogram and the median are computed over the same field.
MISSING_RANK = -1


# --------------------------------------------------------------------------
# pool and corpus


def pool_row_ids(record: dict[str, Any]) -> list[int]:
    """Raw GTE top-k ranking from the first step of a run.

    In ``fixed`` mode the pool is computed once and reused at every step, so
    the first step is enough. ``first_stage_candidate_idx`` rather than
    ``candidate_idx``: with reranker=none the latter equals the former only
    at step zero; later steps already contain deduplication exclusions.
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
    """Records with the texts of their pool, one corpus read per record."""
    started = time.monotonic()
    for number, record in enumerate(records, start=1):
        row_ids = pool_row_ids(record)
        texts = [item["contents"] for item in corpus.read_rows(row_ids)]
        yield record, row_ids, texts
        if progress_every and number % progress_every == 0:
            LOG.info(
                "%d records, %.0f s (%.1f rec/s)",
                number,
                time.monotonic() - started,
                number / max(time.monotonic() - started, 1e-9),
            )


# --------------------------------------------------------------------------
# sentence matching


def normalize_text(text: str) -> str:
    """One canonical form for a gold sentence and a Wiki-18 chunk.

    The 2017 and 2018 dumps differ in HTML entities and whitespace, as do the
    titles (see ``normalize_title``); comparing raw strings is meaningless.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(folded.split())


def sentence_in_chunk(sentence: str, chunk: str) -> str:
    """How much of a gold sentence landed in a chunk: full, partial or none.

    A Wiki-18 chunk is exactly 100 words and ignores sentence boundaries, so a
    gold sentence is often split across two chunks. Strict containment
    understates the hit rate and half-matching overstates it: the true value
    lies in between, and two bounds are more useful than one number with an
    unclear error.
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
    """Verdict over all gold sentences of a title at once.

    ``full`` only if every sentence landed in the chunk: if a title has two
    and one stayed in the neighbouring chunk, the reader still answers from
    incomplete context. ``none`` means none landed; anything else is
    ``partial``.
    """
    verdicts = [sentence_in_chunk(sentence, chunk) for sentence in sentences]
    if verdicts and all(value == "full" for value in verdicts):
        return "full"
    if not verdicts or all(value == "none" for value in verdicts):
        return "none"
    return "partial"


# Lower is better. The order is "full > half > none" rather than
# "present > absent": a sentence split at a chunk boundary is still better
# than a chunk without the needed text at all.
VERDICT_PENALTY = {"full": 0, "partial": 1, "none": 2}


def gold_sentences_by_title(sample: dict[str, Any]) -> dict[str, list[str]]:
    """Gold sentences from the distractor file, grouped by title.

    Only the distractor file contains gold paragraphs: in the fullwiki file
    the ``context`` field holds the output of the original retriever.
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
# chunk selection


def first_positions(normalized_pool: Sequence[str]) -> dict[str, int]:
    """Title -> its best position in the pool. Computed once per record."""
    best_position: dict[str, int] = {}
    for position, title in enumerate(normalized_pool):
        best_position.setdefault(title, position)
    return best_position


def sentence_penalties(
    normalized_pool: Sequence[str],
    texts: Sequence[str],
    gold_sentences: Mapping[str, Sequence[str]],
) -> list[int]:
    """Penalty of every pool chunk according to ``VERDICT_PENALTY``.

    Computed only for titles whose gold sentences are known: for the others
    all chunks get the same penalty, and the order within a title reduces to
    the GTE rank, which is exactly what the title oracle did.
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
    """Title -> its positions in the pool, best first.

    The order within a title is ``(penalty, GTE rank)``. Without penalties it
    is just the rank, i.e. the original "best-ranked chunk of the article".
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
    """Positions of the k chunks that maximize gold-title coverage.

    First one chunk per found gold title, then, if ``max_per_title`` exceeds
    one, second chunks of the same titles, and only then filling from the top
    of the ranking. The per-title quota also holds while filling: it is an
    invariant of the runs this selection is compared with.

    ``penalties`` sets the preference between chunks of one title: 0 beats 1.
    The title oracle passes none and takes the best-ranked GTE chunk; the
    chunk-aware oracle puts first the chunk containing the gold sentence
    (``VERDICT_PENALTY``). Across titles the penalty decides nothing: the
    title is taken anyway, the only question is which chunk.

    Second chunks are taken in rounds, not consecutively, and only if the
    chunk carries gold text at all (penalty below ``VERDICT_PENALTY["none"]``):
    otherwise the budget would go to a second chunk of an already covered
    article instead of a new title.

    The selection order (round by round, by rank within a round) makes the
    result prefix-consistent across budgets: the 2-chunk selection is a prefix
    of the 6-chunk one. Hence the k=2/4/6 rows are comparable as one
    algorithm with different budgets.
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
    """Rank of each gold title in the GTE ranking, in ascending order.

    Ranks are sorted rather than kept in gold order: for a bridge question
    what matters is not which title comes first in the annotation but how
    deep the worst of the needed ones lies.
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
    """Rank histogram with edges such as 1, 5, 10, ... plus "missing"."""
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

    # The title was found but its chunk lacks the needed sentence. Counted per
    # title (an example has two) and separately per example.
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

        # Chunk granularity: the title is in the pool, but is the sentence?
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
            # The chunk the selection would actually take: the best-ranked one.
            verdict = chunk_verdict(sentences, texts[positions[0]])
            sentence_titles[verdict] += 1
            sentence_titles["total"] += 1
            if verdict != "full":
                lost_here = True
            # The same question if the selection could take any chunk of the
            # article from the pool: the upper bound for granularity, which is
            # exactly what `select --prefer gold-sentence` achieves.
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
                "Ranks of gold titles in the GTE ranking, 0-based, sorted "
                "ascending: 'best' is the easier-found gold title, 'second' "
                f"the next one. {MISSING_RANK} means 'not in the pool'."
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
                "Given the title is in the pool, does its chunk contain the "
                "gold sentence. 'full': the whole sentence is inside the "
                "chunk; 'partial': half of it (Wiki-18 chunks are cut every "
                "100 words and split sentences); 'none': not found. "
                "'selected_chunk': the best-ranked chunk of the title, i.e. "
                "the one selection takes with deduplication on; "
                "'any_chunk_in_pool': the upper bound if any chunk of the "
                "article in the pool could be taken."
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
    # The pool is the same at every step and this variant has no separate
    # hops: no need to log the hundred candidates again, run A saved them.
    result["retrieval_hops"] = [
        {"step": step, "selected_idx": int(row_ids[position]), "pool_rank": position}
        for step, position in enumerate(positions)
    ]
    result["retrieved_titles"] = [pool_titles[position] for position in positions]
    result.update(title_metrics(gold, result["retrieved_titles"]))
    # Not 'none': build_eval_variants.py must refuse to truncate this run,
    # it is not a first-stage baseline.
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
        LOG.info("Wrote %s", destination)
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
    LOG.info("Wrote %s", destination)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser("report", help="pool diagnostics as JSON")
    add_corpus_arguments(report_parser)
    report_parser.add_argument(
        "--gold-source",
        type=Path,
        default=None,
        help=(
            "hotpot_dev_distractor_v1.json: the only dev file whose context "
            "contains gold paragraphs. Without it only the sentence-level "
            "diagnostics are skipped"
        ),
    )
    report_parser.add_argument("--coverage", type=Path, required=True)
    report_parser.add_argument(
        "--budget", type=int, action="append", default=None,
        help=f"chunk budgets for the oracle selection; default {DEFAULT_BUDGETS}",
    )
    report_parser.add_argument("--output", type=Path, default=None)
    report_parser.set_defaults(handler=command_report)

    select_parser = subparsers.add_parser(
        "select", help="JSONL variant with the title-oracle selection"
    )
    add_corpus_arguments(select_parser)
    select_parser.add_argument("--steps", type=int, required=True)
    select_parser.add_argument("--output", type=Path, required=True)
    select_parser.add_argument(
        "--prefer",
        choices=("rank", "gold-sentence"),
        default="rank",
        help=(
            "which chunk of a gold title to take: 'rank' is the best GTE rank "
            "(title oracle), 'gold-sentence' is the one containing the gold "
            "sentence (full > half > none), ties broken by rank. The second "
            "mode is the fair upper bound for a chunk reranker: it picks from "
            "the same chunks as the reranker"
        ),
    )
    select_parser.add_argument(
        "--gold-source",
        type=Path,
        default=None,
        help=(
            "hotpot_dev_distractor_v1.json: required for "
            "--prefer gold-sentence; only its context contains gold "
            "paragraphs"
        ),
    )
    select_parser.add_argument(
        "--max-per-title",
        type=int,
        default=1,
        help=(
            "how many chunks of one article may be taken; 1 means title "
            "deduplication, as in all published runs"
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
        LOG.error("Interrupted")
        raise SystemExit(130)
    except Exception as error:
        LOG.error("Failed: %s", error)
        raise
