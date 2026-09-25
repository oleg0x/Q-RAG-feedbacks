#!/usr/bin/env python3
"""Сводная таблица серии Search-R1: наши три руки против их опубликованной.

Отдельный скрипт, а не колонка в ``RESULTS.md``, по двум причинам. Витрина
``runs_index.py`` строит по таблице на датасет и внутри датасета сравнивает
раны между собой — а здесь нужно поперечное сечение: одна строка на датасет,
семь датасетов рядом. И числа Search-R1 — внешние: они не считаются из наших
ранов, а взяты из статьи, поэтому лежат константой с указанием источника.

    python src/searchr1_table.py                 напечатать таблицы
    python src/searchr1_table.py --update-docs   вставить их в docs/searchr1.md
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Sequence

import runlib


LOG = logging.getLogger("searchr1-table")

DOC = runlib.REPO / "docs" / "searchr1.md"
TABLE_START = "<!-- BEGIN GENERATED TABLE -->"
TABLE_END = "<!-- END GENERATED TABLE -->"

# arXiv:2503.09516v5, Table 5, колонка Qwen2.5-7b-Base/Instruct. Взято
# дословно; здесь константа, потому что из наших ранов это не считается.
SEARCHR1: dict[str, dict[str, float]] = {
    "sr1_nq": {"direct": 0.134, "rag": 0.349, "search_r1": 0.480},
    "sr1_triviaqa": {"direct": 0.408, "rag": 0.585, "search_r1": 0.638},
    "sr1_popqa": {"direct": 0.140, "rag": 0.392, "search_r1": 0.457},
    "sr1_hotpotqa": {"direct": 0.183, "rag": 0.299, "search_r1": 0.433},
    "sr1_2wiki": {"direct": 0.250, "rag": 0.235, "search_r1": 0.382},
    "sr1_musique": {"direct": 0.031, "rag": 0.058, "search_r1": 0.196},
    "sr1_bamboogle": {"direct": 0.120, "rag": 0.208, "search_r1": 0.432},
}

# Датасеты, которых не видел при обучении никто из двоих. Только на них обе
# стороны в равном положении: Search-R1 учился на NQ+HotpotQA, линия A — на
# HotpotQA+2Wiki.
NEUTRAL = ("sr1_triviaqa", "sr1_popqa", "sr1_musique", "sr1_bamboogle")

ARMS = (
    ("noctx", "no retrieval"),
    ("zeroshot", "Прямой поиск, zero-shot GTE, 6 шагов, N=2"),
    ("best", "Прямой поиск, обученная башня, 6 шагов, N=2"),
)

SHORT = {
    "sr1_nq": "NQ", "sr1_triviaqa": "TriviaQA", "sr1_popqa": "PopQA",
    "sr1_hotpotqa": "HotpotQA", "sr1_2wiki": "2wiki",
    "sr1_musique": "MuSiQue", "sr1_bamboogle": "Bamboogle",
}


def collect() -> dict[str, dict[str, dict[str, Any]]]:
    """Метрики серии: датасет → рука → metrics.json.

    Раны ищутся по метке из манифеста, а не по имени каталога: метка — то же,
    чем они склеиваются в парные тесты ``runs_index.DEFAULT_PAIRS``, и
    расхождение между двумя способами адресации было бы источником тихой
    ошибки.
    """
    by_label = {label: arm for arm, label in ARMS}
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for run in runlib.iter_run_dirs():
        manifest = runlib.read_manifest(run)
        config = manifest.get("config", {})
        dataset = runlib.dataset_key(config)
        if dataset not in SEARCHR1 or manifest.get("status") != "ok":
            continue
        arm = by_label.get(config.get("label"))
        metrics = runlib.read_metrics(run)
        if arm is None or not metrics:
            continue
        found.setdefault(dataset, {})[arm] = {**metrics, "run_id": run.name}
    return found


def headline(metrics: dict[str, Any]) -> float | None:
    """Главное число строки: EM максимумом по алиасам, как считает Search-R1."""
    value = metrics.get("em_alias", metrics.get("em"))
    return None if value is None else float(value)


def cell(value: float | None, bold: bool = False) -> str:
    if value is None:
        return "—"
    text = f"{value * 100:.1f}"
    return f"**{text}**" if bold else text


def render_main(found: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "| Датасет | вопросов | без контекста | zero-shot | **обученная** "
        "| Search-R1-base 7B | Δ к Search-R1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, reference in SEARCHR1.items():
        arms = found.get(dataset, {})
        best = headline(arms["best"]) if "best" in arms else None
        theirs = reference["search_r1"]
        delta = None if best is None else best - theirs
        lines.append(
            f"| {SHORT[dataset]} | {runlib.dataset_meta(dataset)['examples']} "
            f"| {cell(headline(arms['noctx']) if 'noctx' in arms else None)} "
            f"| {cell(headline(arms['zeroshot']) if 'zeroshot' in arms else None)} "
            f"| {cell(best, bold=True)} "
            f"| {cell(theirs)} "
            f"| {'—' if delta is None else f'{delta * 100:+.1f}'} |"
        )
    return "\n".join(lines)


def average(found, datasets: Sequence[str], arm: str) -> float | None:
    values = [
        headline(found[dataset][arm])
        for dataset in datasets
        if arm in found.get(dataset, {})
    ]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if len(values) == len(datasets) else None


def render_averages(found: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Средние по двум наборам: все семь и только нейтральные четыре.

    Среднее по семи сопоставимо с колонкой Avg. их таблицы, но обе стороны там
    считают по своим in-domain. Среднее по четырём — единственное, где ни у
    кого нет преимущества обучающего набора.
    """
    groups = (
        ("все семь", tuple(SEARCHR1)),
        ("четыре нейтральных", NEUTRAL),
    )
    lines = [
        "| Набор | без контекста | zero-shot | **обученная** | Search-R1-base 7B |",
        "|---|---:|---:|---:|---:|",
    ]
    for title, datasets in groups:
        theirs = sum(SEARCHR1[d]["search_r1"] for d in datasets) / len(datasets)
        lines.append(
            f"| {title} ({len(datasets)}) "
            f"| {cell(average(found, datasets, 'noctx'))} "
            f"| {cell(average(found, datasets, 'zeroshot'))} "
            f"| {cell(average(found, datasets, 'best'), bold=True)} "
            f"| {cell(theirs)} |"
        )
    return "\n".join(lines)


def render_scale(found: dict[str, dict[str, dict[str, Any]]]) -> str:
    """Их собственная шкала рядом с нашей: без ретривала → RAG → Search-R1."""
    lines = [
        "| Датасет | наш без контекста | их Direct Inference | наш zero-shot "
        "| их RAG (3 пассажа) | наша обученная | их Search-R1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, reference in SEARCHR1.items():
        arms = found.get(dataset, {})
        lines.append(
            f"| {SHORT[dataset]} "
            f"| {cell(headline(arms['noctx']) if 'noctx' in arms else None)} "
            f"| {cell(reference['direct'])} "
            f"| {cell(headline(arms['zeroshot']) if 'zeroshot' in arms else None)} "
            f"| {cell(reference['rag'])} "
            f"| {cell(headline(arms['best']) if 'best' in arms else None, bold=True)} "
            f"| {cell(reference['search_r1'])} |"
        )
    return "\n".join(lines)


def build(found: dict[str, dict[str, dict[str, Any]]]) -> str:
    missing = [
        f"{SHORT[dataset]}/{arm}"
        for dataset in SEARCHR1
        for arm, _ in ARMS
        if arm not in found.get(dataset, {})
    ]
    blocks = [
        "### Главная таблица\n\nEM по алиасам, в процентах. Определение EM то "
        "же, что у Search-R1: максимум по всему `golden_answers`.\n\n"
        + render_main(found),
        "### Средние\n\n" + render_averages(found),
        "### Обе шкалы целиком\n\n" + render_scale(found),
    ]
    if missing:
        blocks.append(
            "> Неполные данные — нет ранов: " + ", ".join(missing)
        )
    return "\n\n".join(blocks)


def splice(table: str) -> None:
    text = DOC.read_text(encoding="utf-8")
    block = f"{TABLE_START}\n\n{table}\n\n{TABLE_END}"
    if TABLE_START not in text or TABLE_END not in text:
        raise SystemExit(f"В {DOC.name} нет маркеров {TABLE_START} / {TABLE_END}")
    head, _, rest = text.partition(TABLE_START)
    _, _, tail = rest.partition(TABLE_END)
    DOC.write_text(head + block + tail, encoding="utf-8")
    LOG.info("Обновлён %s", DOC.name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-docs", action="store_true")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    found = collect()
    LOG.info(
        "Найдено ранов: %d из %d",
        sum(len(arms) for arms in found.values()),
        len(SEARCHR1) * len(ARMS),
    )
    table = build(found)
    print(table)
    if args.update_docs:
        splice(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
