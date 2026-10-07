#!/usr/bin/env python3
"""Summary table of the Search-R1 series: our three arms against their published one.

A separate script rather than a column in ``RESULTS.md``, for two reasons.
``runs_index.py`` builds one table per dataset and compares runs within it,
while here a cross-section is needed: one row per dataset, seven datasets
side by side. And the Search-R1 numbers are external: they are not computed
from our runs but taken from the paper, so they are kept as a constant with
the source cited.

    python src/searchr1_table.py                 print the tables
    python src/searchr1_table.py --update-docs   also splice them into DOC
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

# arXiv:2503.09516v5, Table 5, column Qwen2.5-7b-Base/Instruct. Copied
# verbatim; a constant because it cannot be computed from our runs.
SEARCHR1: dict[str, dict[str, float]] = {
    "sr1_nq": {"direct": 0.134, "rag": 0.349, "search_r1": 0.480},
    "sr1_triviaqa": {"direct": 0.408, "rag": 0.585, "search_r1": 0.638},
    "sr1_popqa": {"direct": 0.140, "rag": 0.392, "search_r1": 0.457},
    "sr1_hotpotqa": {"direct": 0.183, "rag": 0.299, "search_r1": 0.433},
    "sr1_2wiki": {"direct": 0.250, "rag": 0.235, "search_r1": 0.382},
    "sr1_musique": {"direct": 0.031, "rag": 0.058, "search_r1": 0.196},
    "sr1_bamboogle": {"direct": 0.120, "rag": 0.208, "search_r1": 0.432},
}

# Datasets that neither side saw in training. Only on these are both sides on
# equal footing: Search-R1 was trained on NQ+HotpotQA, line A on
# HotpotQA+2Wiki.
NEUTRAL = ("sr1_triviaqa", "sr1_popqa", "sr1_musique", "sr1_bamboogle")

ARMS = (
    ("noctx", "no retrieval"),
    ("zeroshot", "Direct search, zero-shot GTE, 6 steps, N=2"),
    ("best", "Direct search, trained tower, 6 steps, N=2"),
)

SHORT = {
    "sr1_nq": "NQ", "sr1_triviaqa": "TriviaQA", "sr1_popqa": "PopQA",
    "sr1_hotpotqa": "HotpotQA", "sr1_2wiki": "2wiki",
    "sr1_musique": "MuSiQue", "sr1_bamboogle": "Bamboogle",
}


def collect() -> dict[str, dict[str, dict[str, Any]]]:
    """Series metrics: dataset → arm → metrics.json.

    Runs are found by their manifest label, not by directory name: the label
    is also what joins them into the paired tests of
    ``runs_index.DEFAULT_PAIRS``, and a mismatch between two addressing
    schemes would be a source of silent errors.
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
    """Headline number of a row: EM as the max over aliases, as in Search-R1."""
    value = metrics.get("em_alias", metrics.get("em"))
    return None if value is None else float(value)


def cell(value: float | None, bold: bool = False) -> str:
    if value is None:
        return "—"
    text = f"{value * 100:.1f}"
    return f"**{text}**" if bold else text


def render_main(found: dict[str, dict[str, dict[str, Any]]]) -> str:
    lines = [
        "| Dataset | questions | no context | zero-shot | **trained** "
        "| Search-R1-base 7B | Δ vs Search-R1 |",
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
    """Averages over two sets: all seven and only the four neutral ones.

    The average over seven is comparable with the Avg. column of their table,
    but there both sides include their own in-domain sets. The average over
    four is the only one where nobody has a training-set advantage.
    """
    groups = (
        ("all seven", tuple(SEARCHR1)),
        ("four neutral", NEUTRAL),
    )
    lines = [
        "| Set | no context | zero-shot | **trained** | Search-R1-base 7B |",
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
    """Their own scale next to ours: no retrieval → RAG → Search-R1."""
    lines = [
        "| Dataset | our no context | their Direct Inference | our zero-shot "
        "| their RAG (3 passages) | our trained | their Search-R1 |",
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
        "### Main table\n\nAlias EM, in percent. EM is defined as in "
        "Search-R1: the maximum over all of `golden_answers`.\n\n"
        + render_main(found),
        "### Averages\n\n" + render_averages(found),
        "### Both scales in full\n\n" + render_scale(found),
    ]
    if missing:
        blocks.append(
            "> Incomplete data, missing runs: " + ", ".join(missing)
        )
    return "\n\n".join(blocks)


def splice(table: str) -> None:
    text = DOC.read_text(encoding="utf-8")
    block = f"{TABLE_START}\n\n{table}\n\n{TABLE_END}"
    if TABLE_START not in text or TABLE_END not in text:
        raise SystemExit(f"{DOC.name} has no {TABLE_START} / {TABLE_END} markers")
    head, _, rest = text.partition(TABLE_START)
    _, _, tail = rest.partition(TABLE_END)
    DOC.write_text(head + block + tail, encoding="utf-8")
    LOG.info("Updated %s", DOC.name)


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
        "Runs found: %d of %d",
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
