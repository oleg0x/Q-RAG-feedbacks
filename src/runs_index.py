#!/usr/bin/env python3
"""Build runs/INDEX.md and RESULTS.md from run manifests.

INDEX.md is the registry: one line per run, so "what has already been run" is
answered by one short file instead of walking directories. RESULTS.md holds
the results tables: they are built by ``report_phase0.py``, with the run list
taken from manifests rather than typed as flags by hand.

    python src/runs_index.py                  rebuild both files
    python src/runs_index.py --only-index     rebuild INDEX.md only

Comparisons include runs with ``"status": "ok"`` and non-empty metrics,
grouped by dataset: each dataset gets its own table with its own corpus
coverage ceiling, because 7,405 HotpotQA questions and 12,576 2Wiki questions
are different tasks, not different rows of one. Within a dataset only runs
with the modal number of examples are kept: a 20-question smoke run must not
be mixed with a full one.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from pathlib import Path
from typing import Any, Sequence

import runlib


LOG = logging.getLogger("runs-index")

RESULTS_FILE = runlib.REPO / "RESULTS.md"
TABLE_START = "<!-- BEGIN GENERATED TABLE -->"
TABLE_END = "<!-- END GENERATED TABLE -->"

# Pairs for the paired McNemar test: Q-RAG must be compared with first-stage
# retrieval at the same chunk budget, otherwise the table says nothing about
# the contribution of Q-RAG.
DEFAULT_PAIRS = (
    ("GTE top-2", "Q-RAG, 2 steps"),
    ("GTE top-4", "Q-RAG, 4 steps"),
    ("GTE top-6", "Q-RAG, 6 steps"),
    ("no retrieval", "GTE top-6"),
    # Reranking ceiling over the same pool: the gap to GTE at the same chunk
    # budget is what reranker training can gain.
    ("GTE top-2", "title oracle, 2 chunks"),
    ("GTE top-4", "title oracle, 4 chunks"),
    ("GTE top-6", "title oracle, 6 chunks"),
    # Cost of picking a chunk by rank: the same oracle over the same pool, but
    # the chunk of a gold title is chosen by content. The difference is what
    # the title oracle could not reach and a chunk reranker can.
    ("title oracle, 2 chunks", "chunk oracle, 2 chunks"),
    ("title oracle, 4 chunks", "chunk oracle, 4 chunks"),
    ("title oracle, 6 chunks", "chunk oracle, 6 chunks"),
    # What the N=2 quota adds to chunk-aware selection: the same upper bound,
    # but the reranker is allowed what relaxed deduplication allows.
    ("chunk oracle, 2 chunks", "chunk oracle, 2 chunks, 2 per title"),
    ("chunk oracle, 6 chunks", "chunk oracle, 6 chunks, 2 per title"),
    # Relaxed versus strict deduplication at the same chunk budget.
    ("GTE top-4", "GTE top-4, 2 chunks per title"),
    ("GTE top-6", "GTE top-6, 2 chunks per title"),
    # Refresh versus fixed at the same step budget: the modes differ, so the
    # pair is needed as a paired test, not as neighbouring table rows.
    ("GTE top-2", "GTE refresh 1024, 2 steps"),
    ("GTE top-4", "GTE refresh 1024, 4 steps"),
    ("GTE top-6", "GTE refresh 1024, 6 steps"),
    # Cost of query truncation: the same six refresh steps at 256 and 1024 tokens.
    ("GTE refresh 256, 6 steps", "GTE refresh 1024, 6 steps"),
    # Contribution of line A training: the same tower and budget, only the
    # checkpoint differs. One pair serves all datasets: labels are unique within
    # a group, so it expands into a separate test on each of them.
    (
        "Direct search, zero-shot GTE, 6 steps, N=2",
        "Direct search, trained tower, 6 steps, N=2",
    ),
    # Trained direct search versus the strongest first-stage retrieval at the
    # same chunk budget: without this row line A is compared only with itself.
    (
        "GTE top-6, 2 chunks per title",
        "Direct search, trained tower, 6 steps, N=2",
    ),
    # Cost of checkpoint selection. The peak of the eval curve is about one
    # sigma above the plateau, so the "best" checkpoint may be an argmax over
    # noise; a paired test on the full dev set shows whether the choice mattered.
    (
        "Direct search, trained tower, 6 steps, N=2",
        "Direct search, last checkpoint, 6 steps, N=2",
    ),
    # How much the reader answers from memory without any context. On
    # multi-hop datasets this hardly matters, but on the single-hop Search-R1
    # benchmarks it is central: their Direct Inference without retrieval gets
    # .408 on TriviaQA against a final .638, so most of the number is the
    # model's own knowledge.
    ("no retrieval", "Direct search, zero-shot GTE, 6 steps, N=2"),
    # Inference-time quota ablation, both arms. The baseline is N=2 everywhere,
    # not because it is better but because it is the default: the sign of the
    # test reads as "what deviating from the default gives". Zero-shot is the
    # control: it had no N during training, so its curve over N is a pure
    # inference effect, and the difference between the shapes of the two
    # curves is the cost of the train/test mismatch.
    (
        "Direct search, trained tower, 6 steps, N=2",
        "Direct search, trained tower, 6 steps, N=1",
    ),
    (
        "Direct search, trained tower, 6 steps, N=2",
        "Direct search, trained tower, 6 steps, no quota",
    ),
    (
        "Direct search, zero-shot GTE, 6 steps, N=2",
        "Direct search, zero-shot GTE, 6 steps, N=1",
    ),
    (
        "Direct search, zero-shot GTE, 6 steps, N=2",
        "Direct search, zero-shot GTE, 6 steps, no quota",
    ),
)


def percent(value: Any) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{value * 100:.2f}"


def collect() -> list[tuple[Path, dict[str, Any], dict[str, Any] | None]]:
    collected = []
    for run in runlib.iter_run_dirs():
        manifest = runlib.read_manifest(run)
        collected.append((run, manifest, runlib.read_metrics(run)))
    # Row order comes from the config's order field: baselines from weakest to
    # strongest, oracle last. Runs without order go to the end, by name.
    return sorted(
        collected,
        key=lambda row: (row[1].get("config", {}).get("order", 10_000), row[0].name),
    )


def render_index(rows: Sequence[tuple[Path, dict[str, Any], dict[str, Any] | None]]) -> str:
    lines = [
        "# Run registry",
        "",
        "Generated by `python exp.py index`. Do not edit by hand.",
        "",
        "| Run | Dataset | Label | Reranker | Steps | Status | EM | F1 | Judge | title EM |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for run, manifest, metrics in rows:
        config = manifest.get("config", {})
        retrieve = config.get("retrieve", {})
        metrics = metrics or {}
        reranker = retrieve.get("reranker") or retrieve.get("context") or "—"
        rel = run.relative_to(runlib.RUNS).as_posix()
        lines.append(
            f"| [`{run.name}`]({rel}/manifest.json) "
            f"| {runlib.dataset_key(config)} "
            f"| {config.get('label', '—')} "
            f"| {reranker} "
            f"| {retrieve.get('steps', '—')} "
            f"| {manifest.get('status', '—')}"
            f"{' (backfill)' if manifest.get('backfilled') else ''} "
            f"| {percent(metrics.get('em'))} "
            f"| {percent(metrics.get('f1'))} "
            f"| {percent(metrics.get('judge'))} "
            f"| {percent(metrics.get('title_em'))} |"
        )

    lines += ["", "## Legacy file names", ""]
    lines += [
        "Runs made before the registry existed lived in a flat `runs/` with",
        "parameters encoded in file names; `src/migrate_runs.py` moved them into",
        "run directories. The exact commands of every run are in its `cmd.sh`.",
    ]

    lines += [
        "",
        "## Archive",
        "",
        "Runs excluded from comparison are moved by `exp.py archive` to",
        "`runs/_archive/`, with the reason recorded in its README; the",
        "directory appears with the first archived run.",
        "",
    ]
    return "\n".join(lines)


def comparable(
    rows: Sequence[tuple[Path, dict[str, Any], dict[str, Any] | None]]
) -> dict[str, list[tuple[Path, dict[str, Any], dict[str, Any]]]]:
    """Runs grouped by dataset; within a dataset, only mutually comparable ones.

    The selection has two stages, and both are needed. Datasets are separated
    strictly: HotpotQA, 2Wiki and MuSiQue differ in questions and corpus
    coverage ceiling, so their rows can neither share a table nor be compared
    by a paired test. Within a dataset everything that does not match the
    modal number of examples is dropped: this keeps 20-question smoke runs
    out of the results tables.
    """
    groups: dict[str, list[tuple[Path, dict[str, Any], dict[str, Any]]]] = {}
    for run, manifest, metrics in rows:
        if manifest.get("status") != "ok" or not metrics or not metrics.get("samples"):
            continue
        dataset = runlib.dataset_key(manifest.get("config", {}))
        groups.setdefault(dataset, []).append((run, manifest, metrics))

    selected: dict[str, list[tuple[Path, dict[str, Any], dict[str, Any]]]] = {}
    # Table order follows the dataset registry, not directory traversal
    # order: sections must not reshuffle from one run to the next.
    for dataset in runlib.DATASETS:
        members = groups.pop(dataset, [])
        if not members:
            continue
        sizes = [metrics["samples"] for _, _, metrics in members]
        full = max(set(sizes), key=sizes.count)
        dropped = [run.name for run, _, metrics in members if metrics["samples"] != full]
        if dropped:
            LOG.warning(
                "Excluded from table %s (number of examples is not %d): %s",
                dataset, full, ", ".join(dropped),
            )
        selected[dataset] = [
            (run, manifest, metrics)
            for run, manifest, metrics in members
            if metrics["samples"] == full
        ]
    for dataset in groups:
        LOG.warning("Dataset %s is not registered in runlib.DATASETS", dataset)
    return selected


def build_dataset_table(
    dataset: str,
    runs: Sequence[tuple[Path, dict[str, Any], dict[str, Any]]],
) -> str | None:
    meta = runlib.dataset_meta(dataset)
    labels = {manifest["config"]["label"] for _, manifest, _ in runs}
    # The path comes from runlib rather than a string literal: the subprocess
    # runs with cwd=REPO while the module lives in src/, and a bare name here
    # silently broke `make report`, leaving the RESULTS.md table stale.
    argv: list[str] = [
        str(runlib.VENV_PYTHON),
        str(runlib.pipeline_script("report_phase0.py")),
    ]
    # Both flags are optional, for different reasons: there is no ceiling
    # without gold titles, and aliases exist only where the official eval takes
    # EM as the maximum over the list of acceptable answers.
    if meta.get("coverage") is not None:
        argv += ["--coverage", str(meta["coverage"])]
    if meta.get("aliases") is not None:
        argv += ["--aliases", str(meta["aliases"])]
    for run, manifest, _ in runs:
        label = manifest["config"]["label"]
        argv += ["--run", f"{label}={run / runlib.JUDGE_FILE}"]
    for baseline, variant in DEFAULT_PAIRS:
        if baseline in labels and variant in labels:
            argv += ["--pair", baseline, variant]
    LOG.info("Recomputing table %s: %d runs", dataset, len(runs))
    completed = subprocess.run(
        argv, cwd=runlib.REPO, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        LOG.error("report_phase0.py failed:\n%s", completed.stderr.strip()[-2000:])
        return None
    # Drop the trailing JSON dump: the file needs only the tables.
    table, _, _ = completed.stdout.partition("\n{")
    return table.rstrip()


def build_results(rows) -> str | None:
    groups = comparable(rows)
    if not groups:
        LOG.warning("No comparable runs, leaving RESULTS.md untouched")
        return None
    blocks = []
    for dataset, runs in groups.items():
        table = build_dataset_table(dataset, runs)
        if table is None:
            return None
        meta = runlib.dataset_meta(dataset)
        samples = runs[0][2]["samples"]
        blocks.append(
            f"### {meta['label']} — {samples} examples\n\n{table}"
        )
    return "\n\n".join(blocks)


def splice_results(table: str) -> None:
    """Replace the generated block in RESULTS.md, keeping the hand-written text."""
    text = RESULTS_FILE.read_text(encoding="utf-8")
    block = f"{TABLE_START}\n\n{table}\n\n{TABLE_END}"
    if TABLE_START in text and TABLE_END in text:
        head, _, rest = text.partition(TABLE_START)
        _, _, tail = rest.partition(TABLE_END)
        RESULTS_FILE.write_text(head + block + tail, encoding="utf-8")
    else:
        raise SystemExit(
            f"{RESULTS_FILE.name} has no {TABLE_START} / {TABLE_END} markers: "
            "nowhere to insert the table"
        )
    LOG.info("Updated %s", RESULTS_FILE.name)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-index", action="store_true")
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
    rows = collect()
    runlib.INDEX_FILE.write_text(render_index(rows) + "\n", encoding="utf-8")
    LOG.info("Updated %s (%d runs)", runlib.INDEX_FILE.name, len(rows))
    if args.only_index:
        return 0
    table = build_results(rows)
    if table:
        splice_results(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
