#!/usr/bin/env python3
"""Shared helpers for the run registry: paths, manifests, code and input identity.

Each run is a directory ``runs/<YYYY-MM-DD>-<slug>/``. Small text files
(``manifest.json``, ``metrics.json``, ``cmd.sh``, ``notes.md``) are tracked by
git and form the experiment history; heavy outputs (``retrieval.jsonl``,
``answer_judge.json``, ``log.txt``) live next to them but outside the
repository.

The module deliberately avoids torch and does not compute the corpus SHA-256:
corpus and encoder identity are already recorded in the index manifest written
by ``build_index_wiki_gte.py`` / ``build_index_wiki_qrag.py`` and are read
from there.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

from build_index_wiki_gte import (
    atomic_write_json,
    cached_sha256,
    package_version,
    read_json,
    utc_now,
)


# Modules live in src/, while the registry, configs and docs are at the
# repository root, so REPO is one level above this file.
SRC = Path(__file__).resolve().parent
REPO = SRC.parent
RUNS = REPO / "runs"
SHARED = RUNS / "shared"
ARCHIVE = RUNS / "_archive"

MANIFEST_FILE = "manifest.json"
METRICS_FILE = "metrics.json"
CMD_FILE = "cmd.sh"
NOTES_FILE = "notes.md"
LOG_FILE = "log.txt"
RETRIEVAL_FILE = "retrieval.jsonl"
JUDGE_FILE = "answer_judge.json"

COVERAGE_FILE = SHARED / "hotpotqa_dev_title_coverage.json"
INDEX_FILE = RUNS / "INDEX.md"

# Datasets the results tables are built for. The key goes into the run config
# as ``dataset``; it determines both the coverage ceiling and which RESULTS.md
# table the row lands in. Runs on different datasets are not comparable: they
# differ in questions, number of examples and corpus ceiling.
#
# ``coverage`` is optional. The coverage ceiling is computed from gold titles,
# and none of the seven Search-R1 eval splits has them (see
# build_searchr1_eval.py), so they have ``coverage: None``; ``title_em`` and
# ``share_of_ceiling`` then stay ``null`` instead of turning into 100%.
#
# ``aliases`` is the dataset file from which ``export_eval_json.collect_aliases``
# reads acceptable answers besides the primary one. It is needed where the
# official eval takes EM as the maximum over aliases: without it TriviaQA,
# with its fourteen answers per question, is understated several-fold.
DEFAULT_DATASET = "hotpotqa_dev_fullwiki"
SEARCHR1 = SHARED / "searchr1"
DATASETS: dict[str, dict[str, Any]] = {
    "hotpotqa_dev_fullwiki": {
        "label": "HotpotQA dev fullwiki",
        "examples": 7405,
        "coverage": SHARED / "hotpotqa_dev_title_coverage.json",
    },
    "2wiki_dev": {
        "label": "2WikiMultiHopQA dev",
        "examples": 12576,
        "coverage": SHARED / "2wiki_dev_title_coverage.json",
    },
    "musique_ans_dev": {
        "label": "MuSiQue-Ans dev",
        "examples": 2417,
        "coverage": SHARED / "musique_ans_dev_title_coverage.json",
    },
    # Seven Search-R1 benchmarks from a single test.parquet. Separate keys
    # rather than reusing the three above: the questions are normalized (267
    # of them got a trailing "?"), gold comes from golden_answers and there are
    # no title metrics, so these rows must not share a table with the others.
    "sr1_nq": {
        "label": "NQ (Search-R1 test)",
        "examples": 3610,
        "coverage": None,
        "aliases": SEARCHR1 / "nq.jsonl",
    },
    "sr1_triviaqa": {
        "label": "TriviaQA (Search-R1 test)",
        "examples": 11313,
        "coverage": None,
        "aliases": SEARCHR1 / "triviaqa.jsonl",
    },
    "sr1_popqa": {
        "label": "PopQA (Search-R1 test)",
        "examples": 14267,
        "coverage": None,
        "aliases": SEARCHR1 / "popqa.jsonl",
    },
    "sr1_hotpotqa": {
        "label": "HotpotQA (Search-R1 test)",
        "examples": 7405,
        "coverage": None,
        "aliases": SEARCHR1 / "hotpotqa.jsonl",
    },
    "sr1_2wiki": {
        "label": "2WikiMultiHopQA (Search-R1 test)",
        "examples": 12576,
        "coverage": None,
        "aliases": SEARCHR1 / "2wikimultihopqa.jsonl",
    },
    "sr1_musique": {
        "label": "MuSiQue (Search-R1 test)",
        "examples": 2417,
        "coverage": None,
        "aliases": SEARCHR1 / "musique.jsonl",
    },
    "sr1_bamboogle": {
        "label": "Bamboogle (Search-R1 test)",
        "examples": 125,
        "coverage": None,
        "aliases": SEARCHR1 / "bamboogle.jsonl",
    },
}

MANIFEST_SCHEMA_VERSION = 1

# Scripts whose content affects a run's result. Their hashes go into the
# manifest, so "which code produced this file" is answered without git
# archaeology.
PIPELINE_SCRIPTS = (
    "fullwiki_qrag.py",
    "build_eval_variants.py",
    "candidate_pool.py",
    "build_index_wiki_gte.py",
    "build_index_wiki_qrag.py",
    "report_phase0.py",
    # The reader and judge are our fork, not the original script: the EM, F1
    # and reward of every run depend on its content.
    "answer_judge.py",
    "exp.py",
)

# Interpreter used for pipeline stages; defaults to the one running exp.py.
VENV_PYTHON = Path(os.environ.get("QRAG_PYTHON", sys.executable))


def pipeline_script(name: str) -> Path:
    """Path to a pipeline script: `exp.py` at the root, other modules in `src/`.

    Names in `PIPELINE_SCRIPTS` deliberately stay bare, without a directory:
    they also serve as manifest keys, and moving modules to `src/` must not
    split the registry into "before" and "after" by key name.
    """
    return REPO / name if name == "exp.py" else SRC / name


def run_dir(run_id: str) -> Path:
    if "/" in run_id or run_id.startswith("."):
        raise ValueError(f"Invalid run_id: {run_id!r}")
    return RUNS / run_id


def iter_run_dirs() -> Iterator[Path]:
    """Run directories in lexicographic (hence chronological) order.

    Only the runs of the main README table sit at the top of runs/; the rest
    are in runs/runs_history/. The registry and results tables are built from
    both levels, so moving a run between them changes no table row.
    """
    roots = (RUNS, RUNS / "runs_history")
    candidates = (
        path for root in roots if root.exists() for path in root.iterdir()
    )
    for path in sorted(candidates, key=lambda p: p.name):
        if not path.is_dir() or path.is_symlink():
            continue
        if path.name.startswith((".", "_")) or path.name == "shared":
            continue
        if (path / MANIFEST_FILE).exists():
            yield path


def read_manifest(run: Path) -> dict[str, Any]:
    return read_json(run / MANIFEST_FILE)


def write_manifest(run: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(run / MANIFEST_FILE, manifest)


def read_metrics(run: Path) -> dict[str, Any] | None:
    path = run / METRICS_FILE
    return read_json(path) if path.exists() else None


def write_metrics(run: Path, metrics: dict[str, Any]) -> None:
    atomic_write_json(run / METRICS_FILE, metrics)


def file_identity(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    """File identity. ``hash_file=False`` for multi-gigabyte inputs."""
    stat = path.stat()
    identity: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
    }
    if hash_file:
        identity["sha256"] = cached_sha256(
            str(path.resolve()), stat.st_size, stat.st_mtime_ns
        )
    return identity


def git(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *arguments),
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def code_identity() -> dict[str, Any]:
    """Commit, uncommitted-changes flag and pipeline script hashes."""
    status = git("status", "--porcelain", "--", "*.py")
    # git() strips surrounding whitespace, so the first line may lose the
    # leading space of its status (" M file" → "M file"). Split on the first
    # whitespace rather than at a fixed position.
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status),
        "git_dirty_files": sorted(
            line.split(maxsplit=1)[-1] for line in (status or "").splitlines() if line
        ),
        "scripts": {
            name: cached_sha256(
                str(pipeline_script(name)),
                pipeline_script(name).stat().st_size,
                pipeline_script(name).stat().st_mtime_ns,
            )
            for name in PIPELINE_SCRIPTS
            if pipeline_script(name).exists()
        },
    }


def gpu_identity() -> list[str]:
    try:
        result = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader",
            ),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def env_identity() -> dict[str, Any]:
    return {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "executable": sys.executable,
        "torch": package_version("torch"),
        "numpy": package_version("numpy"),
        "faiss": package_version("faiss") or package_version("faiss-gpu-cu12"),
        "transformers": package_version("transformers"),
        "sentence-transformers": package_version("sentence-transformers"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu_identity(),
    }


def index_identity(index_dir: Path) -> dict[str, Any]:
    """Corpus and encoder identity come from the index manifest, not recomputed.

    The SHA-256 of the Wiki-18 corpus means reading 14 GB; it was computed
    when the index was built and recorded in its manifest together with the
    encoder revision.
    """
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        return {"index_dir": str(index_dir), "manifest": None}
    manifest = read_json(manifest_path)
    encoder = manifest.get("encoder", {})
    return {
        "index_dir": str(index_dir),
        "corpus": manifest.get("corpus"),
        "encoder": {
            "model": encoder.get("model"),
            "resolved_revision": encoder.get("resolved_revision"),
            "dimension": encoder.get("dimension"),
            "max_length": encoder.get("max_length"),
        },
        "action_artifact": manifest.get("action_artifact"),
    }


def dataset_key(config: dict[str, Any]) -> str:
    """Dataset of a run; a missing field means HotpotQA.

    The default is safe precisely because it is verifiable: every registry run
    created before the field existed used ``hotpot_dev_fullwiki_v1.json`` and
    has 7,405 examples.
    """
    return str(config.get("dataset") or DEFAULT_DATASET)


def dataset_meta(dataset: str) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise KeyError(
            f"Unknown dataset {dataset!r}; known: {', '.join(DATASETS)}. "
            "A new dataset is added to DATASETS; the coverage file and alias "
            "table are optional, but leaving them out should be a decision, "
            "not an oversight."
        )
    return DATASETS[dataset]


def coverage_ceiling(dataset: str = DEFAULT_DATASET) -> float | None:
    """Corpus coverage ceiling: share of questions whose gold titles are in Wiki-18.

    Each dataset has its own ceiling (HotpotQA 81.67%, 2Wiki 51.05%), so the
    share of ceiling is comparable only within one dataset. ``None`` means the
    dataset has no gold titles to compute a ceiling from.
    """
    path = dataset_meta(dataset).get("coverage")
    if path is None:
        return None
    return float(read_json(path)["title_em_ceiling_normalized"])


def alias_table(dataset: str = DEFAULT_DATASET) -> dict[str, list[str]] | None:
    """Acceptable answers besides the primary one, by example id.

    ``None`` means the dataset has no aliases and alias metrics are not needed:
    they would equal the primary ones and only clutter the table with an
    extra column.
    """
    path = dataset_meta(dataset).get("aliases")
    if path is None:
        return None
    from export_eval_json import collect_aliases

    return collect_aliases(Path(path), None)


def score_judge_file(
    judge_path: Path,
    ceiling: float | None = None,
    dataset: str = DEFAULT_DATASET,
) -> dict[str, Any]:
    """Run metrics recomputed from the reader/judge JSON.

    A thin wrapper over ``report_phase0.score_run``: the same code computes
    both the results table row and the run's ``metrics.json``, so they cannot
    diverge. Per-sample vectors are dropped: they are only needed for paired
    tests and weigh as much as the run itself.
    """
    from report_phase0 import score_run

    if ceiling is None:
        ceiling = coverage_ceiling(dataset)
    scored = score_run(judge_path, ceiling, alias_table(dataset))
    return {
        key: value
        for key, value in scored.items()
        if not key.endswith("_per_sample")
    }


def new_manifest(run_id: str, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "running",
        "created_utc": utc_now(),
        "finished_utc": None,
        "hypothesis": config.get("hypothesis"),
        "config": config,
        "stages": [],
        "code": code_identity(),
        "inputs": {},
        "env": env_identity(),
        "metrics": None,
    }


def format_run_row(manifest: dict[str, Any], metrics: dict[str, Any] | None) -> str:
    config = manifest.get("config", {})
    percent = lambda value: "—" if value is None else f"{value * 100:.2f}"  # noqa: E731
    metrics = metrics or {}
    return (
        f"| `{manifest['run_id']}` "
        f"| {config.get('label', '—')} "
        f"| {config.get('reranker', '—')} "
        f"| {config.get('steps', '—')} "
        f"| {manifest.get('status', '—')} "
        f"| {percent(metrics.get('em'))} "
        f"| {percent(metrics.get('f1'))} "
        f"| {percent(metrics.get('judge'))} "
        f"| {percent(metrics.get('title_em'))} |"
    )


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
