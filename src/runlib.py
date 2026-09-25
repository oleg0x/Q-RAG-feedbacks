#!/usr/bin/env python3
"""Общие помощники реестра ранов: пути, манифесты, идентичность кода и входов.

Реестр устроен так: каждый ран — каталог ``runs/<YYYY-MM-DD>-<slug>/``. Мелкие
текстовые файлы (``manifest.json``, ``metrics.json``, ``cmd.sh``, ``notes.md``)
версионируются git и образуют историю экспериментов; тяжёлая нагрузка
(``retrieval.jsonl``, ``answer_judge.json``, ``log.txt``) лежит рядом, но вне
репозитория.

Модуль намеренно не тянет torch и не считает SHA-256 корпуса: идентичность
корпуса и энкодера уже зафиксирована в манифесте индекса, который собрал
``build_index_wiki_gte.py`` / ``build_index_wiki_qrag.py``, и берётся оттуда.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator

from build_index_wiki_gte import (
    atomic_write_json,
    cached_sha256,
    package_version,
    read_json,
    utc_now,
)


# Модули лаборатории лежат в src/, а реестр, конфиги и документы — в корне
# репозитория, поэтому REPO поднимается на уровень выше этого файла.
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

# Датасеты, на которых считается витрина. Ключ пишется в конфиг рана полем
# ``dataset``; от него зависят и потолок покрытия, и то, в какую таблицу
# RESULTS.md попадёт строка. Раны разных датасетов несравнимы между собой:
# у них разные вопросы, разное число примеров и разный потолок корпуса.
#
# ``coverage`` необязателен. Потолок покрытия считается по gold-титулам, а у
# семи eval-сплитов Search-R1 их нет ни у одного (см. build_searchr1_eval.py),
# поэтому у них ``coverage: None`` — и ``title_em``, и ``share_of_ceiling``
# в таких ранах остаются ``null``, а не превращаются в 100%.
#
# ``aliases`` — файл датасета, из которого ``export_eval_json.collect_aliases``
# достаёт допустимые ответы помимо основного. Он нужен там, где официальный
# эвал считает EM максимумом по алиасам: без него TriviaQA с её четырнадцатью
# ответами на вопрос занижена в разы.
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
    # Семь бенчмарков Search-R1 из одного test.parquet. Отдельные ключи, а не
    # переиспользование трёх верхних: вопросы там нормализованы (у 267 из них
    # дописан «?»), голд взят из golden_answers, а title-метрик нет вовсе —
    # смешивать такие строки со старыми в одной таблице нельзя.
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

# Скрипты, чьё содержимое влияет на результат рана. Хеши попадают в манифест,
# чтобы «каким кодом получен этот файл» отвечалось без git-археологии.
PIPELINE_SCRIPTS = (
    "fullwiki_qrag.py",
    "build_eval_variants.py",
    "candidate_pool.py",
    "build_index_wiki_gte.py",
    "build_index_wiki_qrag.py",
    "report_phase0.py",
    # Ридер и судья с 2026-08-08 — наш форк, а не read-only оригинал: от его
    # содержимого зависят EM, F1 и reward каждого рана.
    "answer_judge.py",
    "exp.py",
)

VENV_PYTHON = Path("/home/a.anokhin/venvs/gpu/bin/python")


def pipeline_script(name: str) -> Path:
    """Путь к скрипту пайплайна: `exp.py` в корне, остальные модули в `src/`.

    Имена в `PIPELINE_SCRIPTS` намеренно остались базовыми, без каталога: они
    же служат ключами в манифесте, и переезд модулей в `src/` не должен
    расщеплять реестр на «до» и «после» по имени ключа.
    """
    return REPO / name if name == "exp.py" else SRC / name


def run_dir(run_id: str) -> Path:
    if "/" in run_id or run_id.startswith("."):
        raise ValueError(f"Некорректный run_id: {run_id!r}")
    return RUNS / run_id


def iter_run_dirs() -> Iterator[Path]:
    """Каталоги ранов в лексикографическом порядке (он же хронологический).

    Наверху runs/ лежат только раны главной таблицы README, остальные — в
    runs/runs_history/. Реестр и витрина строятся по обоим уровням, поэтому
    переезд рана между ними не меняет ни одной строки таблиц.
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
    """Идентичность файла. ``hash_file=False`` для многогигабайтных входов."""
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
    """Коммит, признак незакоммиченных правок и хеши скриптов пайплайна."""
    status = git("status", "--porcelain", "--", "*.py")
    # git() снимает пробелы по краям, поэтому у первой строки может пропасть
    # ведущий пробел статуса (" M file" → "M file"). Режем по первому пробелу,
    # а не по фиксированной позиции.
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
    import sys

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
    """Корпус и энкодер берутся из манифеста индекса, а не пересчитываются.

    SHA-256 корпуса Wiki-18 — это 14 ГБ чтения; он уже посчитан при сборке
    индекса и записан в его манифест вместе с revision энкодера.
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
    """Датасет рана. Отсутствие поля означает HotpotQA — так было до 2026-08-05.

    Умолчание здесь безопасно ровно потому, что оно проверяемо: все раны
    реестра, заведённые до появления поля, сделаны на
    ``hotpot_dev_fullwiki_v1.json`` и содержат 7 405 примеров.
    """
    return str(config.get("dataset") or DEFAULT_DATASET)


def dataset_meta(dataset: str) -> dict[str, Any]:
    if dataset not in DATASETS:
        raise KeyError(
            f"Неизвестный датасет {dataset!r}; известны: {', '.join(DATASETS)}. "
            "Новый датасет добавляется в DATASETS; файл покрытия и таблица "
            "алиасов необязательны, но если их нет — это должно быть решением, "
            "а не забывчивостью."
        )
    return DATASETS[dataset]


def coverage_ceiling(dataset: str = DEFAULT_DATASET) -> float | None:
    """Потолок покрытия корпуса: доля вопросов, чьи gold-титулы есть в Wiki-18.

    Потолок свой у каждого датасета (HotpotQA 81.67%, 2Wiki 51.05%), поэтому
    доля потолка сравнима только внутри одного датасета. ``None`` означает,
    что у датасета нет gold-титулов и потолок считать не из чего.
    """
    path = dataset_meta(dataset).get("coverage")
    if path is None:
        return None
    return float(read_json(path)["title_em_ceiling_normalized"])


def alias_table(dataset: str = DEFAULT_DATASET) -> dict[str, list[str]] | None:
    """Допустимые ответы помимо основного, по id примера.

    ``None`` — у датасета алиасов нет и alias-метрики считать не нужно; они
    совпали бы с основными и только засоряли бы таблицу лишней колонкой.
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
    """Метрики рана, пересчитанные из reader/judge JSON.

    Тонкая обёртка над ``report_phase0.score_run``: один и тот же код считает и
    строку итоговой таблицы, и ``metrics.json`` рана, поэтому они не могут
    разойтись. Пер-сэмпловые векторы EM выбрасываются — они нужны только для
    парных тестов и весят как сам ран.
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
