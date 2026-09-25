#!/usr/bin/env python3
"""Разовая миграция ранов, сделанных до появления реестра (до 2026-07-28).

До миграции ``runs/`` был плоским каталогом на 1.9 ГБ: параметры прогона
кодировались в имени файла (``_steps4``, ``_mt1000``, ``_smoke20``), а
соответствие «файл ↔ команда, которой он получен» существовало только в
README. Скрипт раскладывает эти файлы по каталогам ранов, восстанавливает
для каждого манифест из зафиксированных в README параметров и пересчитывает
метрики из самих файлов.

Что важно понимать про восстановленные манифесты: параметры взяты из
документации, а не записаны в момент запуска, поэтому у них стоит
``"backfilled": true`` и ``"git_commit": null``. Метрики при этом настоящие —
они считаются тем же ``report_phase0.score_run``, что и для новых ранов.

    python src/migrate_runs.py --dry-run    показать план, ничего не трогая
    python src/migrate_runs.py              выполнить

Старые пути остаются рабочими: на каждый перенесённый файл ставится симлинк
с прежним именем, поэтому команды из docs/pipeline.md продолжают работать.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import runlib
from build_index_wiki_gte import cached_sha256


LOG = logging.getLogger("migrate-runs")

DATASETS = Path("/home/a.anokhin/Judge/datasets/data_sources")
HOTPOT_FULLWIKI = DATASETS / "hotpotqa/hotpot_dev_fullwiki_v1.json"
HOTPOT_DISTRACTOR = DATASETS / "hotpotqa/hotpot_dev_distractor_v1.json"
QRAG_INDEX = DATASETS / "full-wiki/wiki18-qrag-jul21-best-raw"
GTE_INDEX = DATASETS / "full-wiki/wiki18-gte"
QRAG_REPO = Path("/home/a.anokhin/Judge/Q-RAG-feedback")
ANSWER_JUDGE = QRAG_REPO / "answer_judge_llms.py"

# Окружение сборки, зафиксированное в docs/reproducibility.md. Восстановленные
# раны не могут сообщить его сами, поэтому оно проставляется из документации.
BACKFILL_ENV = {
    "python": "3.11.14",
    "executable": "/home/a.anokhin/venvs/gpu/bin/python",
    "torch": "2.10.0",
    "numpy": "2.2.6",
    "faiss": "1.14.3",
    "transformers": "4.57.6",
    "sentence-transformers": "5.2.3",
    "gpu": ["NVIDIA H200, 143771 MiB"],
    "source": "docs/reproducibility.md",
}

# Хеш версии fullwiki_qrag.py, которой получены три канонических Q-RAG-рана:
# в ней ещё не было --reranker и --log-candidates. Регрессия из
# docs/pipeline.md §7 подтверждает, что текущая версия с дефолтными флагами
# воспроизводит эти раны по pred_idx, retrieval_hops и title_*_exact.
QRAG_RUN_SCRIPT_SHA = "c9c5db13ce9b70b463daf5913ba3aa0ab2ee78996acc5a3b2d1851e71f2b5418"
CURRENT_SCRIPT_SHA = "9968053b4c2d5a3f6626f28caf721fcedda0499ec80a72afee904bf5a00506ac"
EVAL_VARIANTS_SHA = "5c71e40dddaa3e3552e1236140e0d845b439c6a68cf382c51e439cb49570d1a0"
ANSWER_JUDGE_SHA = "cb1d6ee785b370a85818a03d160864606f49953c83a76b7a03bc48bb94b03ffd"

JUDGE_CONFIG = {
    "script": str(ANSWER_JUDGE),
    "base_url": "http://127.0.0.1:8010/v1",
    "answer_model": "Qwen3-4B",
    "max_tokens": 1000,
    "judge_max_tokens": 100,
    "shards": 32,
}

QRAG_HYPOTHESIS = (
    "Обученный Q-RAG-реранкер поверх top-100 GTE поднимает answer-метрики "
    "относительно голого first-stage."
)
GTE_HYPOTHESIS = (
    "Голый first-stage GTE — baseline, относительно которого измеряется "
    "вклад Q-RAG."
)


def qrag_retrieve_config(steps: int) -> dict[str, Any]:
    return {
        "kind": "qrag",
        "index_dir": str(QRAG_INDEX),
        "candidate_index_dir": str(GTE_INDEX),
        "candidate_backend": "torch-cuda",
        "qrag_repo": str(QRAG_REPO),
        "device": "cuda:0",
        "cuda_visible_devices": "1",
        "model_dtype": "float32",
        "state_source": "critic",
        "input": str(HOTPOT_FULLWIKI),
        "batch_size": 64,
        "top_k": 100,
        "steps": steps,
        "mode": "fixed",
        "dedupe_titles": True,
        "reranker": "qrag",
        "log_candidates": "full",
        "trust_remote_code": True,
        "auto_prepare": False,
    }


def gte_retrieve_config() -> dict[str, Any]:
    config = qrag_retrieve_config(6)
    config.update({"reranker": "none", "log_candidates": "none"})
    return config


def variant_retrieve_config(context: str, **extra: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "kind": "variant",
        "context": context,
        "source_run": "2026-07-28-gte-only-steps6",
    }
    config.update(extra)
    return config


# Раскладка старого плоского runs/ по каталогам ранов. Порядок определяет
# порядок строк в runs/INDEX.md.
RUN_SPECS: list[dict[str, Any]] = [
    {
        "run_id": "2026-07-28-no-retrieval",
        "config_file": "no_retrieval.yaml",
        "order": 10,
        "label": "no retrieval",
        "hypothesis": "Сколько ридер отвечает вообще без контекста — нижняя граница.",
        "retrieval": "fullwiki_no_retrieval.jsonl",
        "judge": "fullwiki_no_retrieval_answer_judge_mt1000.json",
        "retrieve": variant_retrieve_config("none"),
    },
    {
        "run_id": "2026-07-28-gte-only-steps2",
        "config_file": "gte_only_steps2.yaml",
        "order": 20,
        "label": "GTE top-2",
        "hypothesis": GTE_HYPOTHESIS,
        "retrieval": "fullwiki_gte_only_steps2.jsonl",
        "judge": "fullwiki_gte_only_steps2_answer_judge_mt1000.json",
        "retrieve": variant_retrieve_config("truncate", steps=2),
    },
    {
        "run_id": "2026-07-28-gte-only-steps4",
        "config_file": "gte_only_steps4.yaml",
        "order": 30,
        "label": "GTE top-4",
        "hypothesis": GTE_HYPOTHESIS,
        "retrieval": "fullwiki_gte_only_steps4.jsonl",
        "judge": "fullwiki_gte_only_steps4_answer_judge_mt1000.json",
        "retrieve": variant_retrieve_config("truncate", steps=4),
    },
    {
        "run_id": "2026-07-28-gte-only-steps6",
        "config_file": "gte_only_steps6.yaml",
        "order": 40,
        "label": "GTE top-6",
        "hypothesis": GTE_HYPOTHESIS,
        "retrieval": "fullwiki_gte_only_steps6.jsonl",
        "judge": "fullwiki_gte_only_steps6_answer_judge_mt1000.json",
        "retrieve": gte_retrieve_config(),
    },
    {
        "run_id": "2026-07-28-qrag-fixed-steps2",
        "config_file": "qrag_steps2.yaml",
        "order": 50,
        "label": "Q-RAG, 2 шага",
        "hypothesis": QRAG_HYPOTHESIS,
        "retrieval": "fullwiki_qrag_fixed.jsonl",
        "judge": "fullwiki_qrag_fixed_answer_judge_mt1000.json",
        "retrieve": qrag_retrieve_config(2),
        "script_sha": QRAG_RUN_SCRIPT_SHA,
    },
    {
        "run_id": "2026-07-28-qrag-fixed-steps4",
        "config_file": "qrag_steps4.yaml",
        "order": 60,
        "label": "Q-RAG, 4 шага",
        "hypothesis": QRAG_HYPOTHESIS,
        "retrieval": "fullwiki_qrag_fixed_steps4.jsonl",
        "judge": "fullwiki_qrag_fixed_steps4_answer_judge_mt1000.json",
        "retrieve": qrag_retrieve_config(4),
        "script_sha": QRAG_RUN_SCRIPT_SHA,
    },
    {
        "run_id": "2026-07-28-qrag-fixed-steps6",
        "config_file": "qrag_steps6.yaml",
        "order": 70,
        "label": "Q-RAG, 6 шагов",
        "hypothesis": QRAG_HYPOTHESIS,
        "retrieval": "fullwiki_qrag_fixed_steps6.jsonl",
        "judge": "fullwiki_qrag_fixed_steps6_answer_judge_mt1000.json",
        "retrieve": qrag_retrieve_config(6),
        "script_sha": QRAG_RUN_SCRIPT_SHA,
        "extra_files": {"judge_diagnostics_steps6.json": "judge_diagnostics.json"},
    },
    {
        "run_id": "2026-07-28-oracle-sf",
        "config_file": "oracle_sf.yaml",
        "order": 80,
        "label": "oracle (gold sentences)",
        "hypothesis": "Потолок ридера при идеальном ретривале по HotpotQA.",
        "retrieval": "fullwiki_oracle_sf.jsonl",
        "judge": "fullwiki_oracle_sf_answer_judge_mt1000.json",
        "retrieve": variant_retrieve_config(
            "oracle", oracle_source=str(HOTPOT_DISTRACTOR)
        ),
    },
]

SHARED_FILES = {
    "hotpotqa_dev_title_coverage.json": "потолок покрытия корпуса Wiki-18",
    "phase0_report.json": "вывод report_phase0.py по восьми каноническим ранам",
}

ARCHIVE_FILES = {
    "fullwiki_qrag_fixed_answer_judge.json": (
        "Предварительный прогон Q-RAG steps=2 с лимитами reader 128 / judge 32. "
        "Не сопоставим с каноническими 1000/100 и не должен использоваться для "
        "сравнения моделей."
    ),
    "fullwiki_qrag_fixed_smoke20.jsonl": (
        "Smoke-прогон на 20 примерах при отладке retrieval, метрик не даёт."
    ),
    "fullwiki_qrag_fixed_dedupe_smoke20.jsonl": (
        "Smoke-прогон на 20 примерах для проверки title deduplication."
    ),
}


def mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def sha256_of(path: Path) -> str:
    stat = path.stat()
    return cached_sha256(str(path.resolve()), stat.st_size, stat.st_mtime_ns)


def retrieve_argv(spec: dict[str, Any], run: Path) -> list[str]:
    """Команда retrieve, воспроизводящая ран сегодня, с путями внутри runs/."""
    config = spec["retrieve"]
    output = str(run / runlib.RETRIEVAL_FILE)
    python = str(runlib.VENV_PYTHON)
    if config["kind"] == "qrag":
        argv = [
            python,
            "fullwiki_qrag.py",
            "retrieve",
            "--index-dir", config["index_dir"],
            "--candidate-index-dir", config["candidate_index_dir"],
            "--candidate-backend", config["candidate_backend"],
            "--qrag-repo", config["qrag_repo"],
            "--device", config["device"],
            "--model-dtype", config["model_dtype"],
            "--state-source", config["state_source"],
            "--input", config["input"],
            "--output", output,
            "--batch-size", str(config["batch_size"]),
            "--top-k", str(config["top_k"]),
            "--steps", str(config["steps"]),
            "--mode", config["mode"],
            "--dedupe-titles",
            "--reranker", config["reranker"],
            "--log-candidates", config["log_candidates"],
            "--trust-remote-code",
            "--no-auto-prepare",
        ]
        return argv
    source = runlib.run_dir(config["source_run"]) / runlib.RETRIEVAL_FILE
    argv = [
        python,
        "build_eval_variants.py",
        "--context", config["context"],
        "--input", str(source),
        "--output", output,
    ]
    if "steps" in config:
        argv += ["--steps", str(config["steps"])]
    if "oracle_source" in config:
        argv += ["--oracle-source", config["oracle_source"]]
    return argv


def judge_command(spec: dict[str, Any]) -> str:
    """Команда reader+judge, как её собирает exp.py (32-way sharding)."""
    return (
        f"python exp.py run --config configs/{spec['config_file']} --only judge\n"
        f"# внутри: split -n l/32 → 32 × answer_judge_llms.py → jq -s add\n"
        f"# подробности схемы: docs/pipeline.md §6"
    )


def build_manifest(spec: dict[str, Any], run: Path) -> dict[str, Any]:
    retrieval = run / runlib.RETRIEVAL_FILE
    judge = run / runlib.JUDGE_FILE
    config = {
        "label": spec["label"],
        "order": spec["order"],
        "hypothesis": spec["hypothesis"],
        "retrieve": spec["retrieve"],
        "judge": JUDGE_CONFIG,
        "steps": spec["retrieve"].get("steps"),
        "reranker": spec["retrieve"].get("reranker", spec["retrieve"].get("context")),
    }
    scripts = {
        "fullwiki_qrag.py": spec.get("script_sha", CURRENT_SCRIPT_SHA),
        "answer_judge_llms.py": ANSWER_JUDGE_SHA,
    }
    if spec["retrieve"]["kind"] == "variant":
        scripts["build_eval_variants.py"] = EVAL_VARIANTS_SHA
    return {
        "schema_version": runlib.MANIFEST_SCHEMA_VERSION,
        "run_id": spec["run_id"],
        "status": "ok",
        "backfilled": True,
        "backfill_note": (
            "Ран выполнен до появления exp.py. Параметры восстановлены из "
            "docs/pipeline.md §4, §4b и §6, окружение и хеши скриптов — из "
            "docs/reproducibility.md. Метрики пересчитаны из файлов рана."
        ),
        "created_utc": mtime_utc(retrieval),
        "finished_utc": mtime_utc(judge),
        "hypothesis": spec["hypothesis"],
        "config": config,
        "stages": [
            {
                "name": "retrieve",
                "argv": retrieve_argv(spec, run),
                "exit_code": 0,
                "output": runlib.RETRIEVAL_FILE,
                "size_bytes": retrieval.stat().st_size,
                "sha256": sha256_of(retrieval),
            },
            {
                "name": "judge",
                "argv": None,
                "command": judge_command(spec),
                "exit_code": 0,
                "output": runlib.JUDGE_FILE,
                "size_bytes": judge.stat().st_size,
                "sha256": sha256_of(judge),
            },
        ],
        "code": {
            "git_commit": None,
            "git_dirty": None,
            "scripts": scripts,
            "note": "Репозиторий заведён после этих ранов, коммита не существует.",
        },
        "inputs": {
            "dataset": str(HOTPOT_FULLWIKI),
            "samples": 7405,
            **runlib.index_identity(QRAG_INDEX),
            "candidate_index": runlib.index_identity(GTE_INDEX)["index_dir"],
            "checkpoint_sha256": (
                "865fc68fb9aa023224aac122257cc0671dcf0cedd343683cdcf40f8c00b55fa3"
            ),
        },
        "env": BACKFILL_ENV,
        "metrics": None,
    }


class Migration:
    def __init__(self, dry_run: bool, symlinks: bool):
        self.dry_run = dry_run
        self.symlinks = symlinks
        self.actions: list[str] = []

    def note(self, message: str) -> None:
        self.actions.append(message)
        LOG.info("%s%s", "[dry-run] " if self.dry_run else "", message)

    def mkdir(self, path: Path) -> None:
        self.note(f"mkdir {path.relative_to(runlib.REPO)}")
        if not self.dry_run:
            path.mkdir(parents=True, exist_ok=True)

    def move(self, source: Path, destination: Path) -> None:
        if destination.exists():
            raise FileExistsError(f"Цель уже существует: {destination}")
        size = source.stat().st_size / 1e6
        self.note(
            f"mv {source.name} → "
            f"{destination.relative_to(runlib.REPO)} ({size:.0f} МБ)"
        )
        if not self.dry_run:
            shutil.move(str(source), str(destination))

    def symlink(self, link: Path, target: Path) -> None:
        if not self.symlinks:
            return
        relative = os.path.relpath(target, link.parent)
        self.note(f"ln -s {relative} {link.relative_to(runlib.REPO)}")
        if not self.dry_run:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(relative)

    def write(self, path: Path, content: str) -> None:
        self.note(f"write {path.relative_to(runlib.REPO)}")
        if not self.dry_run:
            path.write_text(content, encoding="utf-8")


def write_run_metadata(migration: Migration, spec: dict[str, Any]) -> None:
    """Манифест, метрики и cmd.sh для уже разложенного рана."""
    run = runlib.run_dir(spec["run_id"])
    if migration.dry_run:
        migration.note(f"manifest + metrics + cmd.sh для {spec['run_id']}")
        return

    manifest = build_manifest(spec, run)
    LOG.info("Пересчитываю метрики: %s", spec["run_id"])
    manifest["metrics"] = runlib.score_judge_file(
        run / runlib.JUDGE_FILE,
        dataset=runlib.dataset_key(manifest.get("config", {})),
    )
    runlib.write_manifest(run, manifest)
    runlib.write_metrics(run, manifest["metrics"])

    cmd = [
        "#!/usr/bin/env bash",
        "# Восстановлено migrate_runs.py: этот ран сделан до появления exp.py,",
        "# команды реконструированы из docs/pipeline.md и приведены к путям",
        "# внутри каталога рана.",
        "set -euo pipefail",
        "cd /home/a.anokhin/Judge/full-wiki",
        "",
    ]
    environment = spec["retrieve"].get("cuda_visible_devices")
    prefix = (
        f"CUDA_VISIBLE_DEVICES={environment} HF_HUB_OFFLINE=1 "
        "TRANSFORMERS_OFFLINE=1 \\\n  "
        if environment
        else ""
    )
    cmd.append(prefix + " ".join(manifest["stages"][0]["argv"]))
    cmd.append("")
    cmd.append("# reader + judge:")
    cmd += [
        line if line.startswith("#") else f"# {line}"
        for line in judge_command(spec).splitlines()
    ]
    migration.write(run / runlib.CMD_FILE, "\n".join(cmd) + "\n")


def refresh_metadata(migration: Migration) -> None:
    """Перегенерировать метаданные уже перенесённых ранов.

    Нужно, когда меняется схема манифеста: файлы рана трогать не нужно, а
    восстановленные манифесты должны остаться в актуальном формате.
    """
    for spec in RUN_SPECS:
        run = runlib.run_dir(spec["run_id"])
        if not (run / runlib.JUDGE_FILE).exists():
            LOG.warning("Пропускаю %s: нет %s", run.name, runlib.JUDGE_FILE)
            continue
        write_run_metadata(migration, spec)


def migrate(migration: Migration) -> None:
    runs = runlib.RUNS
    missing = [
        name
        for spec in RUN_SPECS
        for name in (spec["retrieval"], spec["judge"])
        if not (runs / name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "В runs/ нет ожидаемых файлов, миграция уже выполнена или "
            f"каталог изменён: {', '.join(missing)}. "
            "Для перегенерации манифестов используйте --metadata-only."
        )

    migration.mkdir(runlib.SHARED)
    for name, purpose in SHARED_FILES.items():
        source = runs / name
        if source.exists():
            migration.move(source, runlib.SHARED / name)
            migration.symlink(runs / name, runlib.SHARED / name)
        else:
            LOG.warning("Общий артефакт отсутствует: %s (%s)", name, purpose)

    migration.mkdir(runlib.ARCHIVE)
    archived = []
    for name, reason in ARCHIVE_FILES.items():
        source = runs / name
        if source.exists():
            migration.move(source, runlib.ARCHIVE / name)
            archived.append((name, reason))
    if archived:
        lines = [
            "# Архив ранов",
            "",
            "Файлы здесь сохранены как след экспериментов, но не участвуют в",
            "сравнении моделей и не входят в git.",
            "",
        ]
        for name, reason in archived:
            lines += [f"## `{name}`", "", reason, ""]
        lines.append("")
        migration.write(runlib.ARCHIVE / "README.md", "\n".join(lines))

    for spec in RUN_SPECS:
        run = runlib.run_dir(spec["run_id"])
        migration.mkdir(run)
        migration.move(runs / spec["retrieval"], run / runlib.RETRIEVAL_FILE)
        migration.symlink(runs / spec["retrieval"], run / runlib.RETRIEVAL_FILE)
        migration.move(runs / spec["judge"], run / runlib.JUDGE_FILE)
        migration.symlink(runs / spec["judge"], run / runlib.JUDGE_FILE)
        for source_name, destination_name in spec.get("extra_files", {}).items():
            source = runs / source_name
            if source.exists():
                migration.move(source, run / destination_name)

        write_run_metadata(migration, spec)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="показать план перемещений и выйти",
    )
    parser.add_argument(
        "--no-symlinks",
        action="store_true",
        help="не оставлять симлинки на прежних именах файлов",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="ничего не перемещать, только перегенерировать манифесты и cmd.sh",
    )
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
    migration = Migration(dry_run=args.dry_run, symlinks=not args.no_symlinks)
    if args.metadata_only:
        refresh_metadata(migration)
    else:
        migrate(migration)
    LOG.info(
        "%s: %d действий",
        "План" if args.dry_run else "Выполнено",
        len(migration.actions),
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOG.error("Прервано")
        raise SystemExit(130)
    except Exception as error:
        LOG.error("Не удалось: %s", error)
        raise
