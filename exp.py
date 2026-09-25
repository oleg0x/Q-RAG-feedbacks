#!/usr/bin/env python3
"""Драйвер экспериментов: конфиг → каталог рана → retrieve → judge → score.

Зачем он нужен. Раньше эксперимент запускался копипастой из README: retrieve
с путями, вбитыми в ``case``, затем ``split -n l/32``, 32 процесса
``answer_judge_llms.py``, ``jq -s add`` и ``report_phase0.py``. Пути выхода
приходилось придумывать вручную, повторный запуск молча затирал результат, а
знание «каким кодом и с какими флагами это получено» жило только в голове.

Теперь один вызов::

    python exp.py run --config configs/qrag_steps6.yaml

создаёт ``runs/<YYYY-MM-DD>-<slug>/``, пишет туда манифест с параметрами,
хешами скриптов, коммитом и версиями окружения, прогоняет все стадии, ведёт
``log.txt`` и кладёт рядом ``metrics.json``. Каталог рана отказывается
перезаписываться без ``--resume``.

Остальные подкоманды::

    python exp.py show <run_id>      компактная сводка рана
    python exp.py list               все раны реестра
    python exp.py index              пересобрать runs/INDEX.md и RESULTS.md
    python exp.py archive <run_id> --reason "..."

Флаг ``--dry-run`` печатает точные argv всех стадий и ничего не выполняет:
это основной режим, когда эксперимент готовится, но запускать его должен
человек — GPU и vLLM стоят дорого.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Sequence

# Драйвер намеренно остался в корне: `python exp.py run …` — команда, которой
# записаны все cmd.sh реестра и весь runbook, и переезд модулей в src/ не
# должен её менять. Цена — эта вставка: без неё `import runlib` не находится.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

import runlib
from build_index_wiki_gte import utc_now


LOG = logging.getLogger("exp")

STAGES = ("retrieve", "judge", "score")
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

# Стадии запускаются подпроцессами с cwd=runlib.REPO, поэтому пути к скриптам
# относительные — в этом виде они и попадают в cmd.sh рана.
RETRIEVE_SCRIPT = "src/fullwiki_qrag.py"
POOL_SCRIPT = "src/candidate_pool.py"
VARIANTS_SCRIPT = "src/build_eval_variants.py"


class StageFailed(RuntimeError):
    pass


# --------------------------------------------------------------------------
# конфиг


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Конфиг должен быть YAML-объектом: {path}")
    for key in ("label", "hypothesis", "retrieve"):
        if key not in config:
            raise ValueError(f"В конфиге {path} нет обязательного поля {key!r}")
    # Датасет проверяется до всего остального: опечатка в нём означает не
    # только чужой потолок покрытия в metrics.json, но и чужую таблицу в
    # RESULTS.md, а замечено это будет уже после часа GPU.
    if "dataset" in config and config["dataset"] not in runlib.DATASETS:
        raise ValueError(
            f"Неизвестный dataset {config['dataset']!r} в {path}; известны: "
            f"{', '.join(runlib.DATASETS)}"
        )
    retrieve = config["retrieve"]
    kind = retrieve.get("kind")
    if kind not in ("qrag", "variant", "pool", "search"):
        raise ValueError(
            f"retrieve.kind должен быть qrag, variant, pool или search, получено {kind!r}"
        )
    if kind == "search":
        # steps у CLI имеет дефолт, но бюджет eval-рана обязан быть назван
        # в конфиге явно: он определяет сравнимость строки в таблице.
        for key in ("index_dir", "qrag_repo", "input", "steps"):
            if not retrieve.get(key):
                raise ValueError(f"retrieve.{key} обязателен для kind=search")
    if kind == "variant":
        if retrieve.get("context") not in ("truncate", "none", "oracle"):
            raise ValueError("retrieve.context: truncate, none или oracle")
        if not retrieve.get("source_run"):
            raise ValueError("retrieve.source_run обязателен для kind=variant")
    if kind == "pool":
        if not retrieve.get("source_run"):
            raise ValueError("retrieve.source_run обязателен для kind=pool")
        if not retrieve.get("index_dir"):
            raise ValueError("retrieve.index_dir обязателен для kind=pool")
        if not retrieve.get("steps"):
            raise ValueError("retrieve.steps обязателен для kind=pool")
        if retrieve.get("prefer") == "gold-sentence" and not retrieve.get("gold_source"):
            raise ValueError(
                "retrieve.gold_source обязателен при prefer=gold-sentence: "
                "gold-предложения есть только в hotpot_dev_distractor_v1.json"
            )
    config.setdefault("judge", {})
    config["config_file"] = path.name
    return config


def resolve_run_id(config: dict[str, Any], args: argparse.Namespace) -> str:
    if args.run_id:
        return args.run_id
    slug = config.get("slug") or Path(config["config_file"]).stem.replace("_", "-")
    if args.tag:
        slug = f"{slug}-{args.tag}"
    return f"{date.today().isoformat()}-{slug}"


# --------------------------------------------------------------------------
# запуск стадий


def child_environment(extra: dict[str, str] | None = None, drop: Sequence[str] = ()) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(OFFLINE_ENV)
    environment.update(extra or {})
    for name in drop:
        environment.pop(name, None)
    return environment


def run_process(
    argv: Sequence[str],
    log_path: Path,
    environment: dict[str, str] | None = None,
) -> int:
    """Запустить процесс, направив stdout и stderr в общий лог рана."""
    LOG.info("$ %s", " ".join(argv))
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(argv)}\n")
        log.flush()
        completed = subprocess.run(
            list(argv),
            cwd=runlib.REPO,
            env=environment or child_environment(),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return completed.returncode


def retrieve_argv(config: dict[str, Any], run: Path) -> list[str]:
    retrieve = config["retrieve"]
    output = str(run / runlib.RETRIEVAL_FILE)
    python = str(runlib.VENV_PYTHON)
    if retrieve["kind"] == "qrag":
        argv = [
            python, RETRIEVE_SCRIPT, "retrieve",
            "--index-dir", str(retrieve["index_dir"]),
            "--qrag-repo", str(retrieve["qrag_repo"]),
            "--device", str(retrieve.get("device", "cuda:0")),
            "--model-dtype", str(retrieve.get("model_dtype", "float32")),
            "--state-source", str(retrieve.get("state_source", "critic")),
            "--input", str(retrieve["input"]),
            "--output", output,
            "--batch-size", str(retrieve.get("batch_size", 64)),
            "--top-k", str(retrieve.get("top_k", 100)),
            "--steps", str(retrieve["steps"]),
            "--mode", str(retrieve.get("mode", "fixed")),
            "--reranker", str(retrieve.get("reranker", "qrag")),
            "--log-candidates", str(retrieve.get("log_candidates", "full")),
        ]
        if retrieve.get("candidate_index_dir"):
            argv += ["--candidate-index-dir", str(retrieve["candidate_index_dir"])]
            argv += [
                "--candidate-backend",
                str(retrieve.get("candidate_backend", "torch-cuda")),
            ]
        argv.append(
            "--dedupe-titles" if retrieve.get("dedupe_titles", True) else "--no-dedupe-titles"
        )
        # Флаг добавляется только когда квота задана явно: иначе argv всех
        # прежних конфигов изменился бы, а он попадает в манифест и cmd.sh.
        if retrieve.get("max_chunks_per_title") is not None:
            argv += [
                "--max-chunks-per-title",
                str(retrieve["max_chunks_per_title"]),
            ]
        if retrieve.get("trust_remote_code", True):
            argv.append("--trust-remote-code")
        if not retrieve.get("auto_prepare", False):
            argv.append("--no-auto-prepare")
        if retrieve.get("first_stage_query_length") is not None:
            argv += [
                "--first-stage-query-length",
                str(retrieve["first_stage_query_length"]),
            ]
        if retrieve.get("max_samples"):
            argv += ["--max-samples", str(retrieve["max_samples"])]
        return argv

    if retrieve["kind"] == "search":
        argv = [
            python, RETRIEVE_SCRIPT, "search",
            "--index-dir", str(retrieve["index_dir"]),
            "--qrag-repo", str(retrieve["qrag_repo"]),
            "--device", str(retrieve.get("device", "cuda:0")),
            "--model-dtype", str(retrieve.get("model_dtype", "float32")),
            "--input", str(retrieve["input"]),
            "--output", output,
            "--batch-size", str(retrieve.get("batch_size", 64)),
            "--top-k", str(retrieve.get("top_k", 100)),
            "--steps", str(retrieve["steps"]),
            "--max-chunks-per-title", str(retrieve.get("max_chunks_per_title", 2)),
            "--log-candidates", str(retrieve.get("log_candidates", "none")),
        ]
        # Без чекпоинта башня — стоковая GTE: это zero-shot ступени 3
        # лестницы, стартовая точка, против которой читаются обученные цифры.
        if retrieve.get("checkpoint"):
            argv += ["--checkpoint", str(retrieve["checkpoint"])]
            argv += ["--state-source", str(retrieve.get("state_source", "critic"))]
        if retrieve.get("title_table"):
            argv += ["--title-table", str(retrieve["title_table"])]
        if not retrieve.get("auto_prepare", False):
            argv.append("--no-auto-prepare")
        if retrieve.get("max_samples"):
            argv += ["--max-samples", str(retrieve["max_samples"])]
        return argv

    if retrieve["kind"] == "pool":
        source = runlib.run_dir(str(retrieve["source_run"])) / runlib.RETRIEVAL_FILE
        argv = [
            python, POOL_SCRIPT, "select",
            "--input", str(source),
            "--index-dir", str(retrieve["index_dir"]),
            "--steps", str(retrieve["steps"]),
            "--output", output,
        ]
        # Оба флага добавляются только когда заданы: argv прежних конфигов
        # обязан остаться прежним, он попадает в манифест и cmd.sh.
        if retrieve.get("prefer"):
            argv += ["--prefer", str(retrieve["prefer"])]
            argv += ["--gold-source", str(retrieve["gold_source"])]
        if retrieve.get("max_per_title") is not None:
            argv += ["--max-per-title", str(retrieve["max_per_title"])]
        return argv

    source = runlib.run_dir(str(retrieve["source_run"])) / runlib.RETRIEVAL_FILE
    argv = [
        python, VARIANTS_SCRIPT,
        "--context", str(retrieve["context"]),
        "--input", str(source),
        "--output", output,
    ]
    if retrieve["context"] == "truncate":
        argv += ["--steps", str(retrieve["steps"])]
    if retrieve["context"] == "oracle":
        argv += ["--oracle-source", str(retrieve["oracle_source"])]
    return argv


def jsonl_lines(path: Path) -> list[str]:
    """Строки JSONL, разрезанные только по ``\\n``.

    Не ``splitlines()``. Тот считает переводом строки ещё восемь символов, и
    три из них — U+0085, U+2028, U+2029 — больше 0x1F, поэтому ``json.dumps``
    оставляет их в файле как есть (остальные пять он экранирует). Запись, внутри
    которой такой символ оказался, разваливается на две, судья получает обломок
    и падает на ``JSONDecodeError``, а число записей в логе тихо расходится с
    входом — то есть неверный результат отличим от верного только по этому
    расхождению, если на него посмотреть.

    Так и случилось: у трёх вопросов TriviaQA из eval-таблицы Search-R1 внутри
    текста стоит U+0085 (NEXT LINE), 11 313 записей превратились в 11 316, и
    три шарда из тридцати двух упали.
    """
    text = path.read_text(encoding="utf-8")
    return [line for line in text.split("\n") if line.strip()]


def check_jsonl_shape(lines: Sequence[str], path: Path) -> None:
    """Убедиться, что каждая строка — целый JSON-объект, до запуска судьи.

    Проверка формы, а не разбор: полный ``json.loads`` на сотнях мегабайт стоит
    десятки секунд на ран, а обломок развалившейся записи ловится тем, что не
    начинается с ``{`` или не заканчивается ``}``. Без неё поломка входа
    проявляется как «три шарда из тридцати двух упали» через четыре минуты
    работы тридцати двух процессов.
    """
    broken = [
        number
        for number, line in enumerate(lines, 1)
        if not (line.startswith("{") and line.rstrip().endswith("}"))
    ]
    if broken:
        raise StageFailed(
            f"{path}: строки {broken[:5]}"
            f"{' и ещё ' + str(len(broken) - 5) if len(broken) > 5 else ''} "
            "не являются целыми JSON-объектами. Обычная причина — символ, "
            "который json.dumps не экранирует, а разрезающий код считает "
            "переводом строки (см. jsonl_lines)."
        )


def stage_retrieve(config: dict[str, Any], run: Path) -> dict[str, Any]:
    retrieve = config["retrieve"]
    argv = retrieve_argv(config, run)
    extra = {}
    if retrieve.get("cuda_visible_devices") is not None:
        extra["CUDA_VISIBLE_DEVICES"] = str(retrieve["cuda_visible_devices"])
    started = time.monotonic()
    code = run_process(argv, run / runlib.LOG_FILE, child_environment(extra))
    output = run / runlib.RETRIEVAL_FILE
    if code != 0 or not output.exists():
        raise StageFailed(f"retrieve завершился с кодом {code}, см. {run / runlib.LOG_FILE}")

    # У build_eval_variants.py и candidate_pool.py нет своего --max-samples:
    # производный набор обязан повторять исходный ран целиком. Для
    # smoke-прогонов урезаем его здесь, уже после того как вариант собран.
    limit = retrieve.get("max_samples")
    if limit and retrieve["kind"] in ("variant", "pool"):
        lines = jsonl_lines(output)[:limit]
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        LOG.info("Smoke-режим: оставлено %d записей", len(lines))
    return {
        "name": "retrieve",
        "argv": argv,
        "env": extra | OFFLINE_ENV,
        "exit_code": code,
        "seconds": round(time.monotonic() - started, 1),
        "output": runlib.RETRIEVAL_FILE,
        "size_bytes": output.stat().st_size,
        "sha256": runlib.file_identity(output)["sha256"],
        "records": sum(1 for _ in output.open("r", encoding="utf-8")),
    }


def judge_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Настройки стадии judge с умолчаниями.

    Скрипт — наш форк, а не read-only оригинал: контракт награды v2 считает
    ``em_alias`` максимумом по вариантам ответа, и без него у NQ занижен EM
    (11.97% против 9.25% на одних и тех же предсказаниях). ``contract: v1``
    в конфиге возвращает прежнее поведение бит в бит — им сняты старые раны.
    """
    judge = dict(config.get("judge") or {})
    judge.setdefault("script", str(runlib.pipeline_script("answer_judge.py")))
    judge.setdefault("base_url", "http://127.0.0.1:8010/v1")
    judge.setdefault("answer_model", "Qwen3-4B")
    judge.setdefault("max_tokens", 1000)
    judge.setdefault("judge_max_tokens", 100)
    judge.setdefault("shards", 32)
    judge.setdefault("contract", "v2")
    judge.setdefault("dataset", runlib.dataset_key(config))
    return judge


def shard_argv(judge: dict[str, Any], shard_input: Path, shard_output: Path) -> list[str]:
    argv = [
        str(runlib.VENV_PYTHON),
        str(judge["script"]),
        "--retriever-logfile", str(shard_input),
        "--output-file", str(shard_output),
        "--base-url", str(judge["base_url"]),
        "--answer-model", str(judge["answer_model"]),
        "--max-samples", "100000",
        "--max-tokens", str(judge["max_tokens"]),
        "--judge-max-tokens", str(judge["judge_max_tokens"]),
    ]
    # Оригинал этих флагов не знает: конфиг с contract: v1 и явным script
    # оставляет прежнюю командную строку, чтобы старые раны воспроизводились.
    if Path(judge["script"]).name == "answer_judge.py":
        argv += ["--contract", str(judge["contract"])]
        # Варианты ответа берутся из таблицы датасета, а не из retrieval.jsonl:
        # стадия ретривала алиасы роняет.
        if judge["contract"] == "v2" and judge.get("dataset"):
            argv += ["--dataset", str(judge["dataset"])]
        if judge.get("judge_all"):
            argv += ["--judge-all"]
    return argv


def check_vllm(base_url: str) -> None:
    """Ранняя проверка, что vLLM поднят: иначе 32 процесса упадут поодиночке."""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            response.read(1)
    except (urllib.error.URLError, OSError) as error:
        raise StageFailed(
            f"vLLM не отвечает на {url}: {error}. "
            "Разверните сервер по docs/pipeline.md §5"
        ) from error


def stage_judge(config: dict[str, Any], run: Path) -> dict[str, Any]:
    judge = judge_settings(config)
    retrieval = run / runlib.RETRIEVAL_FILE
    if not retrieval.exists():
        raise StageFailed(f"Нет {retrieval}: сначала стадия retrieve")
    check_vllm(str(judge["base_url"]))

    records = jsonl_lines(retrieval)
    check_jsonl_shape(records, retrieval)
    shards = max(1, min(int(judge["shards"]), len(records)))
    parts_dir = run / "judge-shards"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)

    # Разбиение непрерывными кусками: порядок id внутри и между шардами
    # сохраняется, поэтому склейка возвращает исходную последовательность.
    per_shard = -(-len(records) // shards)
    shard_files = []
    for number in range(shards):
        chunk = records[number * per_shard : (number + 1) * per_shard]
        if not chunk:
            break
        shard_input = parts_dir / f"input-{number:02d}.jsonl"
        shard_input.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        shard_files.append((shard_input, parts_dir / f"input-{number:02d}.json"))

    LOG.info("judge: %d записей, %d шардов", len(records), len(shard_files))
    started = time.monotonic()
    environment = child_environment(drop=("VLLM_API_KEY",))
    log_path = run / runlib.LOG_FILE
    processes = []
    with log_path.open("a", encoding="utf-8") as log:
        for shard_input, shard_output in shard_files:
            argv = shard_argv(judge, shard_input, shard_output)
            log.write(f"\n$ {' '.join(argv)}\n")
            log.flush()
            processes.append(
                (
                    subprocess.Popen(
                        argv,
                        cwd=runlib.REPO,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    ),
                    shard_output,
                )
            )
        failures = [
            str(output) for process, output in processes if process.wait() != 0
        ]
    if failures:
        raise StageFailed(
            f"{len(failures)} из {len(processes)} шардов judge упали, "
            f"см. {log_path}"
        )

    merged: list[dict[str, Any]] = []
    for _, shard_output in processes:
        with shard_output.open("r", encoding="utf-8") as stream:
            merged.extend(json.load(stream))
    destination = run / runlib.JUDGE_FILE
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(merged, stream, ensure_ascii=False)

    # Порядок и состав должны совпадать с retrieval JSONL — иначе метрики
    # считаются не по тем вопросам.
    expected = [json.loads(line)["id"] for line in records]
    actual = [record["id"] for record in merged]
    if expected != actual:
        raise StageFailed(
            "Порядок id после склейки шардов не совпадает с retrieval.jsonl"
        )
    shutil.rmtree(parts_dir)
    return {
        "name": "judge",
        "argv": shard_argv(judge, Path("<shard>.jsonl"), Path("<shard>.json")),
        "shards": len(processes),
        "exit_code": 0,
        "seconds": round(time.monotonic() - started, 1),
        "output": runlib.JUDGE_FILE,
        "size_bytes": destination.stat().st_size,
        "sha256": runlib.file_identity(destination)["sha256"],
        "records": len(merged),
    }


def stage_score(config: dict[str, Any], run: Path) -> dict[str, Any]:
    judge_file = run / runlib.JUDGE_FILE
    if not judge_file.exists():
        raise StageFailed(f"Нет {judge_file}: сначала стадия judge")
    started = time.monotonic()
    dataset = runlib.dataset_key(config)
    metrics = runlib.score_judge_file(judge_file, dataset=dataset)
    runlib.write_metrics(run, metrics)
    # Форматируется через помощник, а не через %.4f: у датасета без
    # gold-титулов title_em равен None, и «%.4f» на нём падает уже после того,
    # как метрики записаны, — то есть ран выглядел бы упавшим, будучи целым.
    optional = lambda value: "—" if value is None else f"{value:.4f}"  # noqa: E731
    LOG.info(
        "%s: EM=%.4f F1=%.4f judge=%.4f title_em=%s%s",
        dataset,
        metrics["em"], metrics["f1"], metrics["judge"],
        optional(metrics["title_em"]),
        f" em_alias={metrics['em_alias']:.4f}" if "em_alias" in metrics else "",
    )
    return {
        "name": "score",
        "argv": None,
        "command": "runlib.score_judge_file → metrics.json",
        "dataset": dataset,
        "exit_code": 0,
        "seconds": round(time.monotonic() - started, 1),
        "output": runlib.METRICS_FILE,
        "metrics": metrics,
    }


# --------------------------------------------------------------------------
# подкоманды


def write_cmd_script(config: dict[str, Any], run: Path) -> None:
    retrieve = config["retrieve"]
    judge = judge_settings(config)
    prefix = ""
    if retrieve.get("cuda_visible_devices") is not None:
        prefix = f"CUDA_VISIBLE_DEVICES={retrieve['cuda_visible_devices']} "
    lines = [
        "#!/usr/bin/env bash",
        f"# Ран {run.name}, собран exp.py из configs/{config['config_file']}.",
        "# Файл пишется до запуска, поэтому ран воспроизводим даже если",
        "# процесс был убит.",
        "set -euo pipefail",
        f"cd {runlib.REPO}",
        "",
        prefix + "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\",
        "  " + " ".join(retrieve_argv(config, run)),
        "",
        f"# judge: {judge['shards']} шардов answer_judge_llms.py, затем склейка",
        "# " + " ".join(shard_argv(judge, Path("<shard>.jsonl"), Path("<shard>.json"))),
        "",
        f"python exp.py run --config configs/{config['config_file']} --only score",
    ]
    (run / runlib.CMD_FILE).write_text("\n".join(lines) + "\n", encoding="utf-8")


def command_run(args: argparse.Namespace) -> int:
    config = load_config(args.config.expanduser().resolve())
    if args.max_samples is not None:
        config["retrieve"]["max_samples"] = args.max_samples
    run_id = resolve_run_id(config, args)
    run = runlib.run_dir(run_id)
    stages = args.only or list(STAGES)

    if args.dry_run:
        print(f"run_id: {run_id}")
        print(f"каталог: {run.relative_to(runlib.REPO)}")
        print(f"гипотеза: {config['hypothesis']}")
        for stage in stages:
            print(f"\n— стадия {stage}")
            if stage == "retrieve":
                environment = config["retrieve"].get("cuda_visible_devices")
                if environment is not None:
                    print(f"  CUDA_VISIBLE_DEVICES={environment} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
                print("  " + " ".join(retrieve_argv(config, run)))
            elif stage == "judge":
                judge = judge_settings(config)
                print(f"  {judge['shards']} × answer_judge_llms.py (VLLM_API_KEY снимается)")
                print("  " + " ".join(shard_argv(judge, Path("<shard>.jsonl"), Path("<shard>.json"))))
            else:
                print("  runlib.score_judge_file → metrics.json")
        return 0

    if run.exists() and not args.resume:
        raise SystemExit(
            f"Каталог {run} уже существует. Задайте --tag, --run-id или "
            "--resume, чтобы не потерять прежний результат."
        )
    run.mkdir(parents=True, exist_ok=True)

    if (run / runlib.MANIFEST_FILE).exists() and args.resume:
        manifest = runlib.read_manifest(run)
        manifest["status"] = "running"
        manifest["resumed_utc"] = utc_now()
    else:
        manifest = runlib.new_manifest(run_id, config)
        manifest["inputs"] = collect_inputs(config)
    runlib.write_manifest(run, manifest)
    write_cmd_script(config, run)

    if manifest["code"]["git_dirty"]:
        LOG.warning(
            "В репозитории есть незакоммиченные правки: %s. "
            "Ран будет помечен git_dirty.",
            ", ".join(manifest["code"]["git_dirty_files"][:5]),
        )

    done = {stage["name"] for stage in manifest["stages"] if stage.get("exit_code") == 0}
    for stage in stages:
        if args.resume and stage in done:
            LOG.info("Пропускаю стадию %s: уже выполнена", stage)
            continue
        LOG.info("Стадия %s", stage)
        try:
            if stage == "retrieve":
                result = stage_retrieve(config, run)
            elif stage == "judge":
                result = stage_judge(config, run)
            else:
                result = stage_score(config, run)
                manifest["metrics"] = result.pop("metrics")
        except StageFailed as error:
            manifest["status"] = "failed"
            manifest["finished_utc"] = utc_now()
            manifest["stages"].append(
                {"name": stage, "exit_code": 1, "error": str(error)}
            )
            runlib.write_manifest(run, manifest)
            LOG.error("Стадия %s не удалась: %s", stage, error)
            return 1
        manifest["stages"] = [
            existing for existing in manifest["stages"] if existing["name"] != stage
        ] + [result]
        runlib.write_manifest(run, manifest)

    manifest["status"] = "ok"
    manifest["finished_utc"] = utc_now()
    runlib.write_manifest(run, manifest)
    LOG.info("Готово: %s", run.relative_to(runlib.REPO))
    LOG.info("Не забудьте про запись в EXPERIMENTS.md и `make report`")
    return 0


def collect_inputs(config: dict[str, Any]) -> dict[str, Any]:
    retrieve = config["retrieve"]
    inputs: dict[str, Any] = {}
    if retrieve["kind"] in ("qrag", "search"):
        inputs.update(runlib.index_identity(Path(retrieve["index_dir"])))
        if retrieve.get("candidate_index_dir"):
            inputs["candidate_index"] = str(retrieve["candidate_index_dir"])
        dataset = Path(retrieve["input"])
        if dataset.exists():
            inputs["dataset"] = runlib.file_identity(dataset)
        # Для обученных чекпоинтов линия строки таблицы определяется весами:
        # без их идентичности ран невоспроизводим.
        if retrieve.get("checkpoint"):
            checkpoint = Path(retrieve["checkpoint"])
            if checkpoint.exists():
                inputs["checkpoint"] = runlib.file_identity(checkpoint)
    else:
        inputs["source_run"] = retrieve["source_run"]
        source_manifest = runlib.run_dir(str(retrieve["source_run"])) / runlib.MANIFEST_FILE
        if source_manifest.exists():
            inputs["source_inputs"] = runlib.read_json(source_manifest).get("inputs")
        if retrieve["kind"] == "pool":
            # Корпус и его смещения нужны, чтобы резолвить row ID пула в
            # тексты; идентичность берётся из манифеста индекса, как у qrag.
            inputs.update(runlib.index_identity(Path(retrieve["index_dir"])))
        for key in ("oracle_source", "gold_source"):
            if retrieve.get(key):
                source = Path(retrieve[key])
                if source.exists():
                    inputs[key] = runlib.file_identity(source)
    return inputs


def command_show(args: argparse.Namespace) -> int:
    run = runlib.run_dir(args.run_id)
    manifest = runlib.read_manifest(run)
    metrics = runlib.read_metrics(run) or {}
    config = manifest.get("config", {})
    print(f"{manifest['run_id']}  [{manifest['status']}]")
    print(f"метка:     {config.get('label')}")
    print(f"гипотеза:  {manifest.get('hypothesis')}")
    print(f"начат:     {manifest.get('created_utc')}")
    print(f"закончен:  {manifest.get('finished_utc')}")
    print(f"коммит:    {manifest['code'].get('git_commit')}"
          f"{' (dirty)' if manifest['code'].get('git_dirty') else ''}")
    if manifest.get("backfilled"):
        print("восстановлен из документации: параметры не записаны в момент запуска")
    print("\nстадии:")
    for stage in manifest.get("stages", []):
        seconds = stage.get("seconds")
        print(
            f"  {stage['name']:9s} код={stage.get('exit_code')} "
            f"{'' if seconds is None else f'{seconds:.0f} c '}"
            f"{stage.get('output', '')}"
        )
    if metrics:
        print("\nметрики:")
        for key in ("em", "f1", "judge", "title_recall", "title_em", "samples"):
            value = metrics.get(key)
            if isinstance(value, float):
                print(f"  {key:14s} {value * 100:.2f}%")
            else:
                print(f"  {key:14s} {value}")
    retrieve = config.get("retrieve", {})
    print(
        f"\nретривал:  {retrieve.get('kind')}, "
        f"steps={retrieve.get('steps', '—')}, "
        f"reranker={retrieve.get('reranker') or retrieve.get('context') or '—'}"
    )
    return 0


def command_list(args: argparse.Namespace) -> int:
    for run in runlib.iter_run_dirs():
        manifest = runlib.read_manifest(run)
        metrics = runlib.read_metrics(run) or {}
        em = metrics.get("em")
        print(
            f"{run.name:36s} {manifest.get('status', '?'):7s} "
            f"EM={'—' if em is None else f'{em * 100:5.2f}%'}  "
            f"{manifest.get('config', {}).get('label', '')}"
        )
    return 0


def command_index(args: argparse.Namespace) -> int:
    import runs_index

    return runs_index.main([])


def command_archive(args: argparse.Namespace) -> int:
    run = runlib.run_dir(args.run_id)
    if not run.exists():
        raise SystemExit(f"Нет такого рана: {run}")
    runlib.ARCHIVE.mkdir(parents=True, exist_ok=True)
    destination = runlib.ARCHIVE / run.name
    if destination.exists():
        raise SystemExit(f"В архиве уже есть {destination}")
    manifest_path = run / runlib.MANIFEST_FILE
    if manifest_path.exists():
        manifest = runlib.read_manifest(run)
        manifest["archived_utc"] = utc_now()
        manifest["archive_reason"] = args.reason
        runlib.write_manifest(run, manifest)
    shutil.move(str(run), str(destination))
    # runs/INDEX.md обещает, что причина архивации каждого рана записана в
    # README архива — дописываем её туда, а не только в манифест.
    readme = runlib.ARCHIVE / "README.md"
    entry = f"## `{run.name}`\n\n{args.reason}\n\n"
    with readme.open("a", encoding="utf-8") as stream:
        stream.write(entry)
    LOG.info("Ран перенесён в %s: %s", destination.relative_to(runlib.REPO), args.reason)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="выполнить эксперимент")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--run-id", default=None, help="переопределить run_id")
    run_parser.add_argument("--tag", default=None, help="суффикс к слагу run_id")
    run_parser.add_argument(
        "--only", action="append", choices=STAGES, help="выполнить только эти стадии"
    )
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument(
        "--resume", action="store_true", help="продолжить существующий ран"
    )
    run_parser.add_argument(
        "--max-samples", type=int, default=None, help="ограничить выборку (smoke)"
    )
    run_parser.set_defaults(handler=command_run)

    show_parser = subparsers.add_parser("show", help="сводка рана")
    show_parser.add_argument("run_id")
    show_parser.set_defaults(handler=command_show)

    list_parser = subparsers.add_parser("list", help="все раны")
    list_parser.set_defaults(handler=command_list)

    index_parser = subparsers.add_parser("index", help="пересобрать INDEX.md и RESULTS.md")
    index_parser.set_defaults(handler=command_index)

    archive_parser = subparsers.add_parser("archive", help="убрать ран в архив")
    archive_parser.add_argument("run_id")
    archive_parser.add_argument("--reason", required=True)
    archive_parser.set_defaults(handler=command_archive)

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
