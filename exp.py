#!/usr/bin/env python3
"""Experiment driver: config → run directory → retrieve → judge → score.

It replaces a manual sequence (retrieve, ``split -n l/32``, 32
``answer_judge_llms.py`` processes, ``jq -s add``, ``report_phase0.py``) with
one command that never overwrites a result silently and records which code
and flags produced it. A single call::

    python exp.py run --config configs/qrag_steps6.yaml

creates ``runs/<YYYY-MM-DD>-<slug>/``, writes a manifest with the parameters,
script hashes, commit and environment versions, runs every stage, keeps
``log.txt`` and writes ``metrics.json`` next to it. A run directory is never
overwritten without ``--resume``.

Other subcommands::

    python exp.py show <run_id>      compact run summary
    python exp.py list               all runs in the registry
    python exp.py index              rebuild runs/INDEX.md and RESULTS.md
    python exp.py archive <run_id> --reason "..."

``--dry-run`` prints the exact argv of every stage and executes nothing: the
mode for preparing an experiment that someone else will launch, since GPU and
vLLM time is expensive.
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

# The driver deliberately stays at the repo root: `python exp.py run …` is the
# command recorded in every cmd.sh of the registry, and moving modules to src/
# must not change it. The price is this insertion: without it `import runlib`
# fails.
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import yaml

import runlib
from build_index_wiki_gte import utc_now


LOG = logging.getLogger("exp")

STAGES = ("retrieve", "judge", "score")
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}

# Stages run as subprocesses with cwd=runlib.REPO, so script paths are
# relative; this is also how they appear in the run's cmd.sh.
RETRIEVE_SCRIPT = "src/fullwiki_qrag.py"
POOL_SCRIPT = "src/candidate_pool.py"
VARIANTS_SCRIPT = "src/build_eval_variants.py"


class StageFailed(RuntimeError):
    pass


# --------------------------------------------------------------------------
# config


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    for key in ("label", "hypothesis", "retrieve"):
        if key not in config:
            raise ValueError(f"Config {path} lacks required field {key!r}")
    # The dataset is checked first: a typo there means a wrong coverage ceiling
    # in metrics.json and a wrong table in RESULTS.md, noticed only after an
    # hour of GPU time.
    if "dataset" in config and config["dataset"] not in runlib.DATASETS:
        raise ValueError(
            f"Unknown dataset {config['dataset']!r} in {path}; known: "
            f"{', '.join(runlib.DATASETS)}"
        )
    retrieve = config["retrieve"]
    kind = retrieve.get("kind")
    if kind not in ("qrag", "variant", "pool", "search"):
        raise ValueError(
            f"retrieve.kind must be qrag, variant, pool or search, got {kind!r}"
        )
    if kind == "search":
        # steps has a CLI default, but an eval run must name its budget
        # explicitly in the config: it decides which table rows are comparable.
        for key in ("index_dir", "qrag_repo", "input", "steps"):
            if not retrieve.get(key):
                raise ValueError(f"retrieve.{key} is required for kind=search")
    if kind == "variant":
        if retrieve.get("context") not in ("truncate", "none", "oracle"):
            raise ValueError("retrieve.context must be truncate, none or oracle")
        if not retrieve.get("source_run"):
            raise ValueError("retrieve.source_run is required for kind=variant")
    if kind == "pool":
        if not retrieve.get("source_run"):
            raise ValueError("retrieve.source_run is required for kind=pool")
        if not retrieve.get("index_dir"):
            raise ValueError("retrieve.index_dir is required for kind=pool")
        if not retrieve.get("steps"):
            raise ValueError("retrieve.steps is required for kind=pool")
        if retrieve.get("prefer") == "gold-sentence" and not retrieve.get("gold_source"):
            raise ValueError(
                "retrieve.gold_source is required with prefer=gold-sentence: "
                "gold sentences exist only in hotpot_dev_distractor_v1.json"
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
# running stages


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
    """Run a process with stdout and stderr appended to the run log."""
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
        # The flag is added only when the quota is set explicitly: otherwise
        # the argv of every earlier config would change, and it goes into the
        # manifest and cmd.sh.
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
        # Without a checkpoint the tower is stock GTE: the zero-shot starting
        # point that trained numbers are read against.
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
        # Both flags are added only when set: the argv of earlier configs must
        # stay unchanged, since it goes into the manifest and cmd.sh.
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
    """JSONL lines, split on ``\\n`` only.

    Not ``splitlines()``: it treats eight more characters as line breaks, and
    three of them (U+0085, U+2028, U+2029) are above 0x1F, so ``json.dumps``
    leaves them in the file unescaped (it escapes the other five). A record
    containing such a character falls apart in two, the judge gets a fragment
    and fails with ``JSONDecodeError``, and the record count in the log
    silently diverges from the input.

    This is not hypothetical: a few TriviaQA questions in the Search-R1 eval
    set contain U+0085 (NEXT LINE).
    """
    text = path.read_text(encoding="utf-8")
    return [line for line in text.split("\n") if line.strip()]


def check_jsonl_shape(lines: Sequence[str], path: Path) -> None:
    """Check that every line is a complete JSON object before the judge starts.

    A shape check rather than a parse: a full ``json.loads`` over hundreds of
    megabytes costs tens of seconds per run, while a fragment of a broken
    record is caught by not starting with ``{`` or not ending with ``}``.
    Without it a broken input shows up only as a few failed shards after
    minutes of work by 32 processes.
    """
    broken = [
        number
        for number, line in enumerate(lines, 1)
        if not (line.startswith("{") and line.rstrip().endswith("}"))
    ]
    if broken:
        raise StageFailed(
            f"{path}: lines {broken[:5]}"
            f"{' and ' + str(len(broken) - 5) + ' more' if len(broken) > 5 else ''} "
            "are not complete JSON objects. The usual cause is a character "
            "that json.dumps does not escape but the splitting code treats "
            "as a line break (see jsonl_lines)."
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
        raise StageFailed(f"retrieve exited with code {code}, see {run / runlib.LOG_FILE}")

    # build_eval_variants.py and candidate_pool.py have no --max-samples of
    # their own: a derived set must mirror the whole source run. For smoke runs
    # it is truncated here, after the variant is built.
    limit = retrieve.get("max_samples")
    if limit and retrieve["kind"] in ("variant", "pool"):
        lines = jsonl_lines(output)[:limit]
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        LOG.info("Smoke mode: kept %d records", len(lines))
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
    """Judge stage settings with defaults.

    The script is our fork, not the original ``answer_judge_llms.py``: reward
    contract v2 computes ``em_alias`` as the maximum over answer variants, and
    without it NQ EM is understated (11.97% vs 9.25% on the same predictions).
    ``contract: v1`` in the config restores the original behaviour bit for
    bit; older runs were produced with it.
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
    # The original script does not know these flags: a config with
    # contract: v1 and an explicit script keeps the old command line, so older
    # runs stay reproducible.
    if Path(judge["script"]).name == "answer_judge.py":
        argv += ["--contract", str(judge["contract"])]
        # Answer variants come from the dataset table, not from
        # retrieval.jsonl: the retrieval stage drops aliases.
        if judge["contract"] == "v2" and judge.get("dataset"):
            argv += ["--dataset", str(judge["dataset"])]
        if judge.get("judge_all"):
            argv += ["--judge-all"]
    return argv


def check_vllm(base_url: str) -> None:
    """Fail early if vLLM is down; otherwise 32 processes fail one by one."""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            response.read(1)
    except (urllib.error.URLError, OSError) as error:
        raise StageFailed(
            f"vLLM is not responding at {url}: {error}. "
            "Start the server first (see README.md)"
        ) from error


def stage_judge(config: dict[str, Any], run: Path) -> dict[str, Any]:
    judge = judge_settings(config)
    retrieval = run / runlib.RETRIEVAL_FILE
    if not retrieval.exists():
        raise StageFailed(f"Missing {retrieval}: run the retrieve stage first")
    check_vllm(str(judge["base_url"]))

    records = jsonl_lines(retrieval)
    check_jsonl_shape(records, retrieval)
    shards = max(1, min(int(judge["shards"]), len(records)))
    parts_dir = run / "judge-shards"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True)

    # Contiguous chunks: id order within and across shards is preserved, so
    # merging restores the original sequence.
    per_shard = -(-len(records) // shards)
    shard_files = []
    for number in range(shards):
        chunk = records[number * per_shard : (number + 1) * per_shard]
        if not chunk:
            break
        shard_input = parts_dir / f"input-{number:02d}.jsonl"
        shard_input.write_text("\n".join(chunk) + "\n", encoding="utf-8")
        shard_files.append((shard_input, parts_dir / f"input-{number:02d}.json"))

    LOG.info("judge: %d records, %d shards", len(records), len(shard_files))
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
            f"{len(failures)} of {len(processes)} judge shards failed, "
            f"see {log_path}"
        )

    merged: list[dict[str, Any]] = []
    for _, shard_output in processes:
        with shard_output.open("r", encoding="utf-8") as stream:
            merged.extend(json.load(stream))
    destination = run / runlib.JUDGE_FILE
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(merged, stream, ensure_ascii=False)

    # Order and membership must match the retrieval JSONL; otherwise metrics
    # are computed over the wrong questions.
    expected = [json.loads(line)["id"] for line in records]
    actual = [record["id"] for record in merged]
    if expected != actual:
        raise StageFailed(
            "id order after merging shards does not match retrieval.jsonl"
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
        raise StageFailed(f"Missing {judge_file}: run the judge stage first")
    started = time.monotonic()
    dataset = runlib.dataset_key(config)
    metrics = runlib.score_judge_file(judge_file, dataset=dataset)
    runlib.write_metrics(run, metrics)
    # Formatted via a helper rather than %.4f: for a dataset without gold
    # titles title_em is None, and "%.4f" would fail after the metrics are
    # written, making an intact run look failed.
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
# subcommands


def write_cmd_script(config: dict[str, Any], run: Path) -> None:
    retrieve = config["retrieve"]
    judge = judge_settings(config)
    prefix = ""
    if retrieve.get("cuda_visible_devices") is not None:
        prefix = f"CUDA_VISIBLE_DEVICES={retrieve['cuda_visible_devices']} "
    lines = [
        "#!/usr/bin/env bash",
        f"# Run {run.name}, generated by exp.py from configs/{config['config_file']}.",
        "# Written before any stage starts, so the run stays reproducible even if",
        "# the process is killed.",
        "set -euo pipefail",
        f"cd {runlib.REPO}",
        "",
        prefix + "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\",
        "  " + " ".join(retrieve_argv(config, run)),
        "",
        f"# judge: {judge['shards']} shards of answer_judge_llms.py, then merged",
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
        print(f"directory: {run.relative_to(runlib.REPO)}")
        print(f"hypothesis: {config['hypothesis']}")
        for stage in stages:
            print(f"\n— stage {stage}")
            if stage == "retrieve":
                environment = config["retrieve"].get("cuda_visible_devices")
                if environment is not None:
                    print(f"  CUDA_VISIBLE_DEVICES={environment} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
                print("  " + " ".join(retrieve_argv(config, run)))
            elif stage == "judge":
                judge = judge_settings(config)
                print(f"  {judge['shards']} × answer_judge_llms.py (VLLM_API_KEY unset)")
                print("  " + " ".join(shard_argv(judge, Path("<shard>.jsonl"), Path("<shard>.json"))))
            else:
                print("  runlib.score_judge_file → metrics.json")
        return 0

    if run.exists() and not args.resume:
        raise SystemExit(
            f"Directory {run} already exists. Pass --tag, --run-id or "
            "--resume so the previous result is not lost."
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
            "The repository has uncommitted changes: %s. "
            "The run will be marked git_dirty.",
            ", ".join(manifest["code"]["git_dirty_files"][:5]),
        )

    done = {stage["name"] for stage in manifest["stages"] if stage.get("exit_code") == 0}
    for stage in stages:
        if args.resume and stage in done:
            LOG.info("Skipping stage %s: already done", stage)
            continue
        LOG.info("Stage %s", stage)
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
            LOG.error("Stage %s failed: %s", stage, error)
            return 1
        manifest["stages"] = [
            existing for existing in manifest["stages"] if existing["name"] != stage
        ] + [result]
        runlib.write_manifest(run, manifest)

    manifest["status"] = "ok"
    manifest["finished_utc"] = utc_now()
    runlib.write_manifest(run, manifest)
    LOG.info("Done: %s", run.relative_to(runlib.REPO))
    LOG.info("Remember to run `make report`")
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
        # For a trained checkpoint the weights define what the table row
        # measures: without their identity the run is not reproducible.
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
            # The corpus and its offsets are needed to resolve pool row IDs
            # into texts; the identity comes from the index manifest, as for qrag.
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
    print(f"label:      {config.get('label')}")
    print(f"hypothesis: {manifest.get('hypothesis')}")
    print(f"started:    {manifest.get('created_utc')}")
    print(f"finished:   {manifest.get('finished_utc')}")
    print(f"commit:     {manifest['code'].get('git_commit')}"
          f"{' (dirty)' if manifest['code'].get('git_dirty') else ''}")
    if manifest.get("backfilled"):
        print("backfilled: parameters were reconstructed, not recorded at launch")
    print("\nstages:")
    for stage in manifest.get("stages", []):
        seconds = stage.get("seconds")
        print(
            f"  {stage['name']:9s} code={stage.get('exit_code')} "
            f"{'' if seconds is None else f'{seconds:.0f} s '}"
            f"{stage.get('output', '')}"
        )
    if metrics:
        print("\nmetrics:")
        for key in ("em", "f1", "judge", "title_recall", "title_em", "samples"):
            value = metrics.get(key)
            if isinstance(value, float):
                print(f"  {key:14s} {value * 100:.2f}%")
            else:
                print(f"  {key:14s} {value}")
    retrieve = config.get("retrieve", {})
    print(
        f"\nretrieval:  {retrieve.get('kind')}, "
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
        raise SystemExit(f"No such run: {run}")
    runlib.ARCHIVE.mkdir(parents=True, exist_ok=True)
    destination = runlib.ARCHIVE / run.name
    if destination.exists():
        raise SystemExit(f"Archive already contains {destination}")
    manifest_path = run / runlib.MANIFEST_FILE
    if manifest_path.exists():
        manifest = runlib.read_manifest(run)
        manifest["archived_utc"] = utc_now()
        manifest["archive_reason"] = args.reason
        runlib.write_manifest(run, manifest)
    shutil.move(str(run), str(destination))
    # runs/INDEX.md promises that each run's archive reason is recorded in the
    # archive README, so it is appended there, not only to the manifest.
    readme = runlib.ARCHIVE / "README.md"
    entry = f"## `{run.name}`\n\n{args.reason}\n\n"
    with readme.open("a", encoding="utf-8") as stream:
        stream.write(entry)
    LOG.info("Run moved to %s: %s", destination.relative_to(runlib.REPO), args.reason)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run an experiment")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--run-id", default=None, help="override run_id")
    run_parser.add_argument("--tag", default=None, help="suffix appended to the run_id slug")
    run_parser.add_argument(
        "--only", action="append", choices=STAGES, help="run only these stages"
    )
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument(
        "--resume", action="store_true", help="continue an existing run"
    )
    run_parser.add_argument(
        "--max-samples", type=int, default=None, help="limit the number of samples (smoke)"
    )
    run_parser.set_defaults(handler=command_run)

    show_parser = subparsers.add_parser("show", help="run summary")
    show_parser.add_argument("run_id")
    show_parser.set_defaults(handler=command_show)

    list_parser = subparsers.add_parser("list", help="all runs")
    list_parser.set_defaults(handler=command_list)

    index_parser = subparsers.add_parser("index", help="rebuild INDEX.md and RESULTS.md")
    index_parser.set_defaults(handler=command_index)

    archive_parser = subparsers.add_parser("archive", help="move a run to the archive")
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
        LOG.error("Interrupted")
        raise SystemExit(130)
    except Exception as error:
        LOG.error("Failed: %s", error)
        raise
