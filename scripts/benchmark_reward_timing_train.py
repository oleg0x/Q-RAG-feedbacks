#!/usr/bin/env python3
"""
Benchmark wall time of ``AFeedbackModel.reward()`` during Q-RAG training.

Uses ``configs/training.yaml`` as the Hydra root; each profile overrides
``feedback=<group>``, ``feedback.type``, ``envs``, and ``algo`` per dataset.

Profiles → Hydra ``feedback`` group + ``feedback.type``:
  em, f1, semantic_similarity, llm_judge              —  feedback=base_rewards
  relevance_judge                                     —  feedback=relevance_judge
  keyword_f1_judge, answer_critical_judge               —  feedback=keyword_rewards
  sep, ssig                                           —  feedback=defaults
  ig                                                  —  feedback=gold_shift
  candidate_beta                                      —  feedback=candidate_beta

Datasets (``--dataset``):
  hotpotqa  —  envs=hotpotqa, algo=pqn_e5_hotpotqa; CB → hotpotqa_candidate
  musique   —  envs=musique, algo=pqn_e5_musique; CB → musique_candidate

Stops after ``--n-samples`` log lines ``qrag.reward_ms=...`` (see rl.feedback.feedback).
By default only ``is_final=True`` lines are counted (actual LLM reward steps).

On subprocess crash (OOM, API errors, etc.) the run is retried until enough samples
or ``--max-retries`` is exhausted. Retries first lower ``batch_size`` / vLLM concurrency;
context truncation is enabled only on the last recovery tier.

Example:
    python scripts/benchmark_reward_timing_train.py \\
      --dataset hotpotqa musique --rewards em sep candidate_beta --n-samples 100
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REWARD_MS_RE = re.compile(
    r"qrag\.reward_ms=([\d.]+)\s+feedback=(\S+)\s+is_final=(True|False)\s+reward=([-0-9.eE+\-]+)"
)

# Subprocess died with these in the log → retry with safer settings.
FATAL_LOG_PATTERNS = (
    re.compile(r"OutOfMemoryError", re.I),
    re.compile(r"CUDA out of memory", re.I),
    re.compile(r"API error:\s*(429|500|502|503|504)", re.I),
    re.compile(r"Local API (timeout|error)", re.I),
    re.compile(r"ContextWindowExceeded", re.I),
    re.compile(r"maximum context length", re.I),
)


@dataclass(frozen=True)
class DatasetBench:
    """Env + algo Hydra groups for one benchmark dataset."""

    envs: str
    candidate_envs: str
    algo: str


DATASET_BENCH: dict[str, DatasetBench] = {
    "hotpotqa": DatasetBench(
        envs="hotpotqa",
        candidate_envs="hotpotqa_candidate",
        algo="pqn_e5_hotpotqa",
    ),
    "musique": DatasetBench(
        envs="musique",
        candidate_envs="musique_candidate",
        algo="pqn_e5_musique",
    ),
}


@dataclass(frozen=True)
class RewardProfile:
    """Maps a benchmark CLI name to Hydra ``feedback`` group + ``feedback.type``."""

    feedback_config: str
    feedback_type: str
    extra_overrides: tuple[str, ...] = ()


# CLI --rewards → (feedback config group, feedback.type); settings in configs/feedback/*.yaml
REWARD_PROFILES: dict[str, RewardProfile] = {
    "em": RewardProfile("base_rewards", "em"),
    "f1": RewardProfile("base_rewards", "f1"),
    "semantic_similarity": RewardProfile("base_rewards", "semantic_similarity"),
    "llm_judge": RewardProfile("base_rewards", "llm_judge"),
    "relevance_judge": RewardProfile("relevance_judge", "relevance_judge"),
    "keyword_f1_judge": RewardProfile("keyword_rewards", "keyword_f1_judge"),
    "answer_critical_judge": RewardProfile("keyword_rewards", "answer_critical_judge"),
    "sep": RewardProfile("defaults", "sep"),
    "ssig": RewardProfile("defaults", "info_gain"),
    "ig": RewardProfile("gold_shift", "gold_shift"),
    "candidate_beta": RewardProfile("candidate_beta", "candidate_beta"),
}

# LLMGenerator blocks that use ``feedback._formatter`` (YAML anchor).
BASE_REWARDS_LLM_KEYS = frozenset(
    {
        "em",
        "f1",
        "llm_judge",
        "relevance_judge",
        "keyword_f1_judge",
        "answer_critical_judge",
    }
)

DEFAULT_REWARDS = tuple(REWARD_PROFILES.keys())


@dataclass
class RewardRunResult:
    reward_type: str
    dataset: str
    n_samples: int
    avg_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    log_path: Optional[str] = None
    error: Optional[str] = None
    feedback_config: Optional[str] = None
    hydra_envs: Optional[str] = None
    algo: Optional[str] = None
    n_attempts: int = 1


@dataclass
class BenchRunConfig:
    n_samples: int
    max_retries: int
    retry_delay_s: float
    only_final: bool
    max_context_chars: int
    max_chunks: int
    max_chunk_chars: int
    batch_size: int
    accumulate_grads: int
    vllm_max_concurrent: int


def _envs_for_profile(profile: RewardProfile, dataset: DatasetBench) -> str:
    if profile.feedback_type == "candidate_beta":
        return dataset.candidate_envs
    return dataset.envs


def _batch_size_for_tier(base_batch: int, tier: int) -> int:
    """Progressively smaller PQN batch on retries (tier 0 = user/default)."""
    if tier <= 0:
        return base_batch
    if tier == 1:
        return max(1, base_batch // 2)
    return 1


def _vllm_concurrency_for_tier(base: int, tier: int) -> int:
    if tier <= 0:
        return base
    if tier == 1:
        return max(1, base // 2)
    return 1


def _llm_safe_overrides(
    profile: RewardProfile,
    cfg: BenchRunConfig,
    *,
    recovery_tier: int,
    truncate_context: bool,
) -> list[str]:
    """Hydra overrides: smaller batch / concurrency first; truncate only if requested."""
    overrides: list[str] = []
    reward_key = next(
        (k for k, v in REWARD_PROFILES.items() if v == profile),
        profile.feedback_type,
    )
    vllm_conc = _vllm_concurrency_for_tier(cfg.vllm_max_concurrent, recovery_tier)

    if reward_key in BASE_REWARDS_LLM_KEYS:
        if truncate_context:
            overrides.extend(
                [
                    "feedback._formatter._target_=prompts_and_metrics.general_qa.TruncatingSysQAPromptFormatter",
                    f"feedback._formatter.max_context_chars={cfg.max_context_chars}",
                    f"feedback._formatter.max_chunks={cfg.max_chunks}",
                    f"feedback._formatter.max_chunk_chars={cfg.max_chunk_chars}",
                ]
            )
        overrides.append(f"feedback.max_at_same_time={vllm_conc}")
    elif reward_key == "ssig":
        overrides.extend(
            [
                f"feedback.max_at_same_time={vllm_conc}",
                "feedback.ig_sampling_params.max_tokens=32",
            ]
        )
    elif reward_key == "candidate_beta":
        overrides.append(f"feedback.candidate_beta.max_concurrent={vllm_conc}")

    if reward_key == "semantic_similarity":
        overrides.append("feedback.sem_embed_device=cuda:0")

    bs = _batch_size_for_tier(cfg.batch_size, recovery_tier)
    ag = 1 if recovery_tier >= 1 else cfg.accumulate_grads
    overrides.extend(
        [
            f"batch_size={bs}",
            f"accumulate_grads={ag}",
        ]
    )
    return overrides


def _build_hydra_overrides(
    profile: RewardProfile,
    dataset: DatasetBench,
    *,
    bench_label: str,
    cuda_device: str,
    learning_start: int,
    eval_interval: int,
    envs_parallel: int,
    log_reward_timing: bool,
    run_cfg: BenchRunConfig,
    recovery_tier: int,
    truncate_context: bool,
) -> list[str]:
    device = (
        f"cuda:{cuda_device}"
        if cuda_device.isdigit()
        else (cuda_device if cuda_device.startswith("cuda") else f"cuda:{cuda_device}")
    )
    envs = _envs_for_profile(profile, dataset)
    ep = 1 if recovery_tier >= 1 else envs_parallel
    overrides = [
        f"envs={envs}",
        f"algo={dataset.algo}",
        f"feedback={profile.feedback_config}",
        f"feedback.type={profile.feedback_type}",
        f"device={device}",
        f"learning_start={learning_start}",
        f"eval_interval={eval_interval}",
        "eval_episodes=1",
        f"envs_parallel={ep}",
        f"log_reward_timing={'true' if log_reward_timing else 'false'}",
        f"logger.tensorboard.comment=_bench_{bench_label}",
    ]
    overrides.extend(profile.extra_overrides)
    overrides.extend(
        _llm_safe_overrides(
            profile,
            run_cfg,
            recovery_tier=recovery_tier,
            truncate_context=truncate_context,
        )
    )
    return overrides


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def _parse_reward_ms_line(
    line: str,
    *,
    only_final: bool,
    expected_feedback: str,
) -> Optional[float]:
    m = REWARD_MS_RE.search(line)
    if not m:
        return None
    if only_final and m.group(3) != "True":
        return None
    if m.group(2) != expected_feedback:
        return None
    return float(m.group(1))


def _detect_fatal_in_log(tail: str) -> Optional[str]:
    for pat in FATAL_LOG_PATTERNS:
        if pat.search(tail):
            return pat.pattern
    return None


def _stream_train_until_samples(
    proc: subprocess.Popen,
    log_f,
    *,
    run_key: str,
    n_target: int,
    samples: list[float],
    only_final: bool,
    expected_feedback: str,
    t_start: float,
) -> tuple[Optional[int], str, bool]:
    """
    Read subprocess stdout until ``n_target`` new timing lines or process exit.

    Returns (exit_code, log_tail, saw_fatal).
    """
    assert proc.stdout is not None
    log_tail_lines: list[str] = []
    max_tail = 80
    saw_fatal = False

    for line in proc.stdout:
        log_f.write(line)
        log_f.flush()
        log_tail_lines.append(line)
        if len(log_tail_lines) > max_tail:
            log_tail_lines.pop(0)

        if _detect_fatal_in_log(line):
            saw_fatal = True

        ms = _parse_reward_ms_line(
            line,
            only_final=only_final,
            expected_feedback=expected_feedback,
        )
        if ms is None:
            continue
        samples.append(ms)
        if len(samples) % 10 == 0 or len(samples) == n_target:
            print(
                f"  [{run_key}] {len(samples)}/{n_target} "
                f"last={ms:.2f} ms  elapsed={time.perf_counter() - t_start:.1f}s",
                flush=True,
            )
        if len(samples) >= n_target:
            print(f"  [{run_key}] reached {n_target} samples — stopping training.")
            _terminate_process(proc)
            break

    tail = "".join(log_tail_lines)
    if proc.poll() is None:
        _terminate_process(proc)
    rc = proc.wait(timeout=120)
    return rc, tail, saw_fatal


def _run_one_reward(
    run_key: str,
    profile: RewardProfile,
    dataset_name: str,
    dataset: DatasetBench,
    *,
    train_script: Path,
    log_dir: Path,
    hydra_builder,
    run_cfg: BenchRunConfig,
    cuda_device: str,
    learning_start: int,
    eval_interval: int,
    envs_parallel: int,
) -> RewardRunResult:
    log_path = log_dir / f"train_{run_key}.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    envs = _envs_for_profile(profile, dataset)
    expected_feedback = profile.feedback_type

    samples: list[float] = []
    t_start = time.perf_counter()
    n_attempts = 0
    last_error: Optional[str] = None
    recovery_tier = 0
    truncate_context = False
    # Tier 0: user batch; 1–2: halve batch / vLLM concurrency; 3+: truncate context (last resort).
    max_recovery_tier = 3

    with open(log_path, "w", encoding="utf-8") as log_f:
        while len(samples) < run_cfg.n_samples and n_attempts <= run_cfg.max_retries:
            n_attempts += 1
            overrides = hydra_builder(
                recovery_tier=recovery_tier,
                truncate_context=truncate_context,
            )
            cmd = [sys.executable, str(train_script), *overrides]

            if n_attempts > 1:
                sep = f"\n{'=' * 72}\nRETRY attempt {n_attempts}/{run_cfg.max_retries + 1}\n{'=' * 72}\n"
                log_f.write(sep)
                log_f.flush()
                print(f"  [{run_key}] retry {n_attempts}: waiting {run_cfg.retry_delay_s:.0f}s …")
                time.sleep(run_cfg.retry_delay_s)

            print(f"\n{'=' * 72}")
            print(
                f"Run: {run_key}  attempt={n_attempts}  dataset={dataset_name}  "
                f"algo={dataset.algo}  envs={envs}  feedback={profile.feedback_config}  "
                f"type={profile.feedback_type}"
            )
            print(f"Command: {' '.join(cmd)}")
            print(f"Log: {log_path}")
            bs = _batch_size_for_tier(run_cfg.batch_size, recovery_tier)
            print(
                f"Target samples: {run_cfg.n_samples} (have {len(samples)}, only_final={run_cfg.only_final})  "
                f"recovery_tier={recovery_tier} batch_size={bs} truncate={truncate_context}"
            )
            print(f"{'=' * 72}\n")

            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            try:
                rc, tail, saw_fatal = _stream_train_until_samples(
                    proc,
                    log_f,
                    run_key=run_key,
                    n_target=run_cfg.n_samples,
                    samples=samples,
                    only_final=run_cfg.only_final,
                    expected_feedback=expected_feedback,
                    t_start=t_start,
                )
            except Exception as e:
                _terminate_process(proc)
                last_error = str(e)
                continue

            if len(samples) >= run_cfg.n_samples:
                break

            fatal = _detect_fatal_in_log(tail)
            if rc != 0:
                last_error = f"train_q_rag.py exited with code {rc}"
            elif fatal:
                last_error = f"fatal log pattern: {fatal}"
            else:
                last_error = f"only collected {len(samples)}/{run_cfg.n_samples} timing lines"

            context_fatal = bool(
                fatal
                and re.search(
                    r"ContextWindowExceeded|maximum context length",
                    tail,
                    re.I,
                )
            )
            if recovery_tier < max_recovery_tier:
                recovery_tier += 1
            if context_fatal and recovery_tier >= 2:
                truncate_context = True
            elif recovery_tier >= max_recovery_tier:
                truncate_context = True

            if recovery_tier >= 1:
                print(
                    f"  [{run_key}] next attempt: tier={recovery_tier} "
                    f"batch_size={_batch_size_for_tier(run_cfg.batch_size, recovery_tier)} "
                    f"truncate={truncate_context}",
                )

            if n_attempts > run_cfg.max_retries:
                print(f"WARNING [{run_key}]: {last_error} (no retries left)", file=sys.stderr)
                break
            print(f"WARNING [{run_key}]: {last_error} — will retry", file=sys.stderr)

    err = None if len(samples) >= run_cfg.n_samples else last_error
    return RewardRunResult(
        reward_type=run_key,
        dataset=dataset_name,
        n_samples=len(samples),
        avg_ms=statistics.mean(samples) if samples else float("nan"),
        median_ms=statistics.median(samples) if samples else float("nan"),
        min_ms=min(samples) if samples else float("nan"),
        max_ms=max(samples) if samples else float("nan"),
        log_path=str(log_path),
        error=err,
        feedback_config=profile.feedback_config,
        hydra_envs=envs,
        algo=dataset.algo,
        n_attempts=n_attempts,
    )


def _print_summary(results: list[RewardRunResult]) -> None:
    print("\n" + "=" * 96)
    print("Reward timing summary (qrag.reward_ms)")
    print("=" * 96)
    print(
        f"{'run':<28} {'dataset':<10} {'algo':<20} {'envs':<22} {'n':>5} "
        f"{'avg_ms':>9} {'median_ms':>9}  attempts  status"
    )
    print("-" * 96)
    for r in results:
        status = "ok" if r.error is None and r.n_samples > 0 else (r.error or "no samples")
        print(
            f"{r.reward_type:<28} {r.dataset:<10} {r.algo or '':<20} {r.hydra_envs or '':<22} "
            f"{r.n_samples:>5} {r.avg_ms:>9.2f} {r.median_ms:>9.2f}  {r.n_attempts:>3}      {status}"
        )
    print("=" * 96)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--dataset",
        nargs="+",
        default=["hotpotqa"],
        choices=list(DATASET_BENCH.keys()),
        help="Benchmark datasets (hotpotqa and/or musique)",
    )
    p.add_argument(
        "--rewards",
        nargs="+",
        default=list(DEFAULT_REWARDS),
        choices=list(REWARD_PROFILES.keys()),
        help="Reward profile keys (see REWARD_PROFILES; Hydra groups in configs/feedback/)",
    )
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--max-retries", type=int, default=8, help="Restart train_q_rag after crash until n-samples")
    p.add_argument("--retry-delay-s", type=float, default=15.0, help="Sleep between retries (vLLM/GPU recovery)")
    p.add_argument(
        "--only-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Count only is_final=True timing lines (actual LLM reward steps)",
    )
    p.add_argument(
        "--max-context-chars",
        type=int,
        default=12000,
        help="Only used with TruncatingSysQAPromptFormatter on last recovery tier",
    )
    p.add_argument("--max-chunks", type=int, default=16, help="Last-tier truncation only")
    p.add_argument("--max-chunk-chars", type=int, default=1500, help="Last-tier truncation only")
    p.add_argument("--vllm-max-concurrent", type=int, default=4, help="Cap parallel vLLM calls during benchmark")
    p.add_argument("--batch-size", type=int, default=4, help="PQN batch size (halved on retries before truncation)")
    p.add_argument("--accumulate-grads", type=int, default=1)
    p.add_argument("--cuda-device", default="0")
    p.add_argument("--learning-start", type=int, default=0)
    p.add_argument("--eval-interval", type=int, default=1_000_000)
    p.add_argument("--envs-parallel", type=int, default=1, help="Parallel envs (1 is safest with vLLM on same GPU)")
    p.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "reward_timing_benchmark",
    )
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--train-script", type=Path, default=PROJECT_ROOT / "train_q_rag.py")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_json = args.output_json or (args.log_dir / "summary.json")
    run_cfg = BenchRunConfig(
        n_samples=args.n_samples,
        max_retries=args.max_retries,
        retry_delay_s=args.retry_delay_s,
        only_final=args.only_final,
        max_context_chars=args.max_context_chars,
        max_chunks=args.max_chunks,
        max_chunk_chars=args.max_chunk_chars,
        batch_size=args.batch_size,
        accumulate_grads=args.accumulate_grads,
        vllm_max_concurrent=args.vllm_max_concurrent,
    )

    results: list[RewardRunResult] = []
    for dataset_name in args.dataset:
        dataset = DATASET_BENCH[dataset_name]
        for reward_key in args.rewards:
            profile = REWARD_PROFILES[reward_key]
            run_key = f"{dataset_name}_{reward_key}"

            def hydra_builder(*, recovery_tier: int, truncate_context: bool) -> list[str]:
                return _build_hydra_overrides(
                    profile,
                    dataset,
                    bench_label=run_key,
                    cuda_device=args.cuda_device,
                    learning_start=args.learning_start,
                    eval_interval=args.eval_interval,
                    envs_parallel=args.envs_parallel,
                    log_reward_timing=True,
                    run_cfg=run_cfg,
                    recovery_tier=recovery_tier,
                    truncate_context=truncate_context,
                )

            result = _run_one_reward(
                run_key,
                profile,
                dataset_name,
                dataset,
                train_script=args.train_script,
                log_dir=args.log_dir,
                hydra_builder=hydra_builder,
                run_cfg=run_cfg,
                cuda_device=args.cuda_device,
                learning_start=args.learning_start,
                eval_interval=args.eval_interval,
                envs_parallel=args.envs_parallel,
            )
            results.append(result)
            if result.error:
                print(f"WARNING [{run_key}]: {result.error}", file=sys.stderr)

    _print_summary(results)

    payload = {
        "n_samples_target": args.n_samples,
        "only_final": args.only_final,
        "max_retries": args.max_retries,
        "max_context_chars": args.max_context_chars,
        "vllm_max_concurrent": args.vllm_max_concurrent,
        "datasets": {
            k: {"envs": v.envs, "candidate_envs": v.candidate_envs, "algo": v.algo}
            for k, v in DATASET_BENCH.items()
        },
        "profiles": {
            k: {
                "feedback_config": v.feedback_config,
                "feedback_type": v.feedback_type,
            }
            for k, v in REWARD_PROFILES.items()
        },
        "results": [
            {
                "reward_type": r.reward_type,
                "dataset": r.dataset,
                "algo": r.algo,
                "feedback_config": r.feedback_config,
                "hydra_envs": r.hydra_envs,
                "n_samples": r.n_samples,
                "avg_ms": r.avg_ms,
                "median_ms": r.median_ms,
                "min_ms": r.min_ms,
                "max_ms": r.max_ms,
                "log_path": r.log_path,
                "error": r.error,
                "n_attempts": r.n_attempts,
            }
            for r in results
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out_json}")

    failed = [r for r in results if r.error or r.n_samples < args.n_samples]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
