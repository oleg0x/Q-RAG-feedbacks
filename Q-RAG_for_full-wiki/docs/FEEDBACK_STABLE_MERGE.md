# Feedback-stable Merge Inventory

This repository keeps `Q-RAG-pqn` as the conflict source of truth and ports
features from the feedback-stable snapshot at subsystem level.

## Transferred

- Q-ICL loaders and adapters for GSM8K, MATH, HellaSwag, MMLU-Pro, and XL-Sum.
- Source-aware combined HotPotQA+2Wiki datasets and candidate adapters.
- `train-clean` support for prepared HotPotQA/2Wiki q0_s1 JSONL files.
- `LlmAnswer` Q-ICL feedback and math-specific Gold Shift feedback.
- Named Q-ICL/combined training configs; `training.yaml` remains the default.
- Fixed-without-replacement evaluation, fact metrics, and per-source metrics.
- Q-ICL train/eval entrypoints, remote reader/judge and math eval entrypoints.
- Sanitized sync/async vLLM clients and parameterized deployment assets.
- Prompt and MATH preparation utilities plus portable `requirements.txt`.

## Mapped to newer implementations

- Snapshot `candidates/` maps to the newer `gig_pipeline/`; no duplicate was
  copied.
- Snapshot `q-rag.def` maps to the equivalent maintained
  `ms-retr-rl.def`.
- Snapshot `vLLM_clients/tests.py` maps to
  `vLLM_clients/manual_integration.py`; it is never run by pytest.
- Snapshot `run_rsync.sh` was deployment-specific and contained a private
  host/user path. It was not copied; data transfer remains an operator action.

## Conflict decisions

- `rl/agents/pqn.py` and `envs/parallel_env.py` remain the base versions,
  preserving TD(lambda), bootstrap, and final-transition fixes.
- Candidate questions stay verbatim, including trailing `?`, so beta and
  reward prompts remain byte-identical.
- Existing HotPotQA/2Wiki split behavior and normal RNG paths were retained;
  `train-clean` is additive.
- Existing eval CLIs remain available; optional Q-ICL/judge/sample-limit flags
  were added without changing defaults.
- vLLM credentials are supplied only through environment variables or CLI
  values. Snapshot literals and private hosts were not transferred.

## Path-level coverage

The preflight comparison found 51 paths present only in feedback-stable. The
target contains 43 corresponding paths. They are grouped as follows:

- six `VLLM-deployment/` scripts/configs;
- `answer_judge_llms.py`, `eval_llm_math.py`, `eval_q-icl.py`, and
  `train_q-icl.py`;
- six additional `configs/envs/`, `configs/feedback/llm.yaml`, and eight
  named `configs/training_*.yaml` scenarios;
- six `envs/dataloaders/` modules;
- `rl/feedback/llm_answer.py` and
  `rl/feedback/gold_shift_feedback_math.py`;
- `prompts_and_metrics/prompts.py`, all three requested `utils/` modules, and
  the portable `requirements.txt`;
- `vLLM_clients/sync_vllm_client.py`, `vllm_client.py`, `vllm_clients.py`,
  `vllm.def`, and `vllm_config.yaml`.

The remaining eight paths are intentional mappings, not omissions:

| Feedback-stable path | Target equivalent / decision |
| --- | --- |
| `candidates/compute_beta.py` | newer `gig_pipeline/compute_beta.py` |
| `candidates/generate_candidates.py` | newer `gig_pipeline/generate_candidates.py` |
| `candidates/gig_common.py` | maintained `gig_pipeline/gig_common.py` |
| `candidates/judge_candidates.py` | newer `gig_pipeline/judge_candidates.py` |
| `candidates/prepare_dataset.py` | newer `gig_pipeline/prepare_dataset.py` |
| `q-rag.def` | equivalent maintained `ms-retr-rl.def` |
| `run_rsync.sh` | omitted private-host deployment operation |
| `vLLM_clients/tests.py` | explicit manual `vLLM_clients/manual_integration.py` |

Shared paths with content differences were reconciled at feature level. In
particular, dataset registration/adapters, combined datasets, feedback
registration/auth, training fixed-eval metrics, and optional OpenQA flags were
ported; debug prints, stale defaults, and snapshot regressions were not.
