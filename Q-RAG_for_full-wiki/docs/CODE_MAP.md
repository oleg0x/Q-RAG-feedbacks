# Code Map

## Top-Level Layout

```text
/home/a.anokhin/Judge
├── Q-RAG-feedback/              # the codebase (git repo, branch feedback-integration)
│   ├── gig_pipeline/       # Group IG candidate dataset pipeline (moved from ~/Judge root)
│   └── docs/               # this documentation
├── datasets/               # local benchmark data
├── output/                 # generated candidate/beta datasets
├── old/                    # older generated Hotpot artifacts
├── experiments/            # ad-hoc experiment artifacts
└── notebooks/              # ad-hoc experiments
```

`Q-RAG-feedback/` is a git repository since 2026-07-10: history from branch `Judged`, current branch `feedback-integration`, remote
`upstream` = `https://github.com/griver/Q-RAG-feedback-project.git` (private,
needs a classic PAT to fetch). The old `Q-RAG/` near-duplicate tree no longer
exists.

## `Q-RAG-feedback/`

This is the main code tree to read/use first.

### Entrypoints

- `train_q_rag.py` - main RL training loop.
- `eval_retriever.py` - evaluates trained retriever checkpoints and writes JSONL retrieval logs.
- `eval_llm_openqa.py` - reader LLM evaluation for HotpotQA/MuSiQue-style logs (CLI: `--retriever_logfile`, `--llm_name`, `--max_tokens`, `--gpu_util`, `--think`).
- `eval_llm_synthetics.py` - reader LLM evaluation for BabiLong/RULER-style logs with chunk filters.
- `eval_search_r1.py` - Search-R1 baseline inference over HotpotQA/MuSiQue/2Wiki, writes eval_retriever-style JSONL.

Merged from feedback-stable (2026-07-17):

- `train_q-icl.py` - compatibility wrapper: sets
  `Q_RAG_CONFIG_NAME=training_hotpotqa.yaml` and re-executes `train_q_rag.py`.
- `eval_q-icl.py` - evaluates a trained Q-ICL retriever checkpoint
  (`--checkpoint-dir`, `--checkpoint best|last`, `--num-samples`, `--output-file`).
- `eval_llm_math.py` - math reader + judge eval against a deployed vLLM server.
- `answer_judge_llms.py` - OpenQA reader + judge eval against a deployed vLLM
  server (no in-process vLLM).
- `scripts/compose_configs.py` - composes/resolves top-level Hydra configs
  without instantiating models (`--all`, `--show`, `--override`).

Removed in the 2026-07-09 upstream cleanup (merged 2026-07-10): `eval_feedback.py`,
`eval_llm_longbench.py`, `legacy_scripts/`, `requirements-old.txt`.

### Configs

- `configs/training.yaml` - main training defaults and unchanged default selection.
- `configs/training_2wiki.yaml` and other `configs/training_*.yaml` files -
  named scenarios selected with `Q_RAG_CONFIG_NAME=<filename>`; CLI overrides
  remain supported.
- `configs/testing.yaml` - evaluation defaults merged with saved training config.
- `configs/envs/*.yaml` - dataset/environment modes.
- `configs/algo/*.yaml` - embedder and PQN hyperparameters.
- `configs/feedback/*.yaml` - reward model modes.
- `configs/logger/logging.yaml` - run directory and TensorBoard writer.

### Environments and Data Adapters

- `envs/text_env.py` - base text retrieval environment and position processors.
- `envs/qa_env.py` - QA-specific environment that calls feedback model.
- `envs/parallel_env.py` - batched rollout wrapper.
- `envs/utils.py` - `TextMemory`, `TextMemoryItem`, padding/tokenization helpers.
- `envs/qa_dataset_adapter.py` - converts raw datasets into `{id, question, answer, chunks, sf_idx}`.
- `envs/candidate_dataset_adapter.py` - loads Group IG/candidate-beta JSONL and returns candidates/judgements/betas too.
- `envs/combined_dataset.py` - combines HotpotQA/MuSiQue/2Wiki datasets; since
  the feedback-stable merge also `RetrievalCombinedHotpot2Wiki` (source-labeled
  HotpotQA+2Wiki) and `MinChunksDataset` (deterministic min-chunk filter view).
- `envs/chunker.py` - simple fixed-size context splitter for LongBench.
- `envs/words_counter_env.py` - old toy environment; imports non-existent `rl.text_env`, dead.

### Dataset Loaders

- `envs/dataloaders/hotpotqa.py` - HotpotQA train/dev/fullwiki loader; also the
  additive `train-clean` split for prepared q0_s1 JSONL files.
- `envs/dataloaders/musique.py` - MuSiQue loader.
- `envs/dataloaders/twowikimultihopqa.py` - 2WikiMultihopQA loader; also the
  additive `train-clean` split.
- `envs/dataloaders/base.py` - shared `RetrievalBase` for the Q-ICL loaders
  (train/eval example splitting, `separator` constant).
- `envs/dataloaders/gsm8k.py`, `competition_math.py`, `hellaswag.py`,
  `mmlupro.py`, `xlsum.py` - Q-ICL few-shot example loaders merged from
  feedback-stable.
- `envs/dataloaders/longbench.py` - LongBench JSONL loader.
- `envs/dataloaders/niah.py` - NIAH/RULER-like JSONL loader and sentence splitter.
- `envs/dataloaders/ruler_qa.py` - RULER QA parser.
- `envs/dataloaders/globalset.py` - older generic dataset mixer.
- `envs/dataloaders/babilong/*` - bAbI/BabiLong parsing, PG19 noise sampling, retrieval dataset.

### RL and Models

- `rl/agents/pqn.py` - main production agent. Since the 2026-07-10 merge the
  TD(lambda) targets are computed by a standalone `compute_returns()` and
  `PQN.update` expects `q_values` of shape `[num_envs, num_steps + 1]`
  (bootstrap value appended by the rollout).
- `rl/q_module.py` - Q-net, policy, target networks and value functions.
- `rl/bert_predictor.py` - HF encoder wrappers, projection heads, positional RoPE wrappers.
- `rl/optim.py` - warmup-linear/cosine schedulers and optimizer helper.
- `rl/feedback/*.py` - reward models; merged additions: `llm_answer.py`
  (`LlmAnswer`, Q-ICL/reader reward via remote vLLM) and
  `gold_shift_feedback_math.py` (`GoldShiftFeedbackMATH`).
- `rl/langchain_utils.py` - older FAISS/langchain helpers (needs langchain, not in requirements).
- `rl/text_env_bk.py` - backup/old text env; imports non-existent `rl.replay_buffer`, dead.
- `rl/agents/dqn.py`, `sarsa.py`, `sacd.py`, `agent.py` - legacy algorithms/experiments.

### Prompting, Metrics, Filters

- `prompts_and_metrics/general_qa.py` - normalized EM/F1 for generic QA.
- `prompts_and_metrics/babilong.py` - BabiLong prompt templates and task metrics.
- `prompts_and_metrics/chunk_filtering.py` - post-processing selected chunks before reader LLM eval.
- `prompts_and_metrics/answer_metric.py` - base metric interface.
- `prompts_and_metrics/prompts.py` - shared Q-ICL/reader/judge system prompts
  (merged from feedback-stable).

### vLLM Clients and Deployment (merged from feedback-stable)

- `vLLM_clients/vllm_client.py` - `BasicVllmClient` (OpenAI SDK, sync+async),
  used by `LlmAnswer`; `trust_env=False` by default so proxy env vars are
  ignored (opt back in via the `trust_env` parameter).
- `vLLM_clients/sync_vllm_client.py` - `SyncVllmClient` (requests-based), used
  by `eval_llm_math.py` / `answer_judge_llms.py`.
- `vLLM_clients/vllm_clients.py` - `AsyncVllmClient` / `AsyncBatchedVllmClient`
  (aiohttp).
- `vLLM_clients/manual_integration.py` - explicit manual connectivity check for
  a live server; never collected by pytest.
- `vLLM_clients/vllm.def`, `vllm_config.yaml` - serving assets.
- `VLLM-deployment/` - parameterized `run_vllm.sh` (requires `VLLM_API_KEY`
  env var) plus per-model H200 launch scripts/configs.
- `utils/` - `parse_boxed_answers.py`, `process_latex.py`, `prop_split.py` -
  math answer parsing/normalization helpers.

### Scripts, Tests, Notebooks

- `scripts/*.sh` and root `eval_*_babilong.sh` - convenience launchers; root
  `eval_llm_babilong.sh` calls `eval_llm_synthetics.py`, but
  `scripts/eval_llm_babilong.sh` still calls the non-existent `eval_llm.py`.
- `tests/` - clean pytest suites runnable offline: `test_hydra_configs.py`,
  `test_math_utils.py`, `test_prompt_consistency.py`, `test_qicl_datasets.py`,
  `test_train_clean_and_combined.py`, `test_vllm_clients.py`;
  `tests/conftest.py` excludes the two manual checks below from collection.
- `tests/test_candidate_feedback.py` - manual sanity check requiring vLLM
  (excluded from pytest collection).
- `tests/test_gold_shift.py` - manual sanity check requiring vLLM
  (excluded from pytest collection).
- `notebooks/*.ipynb` - experiments: FAISS, BabiLong examples, train model, words counter, early stopping.

### `gig_pipeline/` (Group IG candidate dataset pipeline)

Moved into the repo on 2026-07-10 (previously loose scripts in `~/Judge`).
See `gig_pipeline/README.md`. All pipeline scripts share `gig_common.py`:

- `gig_common.py` - shared QA prompt template, vLLM HTTP helpers, atomic JSONL I/O, resume + parallel scaffolding.
- `generate_candidates.py` - `--dataset {hotpot,musique,2wiki}`, `--mode {sample,guided}` — candidate answers per question under empty context. For 2Wiki it excludes `bridge_comparison`.
- `judge_candidates.py` - asks a judge LLM whether each candidate answer matches the gold answer. Not yet migrated to `gig_common`.
- `compute_beta.py` - computes no-context beta for every (normalized) candidate; by default appends the gold answer as the `K+1`-th positive candidate. Output is the final candidate-beta JSONL.
- `prepare_dataset.py` - `--dataset {hotpot,musique,2wiki}` — merges candidate-beta JSONL with raw context and writes the training file consumed by `CandidateDatasetAdapter`.
- `combine_candidate_train_jsonl.py` - merges several candidate_train files into one (adds `source`).
- `filter_candidate_train_by_success_ids.py` - keeps only questions selected by the success-rate experiment.
- `llm_as_judge_eval.py` - post-evaluates generated reader answers with LLM-as-judge.
- `make_2wiki_oracle_eval.py` - oracle retriever log for 2Wiki (pred = gold supporting facts).
- `run_success_rate_experiment.py` - runs the HotpotQA / 2Wiki question-only vs gold-supporting-facts experiment with Qwen3-4B, writes per-question success-rate JSONL/CSV, and plots histograms.
- `filter_success_rate_results.py` - filters success-rate JSONL records by configurable thresholds over `question_only_success_rate`, `with_support_facts_success_rate`, and their difference.

Removed in the Group IG cleanup (functionality folded into `compute_beta.py` / `prepare_dataset.py`):

- `compute_gold_beta.py`
- `prepare_candidate_dataset.py`
- `prepare_musique_dataset.py`

Root notebooks in `~/Judge/notebooks/` are kept as historical references;
candidate-gen logic has been moved into `generate_candidates.py`.

## Data and Artifacts

- `datasets/` - raw benchmark data: HotpotQA, MuSiQue, LongBench, NovelQA, BabiLong, etc.
- `output/` - generated candidate/beta/final datasets, especially `output/musique/*`, `output/hotpotqa/*`, `output/hotpotqa_guided/*`, `output/2wiki/*`.
- `Q-RAG-feedback/runs/` - ignored checkpoints, saved configs, TensorBoard logs, and eval outputs; historical source runs were not copied.
- `old/` - older Hotpot generated candidate/beta artifacts.

