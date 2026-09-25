# Known Issues and Sharp Edges

These are observations from reading the code, not a full test run.

## Fixed in the 2026-07-10 upstream merge

For history: two long-standing training bugs were fixed by merging
`upstream/Judged` (commits `0dc0c78`, `d1539aa`):

- `PQN.update` computed a wrong TD(lambda) target (off-by-one in the bootstrap
  of the manual first step); replaced by the standalone `compute_returns()`.
- `ParallelTextEnv.rollout` dropped the last transition of every env per
  rollout (`[:-1]` truncation); the new rollout keeps all transitions and
  appends one extra bootstrap q-value column instead.

Runs trained before this merge carry both biases.

## Fixed after the 2026-07-17 feedback-stable merge

- `7d124a0` (2026-07-17): remote vLLM clients (`SyncVllmClient`,
  `AsyncVllmClient`) left Qwen3 thinking enabled; `thinking` is now mapped to
  `chat_template_kwargs.enable_thinking` in the request payload, and the
  reader/judge entrypoints pass `thinking=False`.
- `393896e` (2026-07-21): `BasicVllmClient` inherited `HTTP(S)_PROXY`/
  `ALL_PROXY` env vars, breaking direct connections to a local vLLM server
  behind a proxied shell; it now uses `httpx` clients with `trust_env=False`
  by default (opt back in via the `trust_env` parameter).

Residual (not yet fixed): `SyncVllmClient` still uses a default
`requests.Session()`, which *does* honor proxy env vars — so
`eval_llm_math.py` / `answer_judge_llms.py` / sync `manual_integration.py`
can still route through a proxy; unset the proxy vars or set `NO_PROXY` when
targeting a local server. `AsyncVllmClient` (aiohttp) ignores proxy env by
default.

## Merge audit findings (2026-07-21, Low severity, not fixed)

From the independent post-merge audit; all are edge cases, none affect the
default configs:

- GSM8K Q-ICL loader seed changed vs feedback-stable (100 -> shared
  `RetrievalBase` default 1): example/sample selection is not bit-reproducible
  against old feedback-stable GSM8K runs. Pass `seed` explicitly if you need
  the old subsets. (The example *format* change is intentional: feedback-stable
  formatted GSM8K examples without the `separator`, which made `LlmAnswer`
  silently drop all of them.)
- `RetrievalBase` edge semantics differ from feedback-stable for
  `samples_num=0` (now "all remaining", was "empty") and `examples_num=0/>=len`
  (now "empty"/`ValueError`, was "all"). No current config hits these.
- 2Wiki `train-clean`: `skip_bridge_comparison` now also filters q0_s1 rows
  (feedback-stable applied it only to the raw split); only matters if q0_s1
  rows carry a `type` field.
- HotpotQA `train-clean` shuffles with `np.random` seed 52 (pqn style);
  feedback-stable used `random.Random(52)` — with `length` truncation the
  selected subset differs from old feedback-stable train-clean runs.
- `train-clean` loaders silently fall back to a default-named file in the
  directory if the passed `clean_filename` does not exist.
- `LlmAnswer`: an unfinished generation (`finish_reason != "stop"`) yields the
  sentinel `"[unfinished content]"`, which is non-empty and still triggers an
  LLM-as-judge call comparing garbage to the gold answer.
- `BasicVllmClient` never closes its `httpx` clients; each `LlmAnswer.copy()`
  (one per parallel env) creates two more. Harmless at small `envs_parallel`.
- Combined training with `feedback=candidate_beta`: the eval env inherits the
  same feedback via `${feedback.feedback_dict.${feedback.type}}`, and plain
  eval samples have no candidates/betas, so eval *reward* is always 0 (with a
  warning); per-source `fact_em`/`fact_f1` metrics remain meaningful. This is
  inherited feedback-stable design, not a merge regression.

## Paths Are Inconsistent

Some configs use absolute paths:

```text
/home/a.anokhin/Judge/...
```

Others use paths relative to `Q-RAG-feedback/`:

```text
../datasets/...
```

Most project commands should be run from:

```bash
cd /home/a.anokhin/Judge/Q-RAG-feedback
```

The candidate pipeline scripts (now in `Q-RAG-feedback/gig_pipeline/`) write to
`output/...` relative to the CWD by convention, so run them from
`/home/a.anokhin/Judge` as `python Q-RAG-feedback/gig_pipeline/<script>.py`.

## Guided-Mode Beta Gap

`generate_candidates.py --mode guided` synthesizes 4 paraphrases of the gold answer (intended `G+`) and 4 plausible distractors (intended `G−`) by two prompted calls. These candidates are **not** samples from `p_theta(. | x_i, c0)`, which is the assumption baked into Group IG's beta-subtraction.

In practice on HotpotQA train (`output/hotpotqa_guided/hotpot_final.jsonl`):

```text
intent_correct (G+ side)   mean β = -29.28   ← nonstandard paraphrases, low prior
intent_wrong   (G- side)   mean β = -17.74   ← "plausible" distractors, high prior
per-sample mean(G+) − mean(G−) ≈ −6.62
```

The gap is **inverted** compared to canonical Group IG (where both sides come from the same `p_theta(. | x, c0)` distribution and have comparable betas). Consequences:

- `G−` candidates already sit close to `log p ≈ 0`, so their shift `reward − beta` has a small ceiling. A useful context that helps both `G+` and `G−` similarly will yield near-zero Group IG reward.
- The method is still mathematically calibrated (beta is subtracted per-candidate, so any prior bias is removed). But the dynamic range of the reward is compressed on this dataset compared to the `sample`-mode dataset.

Before drawing conclusions from a guided-mode run, also train on the canonical `sample`-mode dataset as a baseline ablation. Do **not** try to "fix" this by renormalizing betas — that breaks the `reward − beta` identity.

## API Authentication in `gig_common`

`gig_common._vllm_post` and `judge_candidates.vllm_request` now read an optional `VLLM_API_KEY` env var and add `Authorization: Bearer <key>` when set. Required when the vLLM server is started with `--api-key`; harmless otherwise (no env var → no header → unchanged behavior).

Training-time `CandidateBetaFeedback`, its per-step variant, and
`GoldShiftFeedback` now read `VLLM_API_KEY` through their Hydra configs and
send a Bearer header when it is non-empty. The URL and model can likewise be
overridden with `VLLM_BASE_URL` and `VLLM_MODEL`; unauthenticated defaults
remain compatible with the previous flow.

## Candidate Judgement `-1`

`judge_candidates.py` returns `-1` on failed judge calls. `CandidateBetaFeedback` treats any judgement not equal to `1` as negative because it uses:

```python
if j == 1:
    G_plus.append(...)
else:
    G_minus.append(...)
```

Before Group IG training, check/fix unresolved judgements.

## IG / Group IG Prompt Mismatch Risk

For clean `IG` / `Group IG`, beta and reward must use the same prompt template so that `reward − beta` is purely a context-induced likelihood shift.

**Group IG: fixed.** `gig_common.QA_USER_TEMPLATE` (a single `GIVEN PASSAGES:\n{context}\n\nQUESTION:\n{question}\n\n...` template) is used by `generate_candidates.py`, `compute_beta.py` (with `context=""`), and `CandidateBetaFeedback.QA_PROMPT` (mirrored byte-for-byte). `CandidateDatasetAdapter` no longer strips the trailing `?`. At `c=""` reward returns exactly `beta`.

**IG (`gold_shift`) and per-step: verified 2026-07-10, template drift found.**
`GoldShiftFeedback.QA_PROMPT` and `CandidateBetaPerStepFeedback.QA_PROMPT` have
a **leading newline** that `gig_common.QA_USER_TEMPLATE` /
`CandidateBetaFeedback.QA_PROMPT` do not. `gold_shift` is internally consistent
(beta computed at reset with the same template), and per-step rewards are score
deltas where precomputed betas cancel — but absolute per-step scores are on a
different likelihood scale than the stored betas. Align the templates before
mixing scales.

Other normalization notes:

- `compute_beta.py` normalizes candidates via `gig_common.strip_candidate_prefix` and writes the normalized string to JSONL. Reward-time scoring re-tokenizes the string from JSONL, so beta and reward tokenize the same bytes. The normalizer handles `Final answer: X`, `Answer: X`, and embedded `X\nFinal answer: X` patterns (uses `rfind` to keep the substring after the last label).
- `judge_candidates.py` still keeps its own copy of `strip_candidate_prefix` (with `rstrip(".")`); judging strips for display before sending to the judge LLM, but never writes the normalized string back. Could be migrated to `gig_common` for consistency.

## `scripts/eval_llm_babilong.sh` References Missing `eval_llm.py`

The root `Q-RAG-feedback/eval_llm_babilong.sh` was fixed and calls
`eval_llm_synthetics.py`, but the copy in `Q-RAG-feedback/scripts/eval_llm_babilong.sh`
still calls the non-existent `eval_llm.py` (and a hardcoded
`~/.mlspace/envs/msr` python). Use the root script.

## Legacy / Dead Modules Have Stale Imports

Files (kept in the tree deliberately, do not import them):

- `rl/agents/dqn.py`, `rl/agents/sacd.py`, `rl/agents/sarsa.py` - legacy agents.
- `rl/agents/agent.py` - depends on an external `../contriever` checkout and
  `langchain` (not in requirements); also `elif ['opt.scheduler'] == "cosine":`
  compares a list literal to a string, so its cosine branch is unreachable.
- `envs/words_counter_env.py` - imports non-existent `rl.text_env`, written
  against the old TextEnv API.
- `rl/text_env_bk.py` - imports non-existent `rl.replay_buffer`.
- `rl/agents/pqn.py::PQNActor` - calls `stack_text_list` without the required
  `max_length` and references undefined `self.state_tokenizer`.
- `rl/q_module.py::TextMaxQNet` - references `q_net.weight`/`q_net.bias` that
  are commented out in `TextQNet`, and `s.embeds` which is not a `TextMemory`
  field.

`legacy_scripts/` itself was deleted in the 2026-07 upstream cleanup.

## Code-Review Findings (2026-07-10, Not Yet Fixed)

Full review of the active code path; kept as documentation because core files
are intentionally left untouched for now.

**Metrics:**

- `BabilongExactMatch` (`prompts_and_metrics/babilong.py`): `prediction.split(",")`
  is never empty, so the final lowercase comparison is dead code — the actual
  comparison is a **case-sensitive** set match ("Balcony" != "balcony"), and
  `target` is not stripped. Deflates both eval EM and the `babilong_em` reward.
- `BabilongF1`: label matching is substring-based without word boundaries —
  e.g. `'one' in 'none'` is true, so for qa7 a prediction "one" against target
  "none" gets F1≈0.67 instead of 0.

**Feedback / reward path:**

- `GoldShiftFeedback.reset()` crashes with `TypeError` if the vLLM call fails:
  `_score_answer` returns `None` and the `logger.debug(f"β = {self.beta:.4f}")`
  f-string formats it unconditionally.
- `rl/feedback/llm_feedback.py` runs `torch.manual_seed(42)` and
  `logging.basicConfig` **at import time**, and `rl/feedback/__init__.py`
  imports it unconditionally — any `import rl.feedback` reseeds torch's global
  RNG and pulls in vllm.
- `LLMGenerator` API mode uses a 5s total timeout and returns `""` on any
  error (bare `except` in `_fetch_remote_api`) — under server load rewards
  silently become 0.
- All feedback classes use
  `asyncio.run_coroutine_threadsafe(coro, loop).result()` when a loop is
  running **in the same thread** — a guaranteed deadlock if ever called from
  async context (currently saved by the sync-only call path).
- vLLM `/tokenize` calls for candidates/answers do not pass
  `add_special_tokens: false` — fine for Qwen (no BOS), wrong for BOS models.
- `DummyFeedbackModel.FEEDBACK_MODEL_NAME: "dummy"` is a type annotation, not
  an assignment — the attribute does not exist.

**Train/eval skew:**

- `PQN.select_action` (single-sample path used by `evaluate()` and
  `eval_retriever.py`) stacks the state with the **action** tokenizer and
  `pqn.hyperparams.max_action_length_in_memory`, while training rollouts use
  the state tokenizer and the env's value. With `pqn_gte_oleg` both resolve to
  the same top-level value; the e5 configs hardcode `128` vs the top-level
  `220`, giving eval a different state truncation than training.
- `TextVNet` computes top-k over raw logits **without masking unavailable
  actions first** (the policy masks before top-k). Already-selected chunks can
  crowd out available ones from the top-k of the V-target; in the extreme the
  masked logsumexp underflows to `max - 23`.
- `alpha` is annealed proportionally to LR (`pqn.py`), so during warmup the
  exploration temperature starts near 0, grows to `alpha_start` by the end of
  warmup, then decays — inverted relative to a usual exploration schedule.

**Infrastructure:**

- `rl/optim.py::CosineScheduler` has no clamp: for `step > total` the LR goes
  **negative** (gradient ascent); `total == warmup` raises ZeroDivisionError.
  `WarmupLinearScheduler` jumps from `1-ratio` to `1.0` at the end of warmup.
- `PositionalRotaryEmbedding.forward` indexes the cached integer-position
  frequency table with `t.type(torch.int32)` — fractional positions (from
  `interpolate_factor` > 1 or `RelativePositionProcessor`'s linspace) are
  floor-truncated, collapsing the intended interpolation into coarse buckets;
  negative positions silently wrap around.
- `RandomPositionProcessor.get_randomized_idx` crashes for documents with
  fewer than 6 chunks (`np.random.choice(idx[1:n-1], size=4, replace=False)`);
  only `CandidateDatasetAdapter` enforces `min_chunks=6`.
- `envs/chunker.py::chunks_split` emits an empty first chunk when the first
  sentence exceeds `chunk_size`.
- `QAEnv.device` reads non-existent `self.embedder`; stray
  `from nltk.probability import gt_demo` imports in `envs/qa_env.py` and
  `eval_llm_openqa.py`.
- Dataset constructors call `np.random.seed(seed)` (global RNG pollution);
  reproducibility depends on construction order.
- `stack_memory` hardcodes the `"[SEP]"` split while the env separator is
  configurable, and its CLS/SEP merge assumes BERT-style special tokens
  (breaks with the Qwen3 embedder config); the merged state length is not
  re-truncated after joining pieces.

## `RetrievalBabiLong.create()` Is Older Than Hydra Configs

`envs/dataloaders/babilong/retrieval_babilong.py` has a `create()` classmethod that calls `TaskDataset(path, task, split)`, but current `TaskDataset` constructor accepts `dataset_path, max_n_facts=None`.

Hydra configs instantiate `RetrievalBabiLong` directly, so normal configured training does not use this stale classmethod.

## `LocalSetMusique` Has a Suspicious Undefined Variable

In `envs/dataloaders/musique.py`, `LocalSetMusique._load_data()` appends `Task(... context_len, context, ...)` in the no-filtering path before `context_len` and `context` are defined.

The normal loader is `RetrievalMusique`, which overrides `_load_data()`, so this may not affect current training. Do not use `LocalSetMusique` directly without fixing.

## Two Tests Are Manual Sanity Checks (excluded from pytest)

`tests/test_candidate_feedback.py` and `tests/test_gold_shift.py` require a
running vLLM server and hard-coded local data paths; `tests/conftest.py`
excludes them from collection.

The rest of `tests/` (added in the feedback-stable merge) are clean offline
suites — `pytest tests/` runs them in a fresh environment: `test_hydra_configs`,
`test_math_utils`, `test_prompt_consistency`, `test_qicl_datasets`,
`test_train_clean_and_combined`, `test_vllm_clients`.

## `eval_llm_openqa.py` Uses Large Generation Budget

The default is `--max_tokens 4000` with `temperature=0.0`. The prompt asks for
a short answer, but the generation budget is large. This can waste time/memory
if the model does not stop quickly; pass a smaller `--max_tokens` explicitly.

## `feedback/defaults.yaml` API Port Differs From IG / Group IG Configs

`feedback/defaults.yaml` for `AnswerMetricFeedback` uses:

```text
http://localhost:10001/v1
```

Group IG (`candidate_beta`) and IG (`gold_shift`) configs use:

```text
http://localhost:8000
```

Make sure the correct vLLM server is running for the selected feedback mode.

## Config `_target_` for PQN Is Historical

Several algo configs contain:

```yaml
pqn:
  _target_: rl.pqn.PQN
```

But current training does:

```python
from rl.agents.pqn import PQN
agent = PQN(agent_config)
```

So this `_target_` is not the source of truth.

## Checkpoint Eval Config Merge Can Override More Than Expected

`eval_retriever.py` does:

```python
cfg = OmegaConf.merge(train_cfg, eval_cfg)
```

Then resolves. `testing.yaml` can override parts of the training config, and CLI overrides can override both.

If eval behaves unexpectedly, inspect:

```text
<pretrained_path>/config.yaml
Q-RAG-feedback/configs/testing.yaml
CLI overrides
```

## Data/Checkpoint Files Are Large Artifacts

The following are artifacts, not source code:

- `datasets/**`
- `output/**`
- `old/**`
- `Q-RAG-feedback/runs/**`
- `Q-RAG/runs/**`
- `.pt`, TensorBoard event files, generated eval JSON/JSONL

They matter for reproducing runs, but should not be treated as implementation files.

