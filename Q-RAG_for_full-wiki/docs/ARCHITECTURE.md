# Architecture

## High-Level Flow

Q-RAG trains a retriever as an RL agent.

```text
dataset sample
  -> adapter returns question, answer, chunks, sf_idx
  -> QAEnv exposes state and actions
  -> PQN selects a chunk index
  -> QAEnv appends chunk to memory
  -> feedback model returns reward / termination
  -> ParallelTextEnv batches transitions
  -> PQN updates Q-network
```

After training:

```text
eval_retriever.py
  -> JSONL with pred_idx, pred_texts, sf_idx, q_values
  -> eval_llm_openqa.py / eval_llm_synthetics.py
  -> reader LLM answer EM/F1 or LLM-as-judge metrics
```

## Environment Layer

### `TextEnv`

File: `Q-RAG-feedback/envs/text_env.py`

`TextEnv` is the base retrieval environment. It does not know QA semantics. It manages:

- `all_texts`: available chunks;
- `question`: initial state text;
- `memory`: selected chunk ids and state text;
- `available_mask`: chunks that can still be selected;
- `positions`: numeric positions for positional embeddings.

At reset:

```text
state.text = question
available_ids = all chunk indices
positions = positions_processor.initialize_positions(num_chunks)
```

At step(action):

```text
selected chunk is appended to memory
available_mask[action] = false
state.text = question + separator + selected chunks
positions may be updated
```

`sort_by_index=True` means selected chunks are inserted into memory sorted by original chunk index. `False` preserves retrieval order.

### Position Processors

Defined in `envs/text_env.py`:

- `AbsolutePositionProcessor`: positions are `0, 1, 2, ...`.
- `RandomPositionProcessor`: preserves order but inserts random gaps up to `max_chunks_count`.
- `RelativePositionProcessor`: reassigns positions around already selected chunks; intended for relative positional coding.

The chosen processor is selected by `envs.positions_processor` in Hydra configs and controls `algo.action_embed_dict.${envs.positions_processor}`.

### `QAEnv`

File: `Q-RAG-feedback/envs/qa_env.py`

`QAEnv` wraps `TextEnv` for question answering. On reset it reads a sample with:

```python
{
  "id": ...,
  "question": ...,
  "answer": ...,
  "chunks": [...],
  "sf_idx": [...],
  # optional:
  "candidates": [...],
  "judgements": [...],
  "betas": [...]
}
```

It prepares feedback inputs:

```python
obs = {
  "question": question,
  "sample_id": id,
  "pred_idx": selected_indices,
  "pred_chunks": selected_chunk_texts,
}

info = {
  "sf_idx": support_indices,
  "sf_chunks": support_texts,
  "answer": answer,
  "candidates": candidates,
  "judgements": judgements,
  "betas": betas,
}
```

At each step:

1. Calls `TextEnv._step(action)`.
2. Marks episode truncated if no actions remain or `num_steps >= max_steps`.
3. Calls `feedback_model.get_feedback(obs, info, truncated)`.
4. Returns `(next_state, selected_action_item, reward, done)`.

### `ParallelTextEnv`

File: `Q-RAG-feedback/envs/parallel_env.py`

This wrapper runs several `TextEnv`/`QAEnv` instances in parallel and returns a `TrainBatch`:

```python
TrainBatch(
  state,
  action,
  reward,
  next_state,
  not_done,
  q_values,
)
```

It precomputes action embeddings for all chunks in each env, updates positional embeddings each step, calls the agent once over a padded batch, and auto-resets finished episodes.

## Agent Layer

### `PQN`

File: `Q-RAG-feedback/rl/agents/pqn.py`

`PQN` is the main agent. It owns:

- `critic`: `TextQNet(state_embed, action_embed)`;
- `policy`: `TextQNetPolicy`, a hard-updated copy of state embedder using current critic action scoring;
- `random_policy`: random valid chunk selector;
- `v_net_target`: target state embedder for soft value estimates;
- `action_embed_target`: target action embedder;
- optimizer and scheduler.

Selection:

```text
state text -> state embedding
available chunks -> action embeddings
policy scores dot(state, action)
invalid actions masked
eval mode returns argmax
train mode samples among top-k after softmax / alpha
```

Training update (rewritten in the 2026-07-10 upstream merge — the old
`update_old`/`_get_target` recursion had an off-by-one in the bootstrap and the
rollout dropped the last transitions):

1. Receives rollout tensors from `ParallelTextEnv`: `reward`/`not_done` of
   shape `[num_envs, num_steps]` and `q_values` of shape
   `[num_envs, num_steps + 1]` (the extra column is the bootstrap value for the
   state after the last transition).
2. Builds TD(lambda) targets with the standalone `compute_returns(rewards,
   values_next, not_done, gamma, lambda_coef)` — backward recursion
   `G_t = r_t + not_done_t * gamma * ((1 - lambda) * V(s_{t+1}) + lambda * G_{t+1})`.
3. Computes twin-Q MSE loss against targets.
4. Accumulates gradients for `accumulate_grads`.
5. Clips gradients and optimizer step.
6. Updates alpha proportional to LR.
7. Soft-updates target networks with `tau`.
8. Hard-updates policy state embedder from critic.

The code uses `torch.compile` for `policy_apply`.

### Q Modules

File: `Q-RAG-feedback/rl/q_module.py`

`TextQNet` embeds state and action separately, then computes two Q-values by splitting embedding dimension in half:

```text
q1 = dot(state[:D], action[:D])
q2 = dot(state[D:], action[D:])
```

Other modules:

- `TextQNetPolicy`: scores all available action embeddings, masks invalid actions, chooses top-k/sample/argmax.
- `TextRandomPolicy`: uniform random valid action.
- `TextVNet`: target soft value by top-k logsumexp over action scores.
- `ActionEmbedTarget`: target action embedder wrapper with soft updates.

### Embedders

File: `Q-RAG-feedback/rl/bert_predictor.py`

`BertPredictor`:

- takes a pretrained HF encoder;
- clones config and first `num_hidden_layers`;
- adds a linear projection head;
- mean-pools hidden states when `n_output == 1`;
- returns projected embedding divided by `10`.

`SimpleEmbedder`:

- loads arbitrary HF embedding model;
- uses last-token pooling;
- optionally normalizes embeddings;
- used by `configs/algo/pqn_qwen3.yaml`.

Action embed wrappers:

- `EmbedderNone`: no positional transform.
- `EmbedderWithAbsoluteEncoding`: applies RoPE once using absolute/random positions.
- `EmbedderWithRelativeEncoding`: keeps raw embedding and reapplies RoPE when positions change.

## Feedback Layer

Base file: `Q-RAG-feedback/rl/feedback/feedback.py`

All feedback models implement:

```python
reset(obs, info)
reward(obs, info, is_final)
get_feedback(obs, info, truncated)
copy()
```

### `GroundTruthFeedback`

Reward is based on retrieving all support fact indices from `info["sf_idx"]`.

If `penalize_extra_steps=True`, completion reward is scaled by:

```text
0.5 + 0.5 * len(sf_idx) / len(pred_idx)
```

So retrieving extra chunks reduces final reward.

### `AnswerMetricFeedback`

File: `rl/feedback/llm_feedback.py`

Uses `LLMGenerator` to generate an answer from selected chunks, then scores it with configured metric:

- `BabilongExactMatch`
- `BabilongF1`
- generic EM/F1 metrics can be wired similarly

`LLMGenerator` supports:

- local vLLM loaded in-process;
- OpenAI-compatible API, including local vLLM server.

### `CandidateBetaFeedback` / Group IG

File: `rl/feedback/candidate_beta_feedback.py`

Final-step `Group IG` reward:

```text
score(ctx) =
  mean_{g in G+}(log p(g | question, ctx) - beta(g))
  - mean_{g in G-}(log p(g | question, ctx) - beta(g))

reward = score(retrieved_context) * reward_scaling
```

It requires each sample to contain:

- `candidates`
- `judgements` where `1` means correct and `0` incorrect
- `betas`, no-context log-probs for the same candidates

It calls a vLLM server for tokenization and prompt logprobs.

### `CandidateBetaPerStepFeedback` / Stepwise Group IG

File: `rl/feedback/candidate_beta_per_step_feedback.py`

Same score as Group IG, but reward at each step is:

```text
score(context after adding chunk) - score(previous context)
```

At reset it computes baseline score for empty context.

### `GoldShiftFeedback` / IG

File: `rl/feedback/gold_shift_feedback.py`

Single-answer `IG` reward. At reset:

```text
beta = log p(gold_answer | question, empty_context)
```

At reward time:

```text
reward = (log p(gold_answer | question, retrieved_context) - beta) * reward_scaling
```

### `GoldShiftFeedbackMATH` (merged from feedback-stable)

File: `rl/feedback/gold_shift_feedback_math.py`, config `feedback=gold_shift_math`.

Subclass of `GoldShiftFeedback` for MATH-style tasks: tokenizes the scoring
prompt as a math chat (question + retrieved few-shot examples) instead of the
OpenQA context template. Same beta-subtraction reward and env-var
endpoint/model/key handling.

### `LlmAnswer` (merged from feedback-stable)

File: `rl/feedback/llm_answer.py`, config `feedback=llm`.

Terminal-only Q-ICL/reader reward via a remote OpenAI-compatible vLLM server
(`BasicVllmClient`, thinking disabled by config, proxy env ignored by default):

- OpenQA tasks (HotpotQA/MuSiQue/2Wiki/combined): retrieved chunks are passed
  as context passages;
- Q-ICL tasks (GSM8K/MATH/HellaSwag/MMLU-Pro/XL-Sum): retrieved chunks are
  parsed into few-shot examples;
- scoring: exact match -> LLM-as-judge fallback; XL-Sum uses ROUGE-L F1;
- `get_metrics()` exposes `pred`, `EM`, `F1`, `LLM-as-judge`, `mean_logprob`,
  `judgment` for logging.

## Dataset Layer

### Standard QA schema

`QADatasetAdapter` returns:

```python
{
  "id": sample_id,
  "question": question,
  "answer": answer,
  "chunks": chunk_texts,
  "sf_idx": support_chunk_indices,
}
```

Supported sources:

- HotpotQA
- MuSiQue
- BabiLong
- LongBench
- RulerQA
- combined datasets
- 2WikiMultihopQA
- Q-ICL loaders merged from feedback-stable: GSM8K, MATH, HellaSwag, MMLU-Pro,
  XL-Sum (`envs/dataloaders/{gsm8k,competition_math,hellaswag,mmlupro,xlsum}.py`
  on top of shared `envs/dataloaders/base.py`; `chunks` hold formatted few-shot
  examples, `sf_idx=[0]`)

Combined-dataset additions (`envs/combined_dataset.py`):

- `RetrievalCombinedHotpot2Wiki` - HotpotQA+2Wiki with stable `source` labels
  (`hotpotqa` / `2WikiMultihopQA`) used by the per-source fixed eval;
- `MinChunksDataset` - deterministic view filtered by minimum context chunks;
- `CombinedCandidateDataset` (`envs/candidate_dataset_adapter.py`) - candidate
  JSONL with mandatory validated `source` per record.

HotpotQA/2Wiki loaders additionally accept the additive `train-clean` split
for prepared q0_s1 JSONL files (direct file path or directory with
`hotpot_candidate_train_q0_s1.jsonl` / `2wiki_candidate_train_q0_s1.jsonl`).

### Candidate schema

`CandidateDatasetAdapter` returns the same fields plus:

```python
{
  "candidates": [...],
  "judgements": [...],
  "betas": [...],
}
```

It expects Hotpot-style `context`:

```python
[
  [title, [sentence1, sentence2, ...]],
  ...
]
```

For MuSiQue candidate training, `gig_pipeline/prepare_dataset.py --dataset musique` converts MuSiQue paragraphs into this Hotpot-style shape.

## Evaluation Layer

### Retriever Evaluation

File: `Q-RAG-feedback/eval_retriever.py`

Loads:

1. `configs/testing.yaml`;
2. `<pretrained_path>/config.yaml`;
3. CLI overrides.

Then instantiates `PQN`, loads `model_best.pt` or `model_last.pt`, runs `QAEnv` episodes, and writes JSONL entries:

```python
{
  "id": ...,
  "question": ...,
  "answer": ...,
  "sf_idx": [...],
  "pred_idx": [...],
  "q_values": [...],
  "sf_texts": [...],
  "pred_texts": [...],
  "return": ...,
  "text_len": ...,
  "f1": ...,
  "em": ...,
}
```

### Reader LLM Evaluation

- `eval_llm_openqa.py`: HotpotQA/MuSiQue style, uses `pred_texts`.
- `eval_llm_synthetics.py`: BabiLong/RULER style, supports chunk filters.
- `gig_pipeline/llm_as_judge_eval.py`: external judge over generated answers.

(`eval_llm_longbench.py` was removed in the 2026-07 upstream cleanup.)

## Chunk Filters

File: `Q-RAG-feedback/prompts_and_metrics/chunk_filtering.py`

Used before reader LLM answering:

- `none`: use retrieved chunks unchanged.
- `early_stop`: cut after the last retrieved support fact.
- `gt`: use only ground-truth support chunks.
- `no_noise`: remove retrieved non-support chunks.
- `qvalue`: stop when Q-value falls below threshold.
- `retrieval_step`: keep first N retrieval steps.
- `llm`: placeholder, not implemented.

