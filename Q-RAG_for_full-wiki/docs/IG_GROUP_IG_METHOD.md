# IG and Group IG Feedback

Этот документ фиксирует новую исследовательскую линию из изображений: robust credit assignment для retrieval/context selection через likelihood shift относительно default context. В нашей терминологии:

- `IG` - Information Gain reward для одного целевого ответа.
- `Group IG` - Information Gain reward для группы candidate answers, разделенных на judged-correct и judged-incorrect.

В коде старые имена пока остаются:

- `IG` соответствует `GoldShiftFeedback` / `feedback=gold_shift`.
- `Group IG` соответствует `CandidateBetaFeedback` / `feedback=candidate_beta`.
- stepwise `Group IG` соответствует `CandidateBetaPerStepFeedback` / `feedback=candidate_beta_per_step`.

## Motivation

Обычные rewards для frozen LLM pipelines часто слишком шумные:

- Exact Match brittle: правильный смысл может получить ноль из-за формата, артикля, пунктуации, alias или лишней harmless фразы.
- Token F1 мягче, но может награждать поверхностный overlap и не отличать действительно полезный retrieved context.
- LLM-as-judge на каждом шаге дорогой и добавляет еще один noisy evaluator в training loop.

Методика IG/Group IG меняет вопрос. Вместо "совпал ли итоговый текст ответа" она спрашивает:

```text
Насколько выбранный context c увеличил вероятность правильных ответов
и, для Group IG, уменьшил относительное преимущество неправильных ответов?
```

Это лучше подходит для credit assignment в retrieval: reward становится функцией того, какую информацию chunks добавляют к вопросу.

## Setup

Есть dataset:

```text
D = {(x_i, y_i)}_{i=1}^M
```

где:

- `x_i` - input/question;
- `y_i` - reference answer;
- `pi_phi` - обучаемый context manager / retriever;
- `c = pi_phi(x_i)` - context, выбранный retriever-ом;
- `p_theta` - frozen feedback LLM;
- `c0` - default context, обычно пустой или baseline prompt без retrieved chunks.

В Q-RAG роль `pi_phi` играет `PQN`, а `c` строится из выбранных chunks в `QAEnv`.

## IG: Single-Answer Information Gain

`IG` использует только один target answer, обычно gold answer `y_i`.

Precompute:

```text
beta_i(y_i) = log p_theta(y_i | x_i, c0)
```

Reward for learned context:

```text
r_IG(x_i, c) = log p_theta(y_i | x_i, c) - beta_i(y_i)
```

Интерпретация:

- если retrieved context действительно содержит нужную информацию, вероятность gold answer должна вырасти;
- если вопрос был уже "легким" без контекста, высокая prior likelihood вычитается через `beta`;
- reward измеряет добавленную информацию, а не абсолютную уверенность модели.

### Code Mapping

Current implementation:

```text
Q-RAG-feedback/rl/feedback/gold_shift_feedback.py
```

Class:

```python
GoldShiftFeedback
```

Config:

```text
Q-RAG-feedback/configs/feedback/gold_shift.yaml
```

Important methods:

- `GoldShiftFeedback.reset()` computes beta for gold answer with empty context
  (online, 1 vLLM call per episode — the old offline `compute_gold_beta.py`
  script was removed in the Group IG cleanup).
- `GoldShiftFeedback.reward()` scores gold answer with retrieved context and subtracts beta.

## Group IG: Candidate-Set Information Gain

`Group IG` generalizes IG from one answer to a judged candidate set.

For each question `x_i`, build candidate set:

```text
G_i = {g_i,1, ..., g_i,K} union {y_i}
```

Candidates `g_i,1:K` are sampled from frozen feedback model under default context:

```text
g_i,k ~ p_theta(. | x_i, c0)
```

Then a frozen judge `v_tau` marks candidates:

```text
v_tau(x_i, g, y_i) -> {0, 1}
```

Define:

```text
G_i+ = {g in G_i : v_tau(x_i, g, y_i) = 1}
G_i- = {g in G_i : v_tau(x_i, g, y_i) = 0}
```

Precompute for every candidate:

```text
beta_i(g) = log p_theta(g | x_i, c0)
```

Reward for learned context:

```text
r_GroupIG(x_i, c) =
  mean_{g in G_i+} (log p_theta(g | x_i, c) - beta_i(g))
  -
  mean_{g in G_i-} (log p_theta(g | x_i, c) - beta_i(g))
```

If `G_i+` or `G_i-` is empty, the corresponding term is treated as `0` in the current implementation.

### Why Group IG Is Useful

Group IG gives a richer and more robust signal than single-answer IG:

- Multiple correct aliases can be positive, not just the exact gold string.
- Incorrect plausible candidates become negative controls.
- The reward favors context that raises correct candidates more than wrong candidates.
- The beta subtraction removes the model's no-context prior preference for common or memorized answers.
- The training signal is semantically aligned through the judge but does not require judge calls inside every reward computation if judgements are precomputed.

This is especially relevant for QA tasks where there are several valid surface realizations or where a retrieved chunk can support a wrong but plausible answer.

## Stepwise Group IG

The final-step Group IG reward only says whether the whole selected context was useful.

The stepwise variant returns marginal contribution:

```text
score_GroupIG(c_t) - score_GroupIG(c_{t-1})
```

where `c_t` is the context after adding the newest chunk.

This gives denser credit assignment, but is more expensive because candidate likelihoods are recomputed at each step.

### Code Mapping

Current implementation:

```text
Q-RAG-feedback/rl/feedback/candidate_beta_per_step_feedback.py
```

Class:

```python
CandidateBetaPerStepFeedback
```

Config:

```text
Q-RAG-feedback/configs/feedback/candidate_beta_per_step.yaml
```

## Current Code Pipeline for Group IG

The pipeline lives in `Q-RAG-feedback/gig_pipeline/` (moved from the `~/Judge` root
on 2026-07-10) and shares helpers via `gig_common.py` (prompt template, vLLM
client, atomic JSONL, resume + parallel scaffolding). Four scripts, four
artifacts. Run them from `~/Judge` so `output/...` paths resolve, e.g.
`python Q-RAG-feedback/gig_pipeline/generate_candidates.py ...`.

### 1. Candidate generation

Script:

```text
generate_candidates.py
```

Two modes, selected via `--mode`:

**`sample` mode (default, canonical Group IG):**

```bash
python Q-RAG-feedback/gig_pipeline/generate_candidates.py --dataset {hotpot,musique,2wiki} \
    --split train --output output/<ds>/<ds>_candidates.jsonl
```

Samples `K=8` candidates per question from the frozen feedback LLM (Qwen3-4B by default)
under empty context using the shared QA chat template from `gig_common`. This matches the
method definition `g_i,k ~ p_theta(. | x_i, c0)`. For 2Wiki, `bridge_comparison` samples are intentionally excluded by the loader.

**`guided` mode (gold-conditioned synthetic candidates):**

```bash
python Q-RAG-feedback/gig_pipeline/generate_candidates.py --dataset {hotpot,musique,2wiki} --mode guided \
    --split train --output output/<ds>_guided/<ds>_candidates.jsonl
```

Two separate prompted calls per question, both conditioned on the gold answer: one asks for
`K_CORRECT=4` valid surface forms of the gold (aliases, paraphrases), one for `K_WRONG=4`
plausible-but-incorrect distractors of the same entity type. Defaults: `max_tokens=256`,
`temperature=0.8`. Output is parsed from a numbered list (`1. ... 2. ...`).

If the parse misses the requested count, the record falls back to padding/truncation
(correct slots padded with the gold answer, wrong slots padded with `""`) and is flagged
`guided_ok=False`. Empirically `guided_ok=True` on ~99.8% of HotpotQA train records.

Output fields (both modes):

```python
{
  "id": ...,
  "question": ...,
  "answer": ...,
  "candidates": [str] * K   # in guided: first K_CORRECT correct, then K_WRONG wrong
}
# guided mode also adds: "guided_ok": bool
```

Downstream scripts (`judge_candidates`, `compute_beta`, `prepare_dataset`) consume both
schemas without modification.

**Important caveat for `guided` mode:** these candidates are not samples from
`p_theta(. | x_i, c0)`, so `G+` and `G-` end up in different prior-likelihood regimes.
See [KNOWN_ISSUES.md → Guided-mode beta gap](KNOWN_ISSUES.md).

### 2. Candidate judging

Script:

```text
judge_candidates.py
```

Purpose:

- calls a vLLM chat endpoint;
- asks a strict YES/NO equivalence judge;
- writes `judgements`, one per candidate.

Output fields:

```python
{
  ...,
  "candidates": [...],
  "judgements": [0, 1, ...]
}
```

Important: failed judging returns `-1`. Before Group IG training, unresolved `-1` should be fixed or removed because current feedback code treats any non-`1` judgement as negative.

### 3. Candidate beta precompute (with gold-append)

Script:

```text
compute_beta.py
```

Usage:

```bash
python Q-RAG-feedback/gig_pipeline/compute_beta.py \
    --input  output/<ds>/<ds>_judged.jsonl \
    --output output/<ds>/<ds>_final.jsonl
```

For each sample it:

- normalizes every candidate via `gig_common.strip_candidate_prefix` so the
  string written to JSONL is exactly the one scored (handles
  `Final answer: X`, `Answer: X`, and embedded `X\nFinal answer: X`);
- appends the gold answer as a `K+1`-th candidate with `judgement=1` (default;
  disable with `--no-append-gold`);
- tokenizes the shared QA chat prompt with `context=""` and scores each
  candidate via `/v1/completions` with `prompt_logprobs`.

Output is the final candidate-beta file (no separate `_gold_beta` / `_final`
split anymore — gold and merge live inside this single step):

```python
{
  ...,
  "candidates": [normalized str] * (K+1),
  "judgements": [int] * (K+1),
  "betas":      [float] * (K+1)
}
```

### 4. Dataset preparation

Script:

```text
prepare_dataset.py
```

Usage:

```bash
python Q-RAG-feedback/gig_pipeline/prepare_dataset.py --dataset {hotpot,musique,2wiki} \
    --final  output/<ds>/<ds>_final.jsonl \
    --output output/<ds>/<ds>_candidate_train.jsonl
```

Merges the candidate-beta JSONL with the raw HotpotQA / MuSiQue / 2Wiki context, filters
samples where `all(j == 1)` (no `G_-`), and writes the training file consumed
by `CandidateDatasetAdapter`. HotpotQA and 2Wiki keep Hotpot-style `context` +
`supporting_facts`; MuSiQue converts paragraphs into that shape. For 2Wiki,
`bridge_comparison` samples are intentionally excluded. The script uses a single
dispatch table `CONFIGS`.
Replaces the older pair `prepare_candidate_dataset.py` + `prepare_musique_dataset.py`.

### 6. Training dataset adapter

File:

```text
Q-RAG-feedback/envs/candidate_dataset_adapter.py
```

Classes:

```python
CandidateDataset
CandidateDatasetAdapter
```

The adapter returns:

```python
{
  "id": sample["_id"],
  "question": question,
  "answer": sample["answer"],
  "chunks": chunk_texts,
  "sf_idx": sf_idx,
  "candidates": sample["candidates"],
  "judgements": sample["judgements"],
  "betas": sample["betas"],
}
```

### 7. QAEnv passes Group IG fields to feedback

File:

```text
Q-RAG-feedback/envs/qa_env.py
```

`QAEnv._init_from_sample()` stores:

```python
self.candidates = sample.get("candidates")
self.judgements = sample.get("judgements")
self.betas = sample.get("betas")
```

`QAEnv._make_obs_and_info()` passes them in `info`.

### 8. Group IG reward at training time

File:

```text
Q-RAG-feedback/rl/feedback/candidate_beta_feedback.py
```

Class:

```python
CandidateBetaFeedback
```

Code-level formula:

```python
shift = log_prob_with_retrieved_context - beta
reward = mean(shifts for positive candidates) - mean(shifts for negative candidates)
reward *= reward_scaling
```

Config:

```text
Q-RAG-feedback/configs/feedback/candidate_beta.yaml
```

Default:

```yaml
type: candidate_beta
candidate_beta:
  api_base_url: "http://localhost:8000"
  model_name: "Qwen/Qwen3-4B"
  only_at_final: True
  never_terminate: True
  reward_scaling: 0.02
```

Training example:

```bash
cd /home/a.anokhin/Judge/Q-RAG-feedback
python train_q_rag.py \
  envs=musique_candidate \
  feedback=candidate_beta \
  algo=pqn_e5_musique
```

## Current Method-Code Alignment Issues

These are important for the next research phase.

### 1. Beta prompt must match reward prompt (Group IG: FIXED)

The method requires:

```text
beta_i(g) = log p_theta(g | x_i, c0)
reward    = log p_theta(g | x_i, c)
```

with the same prompt template on both sides so that `reward − beta` is a clean
shift caused only by the retrieved context. Beta and reward used to live in
two different templates (question-only vs. `GIVEN PASSAGES:` + question), and
`CandidateDatasetAdapter` additionally stripped the trailing `?` only on the
env side. That has been unified for Group IG:

- `gig_common.QA_USER_TEMPLATE` is now the single `GIVEN PASSAGES:\n{context}\n\nQUESTION:\n{question}\n\nReturn only the short answer.` template;
- `compute_beta.py` calls it with `context=""`;
- `CandidateBetaFeedback.QA_PROMPT` was rewritten to match `gig_common` byte-for-byte;
- `CandidateDatasetAdapter` no longer strips the trailing `?`.

Result: at `c=""` reward returns exactly `beta`. The shift is now `log p(g|x,c) − log p(g|x,c0)` with no template / punctuation noise.

**Verified 2026-07-10:** `GoldShiftFeedback.QA_PROMPT` and
`CandidateBetaPerStepFeedback.QA_PROMPT` still have a **leading newline** that
`gig_common.QA_USER_TEMPLATE` / `CandidateBetaFeedback.QA_PROMPT` do not.

- For `gold_shift` (IG) this is internally consistent — beta is computed at
  reset with the same template — but the likelihood scale differs from the
  gig_common one, so do not mix its scores with precomputed betas.
- For `candidate_beta_per_step` the per-step reward is a delta of scores, so
  the precomputed betas cancel between steps and the template mismatch mostly
  washes out; absolute scores, however, are on a different scale than the
  stored betas. Align the templates before relying on absolute values.

### 2. Group IG requires both positive and negative candidates

`prepare_dataset.py` filters all-correct samples:

```python
if all(j == 1 for j in rec["judgements"]):
    continue
```

This is good for Group IG because `G_i-` empty makes the negative term zero. Also consider filtering all-wrong or unresolved-heavy samples, because Group IG without positives loses the positive signal.

### 3. Current code does not filter `-1` judgements in reward

`judge_candidates.py` can write `-1`. Current `CandidateBetaFeedback` treats it as negative. For clean experiments, normalize:

- retry judge;
- drop unresolved candidates;
- or add explicit `if j not in (0, 1): continue`.

### 4. Cost profile

Final-step Group IG with `K` candidates costs roughly:

```text
K tokenize/score calls per episode reward
```

Stepwise Group IG costs:

```text
K calls at reset + K calls per step
```

Possible optimizations for future work:

- batch candidates per sample more efficiently;
- cache prompt tokenization;
- cache scores for repeated contexts if any;
- reduce candidate count;
- use smaller feedback model for scoring.

### 5. Naming in code vs research naming

For now, use this mapping in papers/docs:

```text
candidate_beta            -> Group IG
candidate_beta_per_step   -> Stepwise Group IG
gold_shift                -> IG
betas                     -> default-context log-likelihood baselines
judgements                -> judge labels for candidate correctness
```

Do not rename code casually unless you also update Hydra configs, saved run configs, and old artifacts.

## Existing Group IG Datasets

### HotpotQA — `sample` mode

```text
output/hotpotqa/hotpot_candidate_train.jsonl
```

Canonical pipeline (`generate_candidates.py` with default `--mode sample`,
`judge_candidates.py` with the strict YES/NO judge prompt, `compute_beta.py`,
`prepare_dataset.py`).

### HotpotQA — `guided` mode (May 2026)

```text
output/hotpotqa_guided/hotpot_candidates.jsonl     # 90 447 records, guided_ok=99.79%
output/hotpotqa_guided/hotpot_judged_my_prompt.jsonl    # strict judge (old prompt)
output/hotpotqa_guided/hotpot_judged_prompt_oleg.jsonl  # semantic-eq judge (current)
output/hotpotqa_guided/hotpot_final.jsonl          # +betas, 9 cands incl. gold
output/hotpotqa_guided/hotpot_candidate_train.jsonl  # 90 282 (165 all-positive dropped)
```

Generation: `Qwen/Qwen3-4B` on `localhost:8000`, two prompted calls per question.
Judging: `Qwen/Qwen3-32B` with the semantic-equivalence prompt (`prompt_oleg`).

Judge-prompt comparison on this dataset (90 447 records, 8 generated candidates each):

```text
                   strict (my_prompt)   semantic-eq (prompt_oleg)
correct slots TPR  86.44%                91.14%   (+4.70pp)
              FN   13.56%                 8.86%   (−4.70pp)
wrong   slots TNR  97.57%                96.81%   (−0.76pp)
              FP    2.43%                 3.19%   (+0.76pp)
unresolved (−1)    0                     0
```

For each FP gained, ~6.6 FN are recovered → `hotpot_judged_prompt_oleg.jsonl`
is the recommended judging file for downstream training.

Beta-precompute (`hotpot_final.jsonl`) statistics:

```text
intent_correct (pos 0-3)   mean β = -29.28
intent_wrong   (pos 4-7)   mean β = -17.74      ← wrong cands have HIGHER prior
gold (pos 8)               mean β = -15.18
gold rank by β             top-1: 26.36%,  top-3: 56.99%,  mean rank 3.27/9
per-sample mean(G+) − mean(G−)  ≈ −6.62  (gap is inverted vs canonical Group IG)
records usable for Group IG (both G+ and G−):  99.82%
```

The inverted beta gap is expected for this mode — see KNOWN_ISSUES.md before
interpreting reward magnitudes.

## Suggested Experiment Checklist

For each dataset/run, record:

- feedback type: `IG`, `Group IG`, or `Stepwise Group IG`;
- feedback model `p_theta`;
- judge model `v_tau`;
- candidate generation model and prompt;
- number of generated candidates `K`;
- whether gold answer was appended;
- filtering policy for all-correct/all-wrong/unresolved samples;
- exact beta prompt template;
- exact reward scoring prompt template;
- reward scaling;
- final retrieval EM/F1;
- reader LLM EM/F1 or LLM-as-judge score.

This checklist matters because Group IG is very sensitive to prompt-template consistency and candidate/judge quality.

