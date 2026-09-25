# GIG (Group IG / candidate-beta) dataset pipeline

Scripts that build the training data consumed by `envs.CandidateDatasetAdapter`
and the `rl.feedback.CandidateBetaFeedback` reward. All of them talk to a vLLM
server (`http://localhost:8000` by default, override auth via the
`VLLM_API_KEY` env var) and share prompts/HTTP helpers from `gig_common.py`.

The QA prompt templates in `gig_common.py` must stay byte-identical to the ones
in `rl/feedback/candidate_beta_feedback.py` — betas and rewards must be computed
on the same likelihood scale.

## Main pipeline (per dataset)

```
1. generate_candidates.py   # K candidate answers per question (--mode sample|guided)
2. judge_candidates.py      # judge each candidate vs gold answer -> judgements 0/1
3. compute_beta.py          # beta(g) = log p(g | question, no context); appends gold answer
4. prepare_dataset.py       # merge with raw HotpotQA/MuSiQue/2Wiki context ->
                            #   candidate_train JSONL for CandidateDatasetAdapter
```

Optional post-processing:

- `combine_candidate_train_jsonl.py` — merge several candidate_train files into
  one (adds a `source` field per record).
- `run_success_rate_experiment.py` — question-only vs gold-facts success rates
  per question (used to pick "interesting" questions).
- `filter_success_rate_results.py` — filter the experiment output.
- `filter_candidate_train_by_success_ids.py` — keep only questions selected by
  the success-rate experiment (default: q0_s1 subset).

## Evaluation helpers

- `llm_as_judge_eval.py` — LLM-as-Judge accuracy over `eval_llm_openqa.py`
  output JSON.
- `make_2wiki_oracle_eval.py` — oracle retriever log for 2Wiki (pred = gold
  supporting facts), same format as `eval_retriever.py` logs.
