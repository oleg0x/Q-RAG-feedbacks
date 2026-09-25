# Manual Smoke Tests

These commands are intentionally **not** part of the automated test suite. They
run training, GPU inference, or requests against a live vLLM server. Run them
from the repository root:

~~~bash
cd /home/a.anokhin/Judge/Q-RAG-feedback
export PYTHON=/home/a.anokhin/venvs/gpu/bin/python3.11
~~~

Every command writes under a new `runs/smoke-*` directory. The `runs/`
tree is ignored by Git. Never reuse a smoke root.

## 1. Config and import preflight

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/compose_configs.py --all
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -c 'import envs, envs.dataloaders, rl.feedback, utils, vLLM_clients; print("imports OK")'
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m pytest -q tests/test_qicl_datasets.py tests/test_train_clean_and_combined.py tests/test_hydra_configs.py tests/test_prompt_consistency.py tests/test_math_utils.py tests/test_vllm_clients.py
~~~

Expected: all configs print `OK`, imports print `imports OK`, and pytest
passes. No model or dataset is loaded by the compose command.

## Common GPU setup

Choose a free training GPU explicitly. The visible device is mapped to
`cuda:0` inside the process.

~~~bash
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to a free training GPU}"
~~~

If a command fails, first record:

~~~bash
nvidia-smi
git status --short
~~~

## 2. Ground Truth training

Each run uses 32 train samples, at most 8 eval samples, two optimizer
iterations, and two eval episodes. Batch size must be at least the env's
`max_steps` so that a full episode terminates inside the first rollout
(otherwise the first `train r_sum` is NaN over an empty slice): HotPotQA and
2Wiki use `max_steps=2` -> `batch_size=2`; MuSiQue uses `max_steps=4` ->
`batch_size=4`.

### HotPotQA

~~~bash
SMOKE_ROOT="runs/smoke-gt-hotpot-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" Q_RAG_CONFIG_NAME=training_gte_combined.yaml "$PYTHON" train_q_rag.py envs=hotpotqa feedback=defaults feedback.type=gt envs.data_path=/home/a.anokhin/Judge/datasets/data_sources/hotpotqa +envs.train_dataset.length=32 +envs.test_dataset.length=8 device=cuda:0 steps_count=2 learning_start=1 eval_interval=1 eval_episodes=2 batch_size=2 accumulate_grads=1 envs_parallel=1 logger.log_dir="$SMOKE_ROOT/"
~~~

### 2WikiMultiHopQA

~~~bash
SMOKE_ROOT="runs/smoke-gt-2wiki-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" Q_RAG_CONFIG_NAME=training_gte_combined.yaml "$PYTHON" train_q_rag.py envs=2wiki feedback=defaults feedback.type=gt envs.data_path=/home/a.anokhin/Judge/datasets/data_sources/2WikiMultiHopQA/data_ids_april7 +envs.train_dataset.length=32 +envs.test_dataset.length=8 device=cuda:0 steps_count=2 learning_start=1 eval_interval=1 eval_episodes=2 batch_size=2 accumulate_grads=1 envs_parallel=1 logger.log_dir="$SMOKE_ROOT/"
~~~

### MuSiQue

~~~bash
SMOKE_ROOT="runs/smoke-gt-musique-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" Q_RAG_CONFIG_NAME=training_gte_combined.yaml "$PYTHON" train_q_rag.py envs=musique feedback=defaults feedback.type=gt envs.data_path=/home/a.anokhin/Judge/datasets/data_sources/musique +envs.train_dataset.length=32 +envs.test_dataset.length=8 device=cuda:0 steps_count=2 learning_start=1 eval_interval=1 eval_episodes=2 batch_size=4 accumulate_grads=1 envs_parallel=1 logger.log_dir="$SMOKE_ROOT/"
~~~

Expected for each run: one timestamped child directory containing
`config.yaml`, `model_last.pt`, `model_best.pt`, and `tb_logs/`.
Find it with:

~~~bash
find "$SMOKE_ROOT" -mindepth 1 -maxdepth 1 -type d -print
~~~

## 3. Candidate / Group IG training

Use an already deployed vLLM server. Set its root URL without a trailing
`/v1`; the clients also accept a URL that already ends in `/v1`.

~~~bash
: "${VLLM_BASE_URL:?Set VLLM_BASE_URL, for example the chosen local server root}"
: "${VLLM_API_KEY:?Set VLLM_API_KEY without writing it to a file}"
export VLLM_MODEL="${VLLM_MODEL:-Qwen3-4B}"
~~~

Optional connectivity diagnostic (this sends one tiny request):

~~~bash
"$PYTHON" -m vLLM_clients.manual_integration --base-url "$VLLM_BASE_URL" --api-key "$VLLM_API_KEY" --model "$VLLM_MODEL"
~~~

### HotPotQA candidate

~~~bash
SMOKE_ROOT="runs/smoke-group-ig-hotpot-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"
sed -n '1,32p' /home/a.anokhin/Judge/output/hotpotqa/hotpot_candidate_train_q0_s1.jsonl > "$SMOKE_ROOT/train.jsonl"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" VLLM_BASE_URL="$VLLM_BASE_URL" VLLM_API_KEY="$VLLM_API_KEY" VLLM_MODEL="$VLLM_MODEL" Q_RAG_CONFIG_NAME=training.yaml "$PYTHON" train_q_rag.py envs=hotpotqa_candidate feedback=candidate_beta algo=pqn_gte envs.data_path="$PWD/$SMOKE_ROOT/train.jsonl" envs.test_data_path=/home/a.anokhin/Judge/datasets/data_sources/hotpotqa feedback.candidate_beta.max_concurrent=1 device=cuda:0 steps_count=2 learning_start=1 eval_interval=1 eval_episodes=2 batch_size=2 accumulate_grads=1 envs_parallel=1 logger.log_dir="$SMOKE_ROOT/output/"
~~~

### Combined HotPotQA + 2Wiki q0_s1

~~~bash
SMOKE_ROOT="runs/smoke-group-ig-combined-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"
rg -m 16 '"source": "hotpotqa"' /home/a.anokhin/Judge/output/combined/hotpotqa_2wiki_candidate_train_q0_s1.jsonl > "$SMOKE_ROOT/train.jsonl"
rg -m 16 '"source": "2WikiMultihopQA"' /home/a.anokhin/Judge/output/combined/hotpotqa_2wiki_candidate_train_q0_s1.jsonl >> "$SMOKE_ROOT/train.jsonl"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" VLLM_BASE_URL="$VLLM_BASE_URL" VLLM_API_KEY="$VLLM_API_KEY" VLLM_MODEL="$VLLM_MODEL" Q_RAG_CONFIG_NAME=training_hotpotqa_2wiki_candidate_llm.yaml "$PYTHON" train_q_rag.py feedback=candidate_beta envs.train_data_path="$PWD/$SMOKE_ROOT/train.jsonl" envs.eval_samples_per_dataset=2 feedback.candidate_beta.max_concurrent=1 device=cuda:0 steps_count=1 learning_start=1 eval_interval=1 eval_episodes=4 batch_size=6 accumulate_grads=1 envs_parallel=1 logger.log_dir="$SMOKE_ROOT/output/"
~~~

Expected: the same checkpoint/config files as Ground Truth runs. The combined
run evaluates two fixed samples from each source and emits per-source
TensorBoard metrics.

## 4. Q-ICL / LlmAnswer feedback on available data

This uses the combined available HotPotQA+2Wiki file; it does not require
GSM8K, MATH, HellaSwag, MMLU-Pro, or XL-Sum.

~~~bash
SMOKE_ROOT="runs/smoke-qicl-combined-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT"
rg -m 8 '"source": "hotpotqa"' /home/a.anokhin/Judge/output/combined/hotpotqa_2wiki_candidate_train_q0_s1.jsonl > "$SMOKE_ROOT/train.jsonl"
rg -m 8 '"source": "2WikiMultihopQA"' /home/a.anokhin/Judge/output/combined/hotpotqa_2wiki_candidate_train_q0_s1.jsonl >> "$SMOKE_ROOT/train.jsonl"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" VLLM_BASE_URL="$VLLM_BASE_URL" VLLM_API_KEY="$VLLM_API_KEY" VLLM_MODEL="$VLLM_MODEL" Q_RAG_CONFIG_NAME=training_hotpotqa_2wiki_candidate_llm.yaml "$PYTHON" train_q_rag.py envs.train_data_path="$PWD/$SMOKE_ROOT/train.jsonl" envs.eval_samples_per_dataset=2 feedback.llm_answer.max_tokens=32 device=cuda:0 steps_count=1 learning_start=1 eval_interval=1 eval_episodes=4 batch_size=6 accumulate_grads=1 envs_parallel=1 logger.log_dir="$SMOKE_ROOT/output/"
~~~

Expected: checkpoints plus `eval/<source>/...` TensorBoard series. A server
or authentication error should appear as an `LlmAnswer reward failed` log.

Note: the runs recorded on 2026-07-21 (`runs/smoke-qicl-hotpot-*`) used a
lighter ad-hoc variant of this smoke — `feedback=llm` (`LlmAnswer`,
`max_tokens=32`) on the plain HotPotQA env with 32/8 samples, `steps_count=1`,
`batch_size=2` (see the saved `config.yaml` in those run dirs) — not the
combined command above. The combined variant is still the canonical smoke for
the per-source eval series.

## 5. Retriever evaluation

Point `TRAIN_RUN` at the timestamped directory produced by one Ground Truth
run (not at its parent `SMOKE_ROOT`).

~~~bash
: "${TRAIN_RUN:?Set TRAIN_RUN to a completed smoke training directory}"
test -f "$TRAIN_RUN/config.yaml"
test -f "$TRAIN_RUN/model_best.pt"
EVAL_ROOT="runs/smoke-retriever-eval-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$EVAL_ROOT"
mkdir -p "$EVAL_ROOT"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON" eval_retriever.py pretrained_path="$TRAIN_RUN" num_samples=2 use_last=false +device=cuda:0 +output_file_path="$PWD/$EVAL_ROOT/retriever.jsonl"
~~~

Expected: `$EVAL_ROOT/retriever.jsonl` with two JSONL records containing
`pred_idx`, `pred_texts`, `sf_idx`, and Q-values.

## 6. Reader LLM evaluation through the deployed server

~~~bash
: "${RETRIEVER_LOG:?Set RETRIEVER_LOG to the retriever.jsonl from step 5}"
READER_ROOT="runs/smoke-reader-eval-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$READER_ROOT"
mkdir -p "$READER_ROOT"
"$PYTHON" answer_judge_llms.py --retriever-logfile "$RETRIEVER_LOG" --output-file "$READER_ROOT/reader.json" --base-url "$VLLM_BASE_URL" --api-key "$VLLM_API_KEY" --answer-model "$VLLM_MODEL" --max-samples 2 --max-tokens 64 --skip-judge
~~~

Expected: `reader.json` with prediction, EM, and F1 for two samples.

## 7. Optional LLM-as-judge

The command below repeats the two reader generations and adds judge calls.

~~~bash
JUDGE_ROOT="runs/smoke-llm-judge-$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$JUDGE_ROOT"
mkdir -p "$JUDGE_ROOT"
"$PYTHON" answer_judge_llms.py --retriever-logfile "$RETRIEVER_LOG" --output-file "$JUDGE_ROOT/judged.json" --base-url "$VLLM_BASE_URL" --api-key "$VLLM_API_KEY" --answer-model "$VLLM_MODEL" --max-samples 2 --max-tokens 64
~~~

Expected: `judged.json` additionally contains `LLM_Judge_Score`.

## Diagnostics after a failure

Config failure:

~~~bash
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" scripts/compose_configs.py training_hotpotqa_2wiki_candidate_llm --show
~~~

Training failure:

~~~bash
find "$SMOKE_ROOT" -maxdepth 3 -type f -printf '%p\n' | sort
find "$SMOKE_ROOT" -name config.yaml -print -exec sed -n '1,220p' {} \;
~~~

vLLM failure:

~~~bash
"$PYTHON" -m vLLM_clients.manual_integration --base-url "$VLLM_BASE_URL" --api-key "$VLLM_API_KEY" --model "$VLLM_MODEL"
~~~

Do not paste the value of `VLLM_API_KEY` into logs or issue reports.
