#!/usr/bin/env bash
# Ран 2026-08-02-pool-oracle-chunks-k2, собран exp.py из configs/pool_oracle_chunks_k2.yaml.
# Файл пишется до запуска, поэтому ран воспроизводим даже если
# процесс был убит.
set -euo pipefail
cd /home/a.anokhin/Judge/full-wiki

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/a.anokhin/venvs/gpu/bin/python candidate_pool.py select --input /home/a.anokhin/Judge/full-wiki/runs/2026-08-02-gte-pool-diag/retrieval.jsonl --index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-qrag-jul21-best-raw --steps 2 --output /home/a.anokhin/Judge/full-wiki/runs/2026-08-02-pool-oracle-chunks-k2/retrieval.jsonl --prefer gold-sentence --gold-source /home/a.anokhin/Judge/datasets/data_sources/hotpotqa/hotpot_dev_distractor_v1.json

# judge: 32 шардов answer_judge_llms.py, затем склейка
# /home/a.anokhin/venvs/gpu/bin/python /home/a.anokhin/Judge/Q-RAG-feedback/answer_judge_llms.py --retriever-logfile <shard>.jsonl --output-file <shard>.json --base-url http://127.0.0.1:8010/v1 --answer-model Qwen3-4B --max-samples 100000 --max-tokens 1000 --judge-max-tokens 100

python exp.py run --config configs/pool_oracle_chunks_k2.yaml --only score
