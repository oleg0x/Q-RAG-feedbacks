#!/usr/bin/env bash
# Ран 2026-08-02-gte-pool-diag-k1000, собран exp.py из configs/gte_pool_diag_k1000.yaml.
# Файл пишется до запуска, поэтому ран воспроизводим даже если
# процесс был убит.
set -euo pipefail
cd /home/a.anokhin/Judge/full-wiki

CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/a.anokhin/venvs/gpu/bin/python fullwiki_qrag.py retrieve --index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-qrag-jul21-best-raw --qrag-repo /home/a.anokhin/Judge/Q-RAG-feedback --device cuda:0 --model-dtype float32 --state-source critic --input /home/a.anokhin/Judge/datasets/data_sources/hotpotqa/hotpot_dev_fullwiki_v1.json --output /home/a.anokhin/Judge/full-wiki/runs/2026-08-02-gte-pool-diag-k1000/retrieval.jsonl --batch-size 64 --top-k 1000 --steps 1 --mode fixed --reranker none --log-candidates full --candidate-index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-gte --candidate-backend torch-cuda --dedupe-titles --trust-remote-code --no-auto-prepare

# judge: 32 шардов answer_judge_llms.py, затем склейка
# /home/a.anokhin/venvs/gpu/bin/python /home/a.anokhin/Judge/Q-RAG-feedback/answer_judge_llms.py --retriever-logfile <shard>.jsonl --output-file <shard>.json --base-url http://127.0.0.1:8010/v1 --answer-model Qwen3-4B --max-samples 100000 --max-tokens 1000 --judge-max-tokens 100

python exp.py run --config configs/gte_pool_diag_k1000.yaml --only score
