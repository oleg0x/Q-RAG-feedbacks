#!/usr/bin/env bash
# Ран 2026-08-02-gte-refresh-q1024-steps2, собран exp.py из configs/gte_refresh_q1024_steps2.yaml.
# Файл пишется до запуска, поэтому ран воспроизводим даже если
# процесс был убит.
set -euo pipefail
cd /home/a.anokhin/Judge/full-wiki

CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/a.anokhin/venvs/gpu/bin/python fullwiki_qrag.py retrieve --index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-qrag-jul21-best-raw --qrag-repo /home/a.anokhin/Judge/Q-RAG-feedback --device cuda:0 --model-dtype float32 --state-source critic --input /home/a.anokhin/Judge/datasets/data_sources/hotpotqa/hotpot_dev_fullwiki_v1.json --output /home/a.anokhin/Judge/full-wiki/runs/2026-08-02-gte-refresh-q1024-steps2/retrieval.jsonl --batch-size 64 --top-k 100 --steps 2 --mode refresh --reranker none --log-candidates none --candidate-index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-gte --candidate-backend torch-cuda --dedupe-titles --trust-remote-code --no-auto-prepare --first-stage-query-length 1024

# judge: 32 шардов answer_judge_llms.py, затем склейка
# /home/a.anokhin/venvs/gpu/bin/python /home/a.anokhin/Judge/Q-RAG-feedback/answer_judge_llms.py --retriever-logfile <shard>.jsonl --output-file <shard>.json --base-url http://127.0.0.1:8010/v1 --answer-model Qwen3-4B --max-samples 100000 --max-tokens 1000 --judge-max-tokens 100

python exp.py run --config configs/gte_refresh_q1024_steps2.yaml --only score
