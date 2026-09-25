#!/usr/bin/env bash
# Восстановлено migrate_runs.py: этот ран сделан до появления exp.py,
# команды реконструированы из docs/pipeline.md и приведены к путям
# внутри каталога рана.
set -euo pipefail
cd /home/a.anokhin/Judge/full-wiki

CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/a.anokhin/venvs/gpu/bin/python fullwiki_qrag.py retrieve --index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-qrag-jul21-best-raw --candidate-index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-gte --candidate-backend torch-cuda --qrag-repo /home/a.anokhin/Judge/Q-RAG-feedback --device cuda:0 --model-dtype float32 --state-source critic --input /home/a.anokhin/Judge/datasets/data_sources/hotpotqa/hotpot_dev_fullwiki_v1.json --output /home/a.anokhin/Judge/full-wiki/runs/2026-07-28-qrag-fixed-steps4/retrieval.jsonl --batch-size 64 --top-k 100 --steps 4 --mode fixed --dedupe-titles --reranker qrag --log-candidates full --trust-remote-code --no-auto-prepare

# reader + judge:
# python exp.py run --config configs/qrag_steps4.yaml --only judge
# внутри: split -n l/32 → 32 × answer_judge_llms.py → jq -s add
# подробности схемы: docs/pipeline.md §6
