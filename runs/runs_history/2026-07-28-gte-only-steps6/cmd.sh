#!/usr/bin/env bash
# Reconstructed by migrate_runs.py: this run predates exp.py, so the
# commands were rebuilt from the documentation and rewritten to use
# paths inside the run directory.
set -euo pipefail
cd /path/to/repo

CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /path/to/venv/bin/python fullwiki_qrag.py retrieve --index-dir /path/to/data/full-wiki/wiki18-qrag-jul21-best-raw --candidate-index-dir /path/to/data/full-wiki/wiki18-gte --candidate-backend torch-cuda --qrag-repo /path/to/Q-RAG-feedback --device cuda:0 --model-dtype float32 --state-source critic --input /path/to/data/hotpotqa/hotpot_dev_fullwiki_v1.json --output /path/to/repo/runs/2026-07-28-gte-only-steps6/retrieval.jsonl --batch-size 64 --top-k 100 --steps 6 --mode fixed --dedupe-titles --reranker none --log-candidates none --trust-remote-code --no-auto-prepare

# reader + judge:
# python exp.py run --config configs/gte_only_steps6.yaml --only judge
# inside: split -n l/32 → 32 × answer_judge_llms.py → jq -s add
