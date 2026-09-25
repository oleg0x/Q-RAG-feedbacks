#!/usr/bin/env bash
# Ран 2026-08-06-lineA-last-hotpotqa, собран exp.py из configs/search_last_hotpotqa.yaml.
# Файл пишется до запуска, поэтому ран воспроизводим даже если
# процесс был убит.
set -euo pipefail
cd /home/a.anokhin/Judge/full-wiki

CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/a.anokhin/venvs/gpu/bin/python fullwiki_qrag.py search --index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-gte --qrag-repo /home/a.anokhin/Judge/full-wiki/Q-RAG_for_full-wiki --device cuda:0 --model-dtype float32 --input /home/a.anokhin/Judge/datasets/data_sources/hotpotqa/hotpot_dev_fullwiki_v1.json --output /home/a.anokhin/Judge/full-wiki/runs/2026-08-06-lineA-last-hotpotqa/retrieval.jsonl --batch-size 64 --top-k 100 --steps 6 --max-chunks-per-title 2 --log-candidates full --checkpoint /home/a.anokhin/Judge/full-wiki/Q-RAG_for_full-wiki/runs/Aug03_12-44-38_lineA_main/model_last.pt --state-source critic --no-auto-prepare

# judge: 32 шардов answer_judge_llms.py, затем склейка
# /home/a.anokhin/venvs/gpu/bin/python /home/a.anokhin/Judge/Q-RAG-feedback/answer_judge_llms.py --retriever-logfile <shard>.jsonl --output-file <shard>.json --base-url http://127.0.0.1:8010/v1 --answer-model Qwen3-4B --max-samples 100000 --max-tokens 1000 --judge-max-tokens 100

python exp.py run --config configs/search_last_hotpotqa.yaml --only score
