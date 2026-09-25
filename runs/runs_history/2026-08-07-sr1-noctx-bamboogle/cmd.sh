#!/usr/bin/env bash
# Ран 2026-08-07-sr1-noctx-bamboogle, собран exp.py из configs/sr1_bamboogle_noctx.yaml.
# Файл пишется до запуска, поэтому ран воспроизводим даже если
# процесс был убит.
set -euo pipefail
cd /home/a.anokhin/Judge/full-wiki

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/a.anokhin/venvs/gpu/bin/python build_eval_variants.py --context none --input /home/a.anokhin/Judge/full-wiki/runs/2026-08-07-sr1-best-bamboogle/retrieval.jsonl --output /home/a.anokhin/Judge/full-wiki/runs/2026-08-07-sr1-noctx-bamboogle/retrieval.jsonl

# judge: 32 шардов answer_judge_llms.py, затем склейка
# /home/a.anokhin/venvs/gpu/bin/python /home/a.anokhin/Judge/Q-RAG-feedback/answer_judge_llms.py --retriever-logfile <shard>.jsonl --output-file <shard>.json --base-url http://127.0.0.1:8010/v1 --answer-model Qwen3-4B --max-samples 100000 --max-tokens 1000 --judge-max-tokens 100

python exp.py run --config configs/sr1_bamboogle_noctx.yaml --only score
