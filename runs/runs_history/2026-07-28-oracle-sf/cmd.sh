#!/usr/bin/env bash
# Восстановлено migrate_runs.py: этот ран сделан до появления exp.py,
# команды реконструированы из docs/pipeline.md и приведены к путям
# внутри каталога рана.
set -euo pipefail
cd /home/a.anokhin/Judge/full-wiki

/home/a.anokhin/venvs/gpu/bin/python build_eval_variants.py --context oracle --input /home/a.anokhin/Judge/full-wiki/runs/2026-07-28-gte-only-steps6/retrieval.jsonl --output /home/a.anokhin/Judge/full-wiki/runs/2026-07-28-oracle-sf/retrieval.jsonl --oracle-source /home/a.anokhin/Judge/datasets/data_sources/hotpotqa/hotpot_dev_distractor_v1.json

# reader + judge:
# python exp.py run --config configs/oracle_sf.yaml --only judge
# внутри: split -n l/32 → 32 × answer_judge_llms.py → jq -s add
# подробности схемы: docs/pipeline.md §6
