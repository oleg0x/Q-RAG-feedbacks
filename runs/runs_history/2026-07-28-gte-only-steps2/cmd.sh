#!/usr/bin/env bash
# Восстановлено migrate_runs.py: этот ран сделан до появления exp.py,
# команды реконструированы из docs/pipeline.md и приведены к путям
# внутри каталога рана.
set -euo pipefail
cd /home/a.anokhin/Judge/full-wiki

/home/a.anokhin/venvs/gpu/bin/python build_eval_variants.py --context truncate --input /home/a.anokhin/Judge/full-wiki/runs/2026-07-28-gte-only-steps6/retrieval.jsonl --output /home/a.anokhin/Judge/full-wiki/runs/2026-07-28-gte-only-steps2/retrieval.jsonl --steps 2

# reader + judge:
# python exp.py run --config configs/gte_only_steps2.yaml --only judge
# внутри: split -n l/32 → 32 × answer_judge_llms.py → jq -s add
# подробности схемы: docs/pipeline.md §6
