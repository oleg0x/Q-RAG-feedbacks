#!/usr/bin/env bash
# Reconstructed by migrate_runs.py: this run predates exp.py, so the
# commands were rebuilt from the documentation and rewritten to use
# paths inside the run directory.
set -euo pipefail
cd /path/to/repo

/path/to/venv/bin/python build_eval_variants.py --context oracle --input /path/to/repo/runs/2026-07-28-gte-only-steps6/retrieval.jsonl --output /path/to/repo/runs/2026-07-28-oracle-sf/retrieval.jsonl --oracle-source /path/to/data/hotpotqa/hotpot_dev_distractor_v1.json

# reader + judge:
# python exp.py run --config configs/oracle_sf.yaml --only judge
# inside: split -n l/32 → 32 × answer_judge_llms.py → jq -s add
