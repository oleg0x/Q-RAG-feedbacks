#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/run_vllm.sh" Qwen3-8B 9315 vllm_config_H200_Qwen3-8B.yaml
