#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${SCRIPT_DIR}/run_vllm.sh" Qwen3-1.7B 9314 vllm_config_H200_Qwen3-1.7B.yaml
