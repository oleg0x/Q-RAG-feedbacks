#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 SERVED_MODEL_NAME DEFAULT_PORT CONFIG_FILE" >&2
    exit 2
fi

SERVED_MODEL_NAME=$1
DEFAULT_PORT=$2
CONFIG_FILE=$3
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${VLLM_MODEL_DIR:?Set VLLM_MODEL_DIR to the host model directory}"
: "${VLLM_GPU:?Set VLLM_GPU to the GPU index (or Docker device selector)}"
: "${VLLM_API_KEY:?Set VLLM_API_KEY; do not place credentials in this script}"

VLLM_PORT=${VLLM_PORT:-$DEFAULT_PORT}
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.12.0}
VLLM_CONTAINER_NAME=${VLLM_CONTAINER_NAME:-vllm-$SERVED_MODEL_NAME}
HF_CACHE_DIR=${HF_CACHE_DIR:-${HOME}/.cache/huggingface}
VLLM_RESTART_POLICY=${VLLM_RESTART_POLICY:-unless-stopped}

docker run \
    --gpus "device=${VLLM_GPU}" \
    -p "${VLLM_PORT}:${VLLM_PORT}" \
    --ipc=host \
    --name "${VLLM_CONTAINER_NAME}" \
    --restart "${VLLM_RESTART_POLICY}" \
    -v "${SCRIPT_DIR}:/config:ro" \
    -v "${VLLM_MODEL_DIR}:/models:ro" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -e HF_HOME=/root/.cache/huggingface \
    "${VLLM_IMAGE}" \
    --config "/config/${CONFIG_FILE}" \
    --model /models \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --port "${VLLM_PORT}" \
    --api-key "${VLLM_API_KEY}"
