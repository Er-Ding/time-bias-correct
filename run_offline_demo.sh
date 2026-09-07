#!/usr/bin/env bash
set -euo pipefail

# ======================== 用户参数区 ========================
PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
PYTHON_BIN="${PYTHON_BIN:-/home/zhujun/miniconda3/bin/python3}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/offline_demo.yaml}"
# ===========================================================

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

cd "${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Config: ${CONFIG_PATH}"
exec "${PYTHON_BIN}" -m time_bias_localization.cli offline-demo --config "${CONFIG_PATH}"

