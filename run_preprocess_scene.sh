#!/usr/bin/env bash
set -euo pipefail

# ======================== 用户参数区 ========================
PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
PYTHON_BIN="${PYTHON_BIN:-/home/zhujun/miniconda3/bin/python3}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/offline_demo.yaml}"
# ===========================================================

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m time_bias_localization.cli prepare-scene --config "${CONFIG_PATH}"

