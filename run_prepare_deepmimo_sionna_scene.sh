#!/usr/bin/env bash
set -euo pipefail

# ======================== 用户参数区 ========================
PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
SIONNA_BASE_ENV="${SIONNA_BASE_ENV:-/data/zhujun/conda_envs/envrecons-differt}"
# 默认使用项目专用环境；可在命令前覆盖 PYTHON_BIN。
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.sionna-venv/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/deepmimo_sionna_munich.yaml}"
# ===========================================================

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache/xdg"
export LD_LIBRARY_PATH="${SIONNA_BASE_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "找不到 Sionna Python：${PYTHON_BIN}" >&2
  echo "请先运行 ${PROJECT_ROOT}/setup_sionna_environment.sh" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Config: ${CONFIG_PATH}"
exec "${PYTHON_BIN}" -m time_bias_localization.cli prepare-sionna-scene \
  --config "${CONFIG_PATH}"
