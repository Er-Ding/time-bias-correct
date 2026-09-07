#!/usr/bin/env bash
set -euo pipefail

# ======================== 用户参数区 ========================
PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
SIONNA_BASE_ENV="${SIONNA_BASE_ENV:-/data/zhujun/conda_envs/envrecons-differt}"
BASE_PYTHON="${BASE_PYTHON:-${SIONNA_BASE_ENV}/bin/python}"
ENV_DIR="${ENV_DIR:-${PROJECT_ROOT}/.sionna-venv}"
DEEPMIMO_VERSION="${DEEPMIMO_VERSION:-4.0.5}"
PYYAML_VERSION="${PYYAML_VERSION:-6.0.3}"
# ===========================================================

if [[ ! -x "${BASE_PYTHON}" ]]; then
  echo "找不到基础 Sionna Python：${BASE_PYTHON}" >&2
  exit 2
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${BASE_PYTHON}" -m venv --system-site-packages "${ENV_DIR}"
fi

"${ENV_DIR}/bin/python" -m pip install \
  --no-build-isolation --no-deps --editable "${PROJECT_ROOT}"
"${ENV_DIR}/bin/python" -m pip install \
  --no-deps "DeepMIMO==${DEEPMIMO_VERSION}" "PyYAML==${PYYAML_VERSION}"

echo "环境已就绪：${ENV_DIR}/bin/python"
