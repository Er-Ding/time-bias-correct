#!/usr/bin/env bash
set -euo pipefail

# ============================ 用户参数区 ============================
PROJECT_ROOT="${PROJECT_ROOT:-/data/zhujun/differt_projects/time-bias-correct}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.sionna-venv/bin/python}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.2}"
CUPY_VERSION="${CUPY_VERSION:-13.6.0}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PROJECT_ROOT}/.cache/pip}"
# ===================================================================

export CUDA_PATH
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" -m pip install --disable-pip-version-check \
  --cache-dir "${PIP_CACHE_DIR}" "cupy-cuda12x==${CUPY_VERSION}"
"${PYTHON_BIN}" -c 'import cupy as cp; print("CuPy:", cp.__version__); print("可见 GPU 数:", cp.cuda.runtime.getDeviceCount()); print("GPU 计算检查:", (cp.arange(4, dtype=cp.float64)**2).get())'
