#!/usr/bin/env bash
set -euo pipefail

# ============================ 用户参数区 ============================
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.sionna-venv/bin/python}"
# 输入：已保存接收 CSI 的原实验；不会重新生成信道或加噪。
SOURCE_EXPERIMENT="${SOURCE_EXPERIMENT:-${PROJECT_ROOT}/outputs/gpu_munich_20260908T020052_1904235}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/deepmimo_sionna_munich_localization.yaml}"
UE_IDS="${UE_IDS:-UE001,UE002,UE006}"
NOISE_REPEATS="${NOISE_REPEATS:-1}"
SAMPLES_PER_PEAK="${SAMPLES_PER_PEAK:-128}"
COMPUTE_BACKEND="${COMPUTE_BACKEND:-cuda}"
DEVICE_ID="${DEVICE_ID:-0}"
CPU_THREADS="${CPU_THREADS:-1}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.2}"
SIONNA_BASE_ENV="${SIONNA_BASE_ENV:-/data/zhujun/conda_envs/envrecons-differt}"
# 输出：独立新目录，已存在时拒绝覆盖；内部含逐 sample 报告与总结。
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/spectrum_check_$(date -u +%Y%m%dT%H%M%S)_$$}"
# ===================================================================

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS}" OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export CUPY_CACHE_DIR="${PROJECT_ROOT}/.cache/cupy"
export CUDA_PATH
export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${SIONNA_BASE_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" scripts/check_spectrum_workflow.py \
  --source-experiment "${SOURCE_EXPERIMENT}" --config "${CONFIG_PATH}" \
  --output "${OUTPUT_ROOT}" --ue-ids "${UE_IDS}" --noise-repeats "${NOISE_REPEATS}" \
  --samples-per-peak "${SAMPLES_PER_PEAK}" --backend "${COMPUTE_BACKEND}" --device-id "${DEVICE_ID}"
