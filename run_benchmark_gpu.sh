#!/usr/bin/env bash
set -euo pipefail

# ============================ 用户参数区 ============================
PROJECT_ROOT="${PROJECT_ROOT:-/data/zhujun/differt_projects/time-bias-correct}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.sionna-venv/bin/python}"
CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_ROOT:-/usr/local/cuda-12.2}"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_ROOT}/outputs/gpu_munich_20260908T020052_1904235/UE001/repeat_000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/gpu_benchmark_$(date -u +%Y%m%dT%H%M%S)_$$}"
# full：完整的谱面采样、反向候选、聚类与联合求解；kernel：一次分解、全局谱与局部连续采样。
# 两种模式均不对观测 CSI 额外加噪；计时排除设备预热。
MODE="${MODE:-full}"
# GPU_ID 是宿主机 GPU 编号。进程只看见该卡，故 Python 中 device_id 固定为 0。
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ANGLE_CHUNK_SIZE="${ANGLE_CHUNK_SIZE:-32}"
BLAS_THREADS="${BLAS_THREADS:-1}"
# 1：CPU/GPU 都定位完成后才独立读取真值评估；0：只比较输出，不读取真值内容。
EVALUATE="${EVALUATE:-0}"
# 定位专用配置：重放历史 CSI 时也使用当前谱面采样配置，不加载原目录中的旧扰动配置。
LOCALIZATION_CONFIG="${LOCALIZATION_CONFIG:-${PROJECT_ROOT}/configs/deepmimo_sionna_munich_localization.yaml}"
# ===================================================================

if [[ ! "${GPU_ID}" =~ ^[0-9]+$ || ! "${EVALUATE}" =~ ^[01]$ ]]; then
  echo "GPU_ID 必须为非负整数；EVALUATE 只能为 0 或 1" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_PATH="${CUDA_TOOLKIT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS}"
export MKL_NUM_THREADS="${BLAS_THREADS}"
export OMP_NUM_THREADS="${BLAS_THREADS}"
export NUMEXPR_NUM_THREADS="${BLAS_THREADS}"
export BLIS_NUM_THREADS="${BLAS_THREADS}"
cd "${PROJECT_ROOT}"
RUN_ARGS=(--input-root "${INPUT_ROOT}" --output-root "${OUTPUT_ROOT}" --mode "${MODE}"
  --device-id 0 --batch-size "${BATCH_SIZE}" --angle-chunk-size "${ANGLE_CHUNK_SIZE}"
  --blas-threads "${BLAS_THREADS}")
if [[ -n "${LOCALIZATION_CONFIG}" ]]; then
  RUN_ARGS+=(--config "${LOCALIZATION_CONFIG}")
fi
if [[ "${EVALUATE}" == "1" ]]; then
  RUN_ARGS+=(--evaluate)
fi
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/benchmark_gpu_localization.py" "${RUN_ARGS[@]}"
