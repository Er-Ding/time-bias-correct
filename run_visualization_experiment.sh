#!/usr/bin/env bash
set -euo pipefail

# ============================ 用户参数区 ============================
PROJECT_ROOT="${PROJECT_ROOT:-/data/zhujun/differt_projects/time-bias-correct}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.sionna-venv/bin/python}"
SIONNA_BASE_ENV="${SIONNA_BASE_ENV:-/data/zhujun/conda_envs/envrecons-differt}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${PROJECT_ROOT}/configs/visualization_experiment_munich.yaml}"
# BS、噪声、偏差、MUSIC 参数在上面配置所引用的 generation_config 中修改。
# UE 数、重复数、采样范围、随机种子在 EXPERIMENT_CONFIG 中修改。
# DeepMIMO 内部场景名自动转为小写；EXPERIMENT_ROOT 可保留大小写。
# 已失败的旧计划不会因 RESUME=1 自动重跑；修复后请用新的 EXPERIMENT_ROOT。
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PROJECT_ROOT}/outputs/munich_30ue_5noise_$(date -u +%Y%m%dT%H%M%S)_$$}"
# 图表按 samples/UE编号/repeat编号/执行步骤保存，总体误差在 summary/。
REPORT_ROOT="${REPORT_ROOT:-${EXPERIMENT_ROOT}/step_report_$(date -u +%Y%m%dT%H%M%S)_$$}"
# 1：只固定采样计划并绘图；0：运行全部实验并绘图。
PLAN_ONLY="${PLAN_ONLY:-0}"
# 1：继续 EXPERIMENT_ROOT 中的已有计划，不读取修改后的配置。
RESUME="${RESUME:-0}"
# 计算参数：numpy 为 CPU；cuda 为 GPU，GPU_IDS 是 nvidia-smi 中的设备编号。
COMPUTE_BACKEND="${COMPUTE_BACKEND:-numpy}"
GPU_IDS="${GPU_IDS:-0}"
WORKERS="${WORKERS:-1}"
# WORKERS 只控制 CPU 模式；GPU 模式固定每个 GPU_IDS 中的编号一个进程。
MUSIC_BATCH_SIZE="${MUSIC_BATCH_SIZE:-4}"
MUSIC_ANGLE_CHUNK_SIZE="${MUSIC_ANGLE_CHUNK_SIZE:-32}"
CPU_THREADS="${CPU_THREADS:-1}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.2}"
# ===================================================================

if [[ ! "${PLAN_ONLY}" =~ ^[01]$ || ! "${RESUME}" =~ ^[01]$ ]]; then
  echo "PLAN_ONLY 和 RESUME 只能为 0 或 1" >&2
  exit 2
fi
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache/xdg"
export LD_LIBRARY_PATH="${SIONNA_BASE_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ "${COMPUTE_BACKEND}" == "cuda" ]]; then
  export CUDA_PATH
  export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
fi
# 在 Python/NumPy 导入前设置，避免每个进程再启动大量 CPU 数值计算线程。
export OPENBLAS_NUM_THREADS="${CPU_THREADS}" OMP_NUM_THREADS="${CPU_THREADS}"
export MKL_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"
cd "${PROJECT_ROOT}"
RUN_ARGS=(--output "${EXPERIMENT_ROOT}" --report-output "${REPORT_ROOT}")
RUN_ARGS+=(--compute-backend "${COMPUTE_BACKEND}" --gpu-ids "${GPU_IDS}"
  --workers "${WORKERS}" --music-batch-size "${MUSIC_BATCH_SIZE}"
  --music-angle-chunk-size "${MUSIC_ANGLE_CHUNK_SIZE}" --cpu-threads "${CPU_THREADS}")
if [[ "${RESUME}" == "0" ]]; then
  RUN_ARGS+=(--config "${EXPERIMENT_CONFIG}")
fi
if [[ "${PLAN_ONLY}" == "1" ]]; then
  RUN_ARGS+=(--plan-only)
fi
exec "${PYTHON_BIN}" -m time_bias_localization.experiment "${RUN_ARGS[@]}"
