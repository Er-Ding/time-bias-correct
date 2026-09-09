#!/usr/bin/env bash
set -euo pipefail

# ============================ 用户参数区 ============================
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.sionna-venv/bin/python}"
# 输入：UE 数、独立噪声次数、区域在实验 YAML；其 generation_config 设置 BS/MUSIC。
# 参考偏差、DBSCAN 邻域距离及最低点数在 generation_config 的 localization 段设置。
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${PROJECT_ROOT}/configs/visualization_experiment_munich.yaml}"
COMPUTE_BACKEND="${COMPUTE_BACKEND:-cuda}"
GPU_IDS="${GPU_IDS:-0}"
# WORKERS 只用于 CPU；GPU 模式每张指定卡运行一个工作进程。
WORKERS="${WORKERS:-1}"
MUSIC_BATCH_SIZE="${MUSIC_BATCH_SIZE:-4}"
MUSIC_ANGLE_CHUNK_SIZE="${MUSIC_ANGLE_CHUNK_SIZE:-32}"
CPU_THREADS="${CPU_THREADS:-1}"
CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-12.2}"
SIONNA_BASE_ENV="${SIONNA_BASE_ENV:-/data/zhujun/conda_envs/envrecons-differt}"
# 输出：新流程使用新目录；RESUME=1 只能继续相同流程保存的计划。
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PROJECT_ROOT}/outputs/fine_dbscan_experiment_$(date -u +%Y%m%dT%H%M%S)_$$}"
REPORT_ROOT="${REPORT_ROOT:-${EXPERIMENT_ROOT}/step_report}"
# PLAN_ONLY=1 只创建实验计划及已有信息的图；0 执行实验。
PLAN_ONLY="${PLAN_ONLY:-0}"
RESUME="${RESUME:-0}"
# ===================================================================

export PROJECT_ROOT PYTHON_BIN EXPERIMENT_CONFIG COMPUTE_BACKEND GPU_IDS WORKERS
export MUSIC_BATCH_SIZE MUSIC_ANGLE_CHUNK_SIZE CPU_THREADS CUDA_PATH SIONNA_BASE_ENV
export EXPERIMENT_ROOT REPORT_ROOT PLAN_ONLY RESUME
exec bash "${PROJECT_ROOT}/run_visualization_experiment.sh"
