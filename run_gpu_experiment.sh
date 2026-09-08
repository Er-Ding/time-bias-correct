#!/usr/bin/env bash
set -euo pipefail

# ============================ 用户参数区 ============================
PROJECT_ROOT="${PROJECT_ROOT:-/data/zhujun/differt_projects/time-bias-correct}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.sionna-venv/bin/python}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${PROJECT_ROOT}/configs/visualization_experiment_munich.yaml}"
# 默认一张卡；例如 GPU_IDS=0,1,2,3,4,5,6,7 使用八张卡，每卡一个常驻进程。
GPU_IDS="${GPU_IDS:-0}"
MUSIC_BATCH_SIZE="${MUSIC_BATCH_SIZE:-4}"
MUSIC_ANGLE_CHUNK_SIZE="${MUSIC_ANGLE_CHUNK_SIZE:-32}"
CPU_THREADS="${CPU_THREADS:-1}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PROJECT_ROOT}/outputs/gpu_munich_$(date -u +%Y%m%dT%H%M%S)_$$}"
REPORT_ROOT="${REPORT_ROOT:-${EXPERIMENT_ROOT}/step_report}"
PLAN_ONLY="${PLAN_ONLY:-0}"
RESUME="${RESUME:-0}"
# ===================================================================

export PROJECT_ROOT PYTHON_BIN EXPERIMENT_CONFIG GPU_IDS
export MUSIC_BATCH_SIZE MUSIC_ANGLE_CHUNK_SIZE CPU_THREADS
export EXPERIMENT_ROOT REPORT_ROOT PLAN_ONLY RESUME
export COMPUTE_BACKEND=cuda
exec bash "${PROJECT_ROOT}/run_visualization_experiment.sh"
