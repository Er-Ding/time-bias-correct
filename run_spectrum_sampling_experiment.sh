#!/usr/bin/env bash
set -euo pipefail

# ============================ 用户参数区 ============================
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.sionna-venv/bin/python}"
# UE 数/实验噪声重复/采样区域在此文件；BS 与谱面采样参数在其 generation_config。
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${PROJECT_ROOT}/configs/visualization_experiment_munich.yaml}"
COMPUTE_BACKEND="${COMPUTE_BACKEND:-cuda}"
GPU_IDS="${GPU_IDS:-0}"
WORKERS="${WORKERS:-1}"
MUSIC_BATCH_SIZE="${MUSIC_BATCH_SIZE:-4}"
MUSIC_ANGLE_CHUNK_SIZE="${MUSIC_ANGLE_CHUNK_SIZE:-32}"
CPU_THREADS="${CPU_THREADS:-1}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PROJECT_ROOT}/outputs/spectrum_experiment_$(date -u +%Y%m%dT%H%M%S)_$$}"
REPORT_ROOT="${REPORT_ROOT:-${EXPERIMENT_ROOT}/step_report}"
PLAN_ONLY="${PLAN_ONLY:-0}"
RESUME="${RESUME:-0}"
# ===================================================================

export PROJECT_ROOT PYTHON_BIN EXPERIMENT_CONFIG COMPUTE_BACKEND GPU_IDS WORKERS
export MUSIC_BATCH_SIZE MUSIC_ANGLE_CHUNK_SIZE CPU_THREADS
export EXPERIMENT_ROOT REPORT_ROOT PLAN_ONLY RESUME
exec bash "${PROJECT_ROOT}/run_visualization_experiment.sh"
