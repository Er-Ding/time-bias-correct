#!/usr/bin/env bash
set -euo pipefail

# ============================ 用户参数区 ============================
PROJECT_ROOT="${PROJECT_ROOT:-/data/zhujun/differt_projects/time-bias-correct}"
SIONNA_BASE_ENV="${SIONNA_BASE_ENV:-/data/zhujun/conda_envs/envrecons-differt}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.sionna-venv/bin/python}"
SMOKE_CONFIG="${SMOKE_CONFIG:-${PROJECT_ROOT}/configs/deepmimo_sionna_smoke.yaml}"
EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-${PROJECT_ROOT}/configs/visualization_experiment_munich.yaml}"
SCENE_JSON_REL="outputs/deepmimo_sionna_smoke_run11/scene/scene_2d.json"
SCENE_JSON="${PROJECT_ROOT}/${SCENE_JSON_REL}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PLAN_ONLY="${PLAN_ONLY:-0}"
RESUME="${RESUME:-0}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-${PROJECT_ROOT}/outputs/gpu_munich_$(date -u +%Y%m%dT%H%M%S)_$$}"
REPORT_ROOT="${REPORT_ROOT:-${EXPERIMENT_ROOT}/step_report}"
# ===================================================================

PROJECT_ROOT="$(readlink -m "${PROJECT_ROOT}")"
SMOKE_CONFIG="$(readlink -m "${SMOKE_CONFIG}")"
EXPERIMENT_CONFIG="$(readlink -m "${EXPERIMENT_CONFIG}")"
SCENE_JSON="$(readlink -m "${SCENE_JSON}")"
EXPERIMENT_ROOT="$(readlink -m "${EXPERIMENT_ROOT}")"
REPORT_ROOT="$(readlink -m "${REPORT_ROOT}")"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "找不到 Python：${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -f "${SMOKE_CONFIG}" ]]; then
  echo "找不到 Smoke 配置：${SMOKE_CONFIG}" >&2
  exit 2
fi

if [[ ! -f "${EXPERIMENT_CONFIG}" ]]; then
  echo "找不到可视化实验配置：${EXPERIMENT_CONFIG}" >&2
  exit 2
fi

if [[ ! -f "${SCENE_JSON}" ]]; then
  echo "缺少可视化实验场景：${SCENE_JSON}，先补一次场景导出" >&2
  CONFIG_PATH="${SMOKE_CONFIG}" \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    SIONNA_BASE_ENV="${SIONNA_BASE_ENV}" \
    "${PROJECT_ROOT}/run_prepare_deepmimo_sionna_scene.sh"
fi

if [[ ! -f "${SCENE_JSON}" ]]; then
  echo "场景准备结束，但仍找不到 ${SCENE_JSON}；请检查 SMOKE_CONFIG 中的 output.root。" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"

PROJECT_ROOT="${PROJECT_ROOT}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG}" \
  GPU_IDS="${GPU_IDS}" \
  MUSIC_BATCH_SIZE="${MUSIC_BATCH_SIZE:-4}" \
  MUSIC_ANGLE_CHUNK_SIZE="${MUSIC_ANGLE_CHUNK_SIZE:-32}" \
  CPU_THREADS="${CPU_THREADS:-1}" \
  EXPERIMENT_ROOT="${EXPERIMENT_ROOT}" \
  REPORT_ROOT="${REPORT_ROOT}" \
  PLAN_ONLY="${PLAN_ONLY}" \
  RESUME="${RESUME}" \
  "${PROJECT_ROOT}/run_gpu_experiment.sh"
