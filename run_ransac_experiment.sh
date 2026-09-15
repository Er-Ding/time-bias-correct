#!/usr/bin/env bash
set -euo pipefail

# ============================== 参数区 ==============================
RANSAC_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
RANSAC_PYTHON_BIN="${RANSAC_PYTHON_BIN:-${RANSAC_PROJECT_ROOT}/.sionna-venv/bin/python}"
# 算法参数集中在 generation_v6.yaml：相似度、两层代表上限、抽样次数、距离门槛。
RANSAC_CONFIG_PATH="${RANSAC_CONFIG_PATH:-${RANSAC_PROJECT_ROOT}/configs/diffraction_boundary_accuracy_v6.yaml}"
RANSAC_SOURCE_ROOT="${RANSAC_SOURCE_ROOT:-${RANSAC_PROJECT_ROOT}/outputs/diffraction_boundary_v3}"
RANSAC_OUTPUT_ROOT="${RANSAC_OUTPUT_ROOT:-${RANSAC_PROJECT_ROOT}/outputs/diffraction_boundary_ransac_v6_$(date -u +%Y%m%dT%H%M%S)_$$}"
# 必须填写本次允许使用的物理 GPU 编号；不会默认使用全部卡。
RANSAC_GPU_IDS="${RANSAC_GPU_IDS:-}"
RANSAC_CPU_THREADS="${RANSAC_CPU_THREADS:-1}"
RANSAC_CUDA_LIB_DIR="${RANSAC_CUDA_LIB_DIR:-/usr/local/cuda-12.2/lib64}"
# 日志和实际任务 PID、起止时间、退出码：<输出目录>/run_records/<运行编号>/。
# ====================================================================

if [[ "${1:-start}" == start && -z "${RANSAC_GPU_IDS//[[:space:]]/}" ]]; then
  echo '请在参数区或环境变量中填写 RANSAC_GPU_IDS，例如 2,5；不会默认使用全部 GPU。' >&2
  exit 2
fi

export MUSIC_RERUN_PYTHON_BIN="${RANSAC_PYTHON_BIN}"
export MUSIC_RERUN_CONFIG_PATH="${RANSAC_CONFIG_PATH}"
export MUSIC_RERUN_SOURCE_ROOT="${RANSAC_SOURCE_ROOT}"
export MUSIC_RERUN_OUTPUT_ROOT="${RANSAC_OUTPUT_ROOT}"
export MUSIC_RERUN_GPU_IDS="${RANSAC_GPU_IDS}"
export MUSIC_RERUN_CPU_THREADS="${RANSAC_CPU_THREADS}"
export MUSIC_RERUN_CUDA_LIB_DIR="${RANSAC_CUDA_LIB_DIR}"
# 共用经过验证的 nohup + setsid 启动和整组进程管理。
exec bash "${RANSAC_PROJECT_ROOT}/run_music_rerun.sh" "$@"
