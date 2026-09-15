#!/usr/bin/env bash
set -euo pipefail

# ============================== 参数区 ==============================
MUSIC_RERUN_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
MUSIC_RERUN_PYTHON_BIN="${MUSIC_RERUN_PYTHON_BIN:-${MUSIC_RERUN_PROJECT_ROOT}/.sionna-venv/bin/python}"
# 实验规模与计时设置；其中 generation_config 指向实际 MUSIC 参数文件。
MUSIC_RERUN_CONFIG_PATH="${MUSIC_RERUN_CONFIG_PATH:-${MUSIC_RERUN_PROJECT_ROOT}/configs/diffraction_boundary_accuracy_v5.yaml}"
# 固定的 30 UE、150 份 CSI 和随机种子；只读引用，实验期间须保留。
MUSIC_RERUN_SOURCE_ROOT="${MUSIC_RERUN_SOURCE_ROOT:-${MUSIC_RERUN_PROJECT_ROOT}/outputs/diffraction_boundary_v3}"
# 每轮使用新目录。可通过环境变量指定；已存在时拒绝启动。
MUSIC_RERUN_OUTPUT_ROOT="${MUSIC_RERUN_OUTPUT_ROOT:-${MUSIC_RERUN_PROJECT_ROOT}/outputs/diffraction_boundary_music_v5_$(date -u +%Y%m%dT%H%M%S)_$$}"
# 必填 nvidia-smi 中的物理编号，如 2,5；不会默认开放全部 GPU。
MUSIC_RERUN_GPU_IDS="${MUSIC_RERUN_GPU_IDS:-}"
MUSIC_RERUN_CPU_THREADS="${MUSIC_RERUN_CPU_THREADS:-1}"
MUSIC_RERUN_CUDA_LIB_DIR="${MUSIC_RERUN_CUDA_LIB_DIR:-/usr/local/cuda-12.2/lib64}"
# 日志：<输出目录>/run_records/<启动时间_编号>/task.log。
# 同目录保存实际任务 PID、启动参数、开始/结束时间及退出码。
# ====================================================================

MUSIC_RERUN_ACTION="${1:-start}"
if [[ "${MUSIC_RERUN_ACTION}" != start ]]; then
  [[ $# -eq 2 ]] || { echo "用法：$0 log|status|stop /绝对输出目录" >&2; exit 2; }
  export BOUNDARY_PYTHON_BIN="${MUSIC_RERUN_PYTHON_BIN}"
  exec bash "${MUSIC_RERUN_PROJECT_ROOT}/run_boundary_experiment.sh" "${MUSIC_RERUN_ACTION}" "$2"
fi
[[ $# -le 1 ]] || { echo 'start 不接受额外位置参数，请在参数区或环境变量设置。' >&2; exit 2; }
[[ -n "${MUSIC_RERUN_GPU_IDS//[[:space:]]/}" ]] || {
  echo '请填写 MUSIC_RERUN_GPU_IDS，例如 MUSIC_RERUN_GPU_IDS=2,5；使用本次允许的物理 GPU 编号。' >&2
  exit 2
}
cd "${MUSIC_RERUN_PROJECT_ROOT}"
[[ -x "${MUSIC_RERUN_PYTHON_BIN}" ]] || { echo "Python 不可执行：${MUSIC_RERUN_PYTHON_BIN}" >&2; exit 1; }
[[ -f "${MUSIC_RERUN_CONFIG_PATH}" ]] || { echo "配置不存在：${MUSIC_RERUN_CONFIG_PATH}" >&2; exit 1; }
[[ -f "${MUSIC_RERUN_SOURCE_ROOT}/experiment.json" && -f "${MUSIC_RERUN_SOURCE_ROOT}/pilot/plan.json" ]] || {
  echo "源目录缺少实验记录或固定观测计划：${MUSIC_RERUN_SOURCE_ROOT}" >&2
  exit 1
}
[[ ! -e "${MUSIC_RERUN_OUTPUT_ROOT}" && ! -L "${MUSIC_RERUN_OUTPUT_ROOT}" ]] || {
  echo "输出目录已存在，请设置新的 MUSIC_RERUN_OUTPUT_ROOT：${MUSIC_RERUN_OUTPUT_ROOT}" >&2
  exit 1
}

export BOUNDARY_PYTHON_BIN="${MUSIC_RERUN_PYTHON_BIN}"
export BOUNDARY_CONFIG_PATH="${MUSIC_RERUN_CONFIG_PATH}"
export BOUNDARY_SOURCE_ROOT="${MUSIC_RERUN_SOURCE_ROOT}"
export BOUNDARY_OUTPUT_ROOT="${MUSIC_RERUN_OUTPUT_ROOT}"
export BOUNDARY_GPU_IDS="${MUSIC_RERUN_GPU_IDS}"
export BOUNDARY_CPU_THREADS="${MUSIC_RERUN_CPU_THREADS}"
export BOUNDARY_CUDA_LIB_DIR="${MUSIC_RERUN_CUDA_LIB_DIR}"
export BOUNDARY_PHASE=pilot BOUNDARY_MODE=experiment
printf '实验配置：%s\n固定 CSI 来源：%s\n新结果目录：%s\n' \
  "${MUSIC_RERUN_CONFIG_PATH}" "${MUSIC_RERUN_SOURCE_ROOT}" "${MUSIC_RERUN_OUTPUT_ROOT}"
# 共用现有 nohup + setsid 启动器；标准输入关闭，日志及时落盘。
exec bash "${MUSIC_RERUN_PROJECT_ROOT}/run_boundary_experiment.sh" start
