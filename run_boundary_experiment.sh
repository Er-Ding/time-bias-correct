#!/usr/bin/env bash
set -euo pipefail

# ============================== 参数区 ==============================
BOUNDARY_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
BOUNDARY_PYTHON_BIN="${BOUNDARY_PYTHON_BIN:-${BOUNDARY_PROJECT_ROOT}/.sionna-venv/bin/python}"
BOUNDARY_CONFIG_PATH="${BOUNDARY_CONFIG_PATH:-${BOUNDARY_PROJECT_ROOT}/configs/diffraction_boundary_experiment.yaml}"
BOUNDARY_OUTPUT_ROOT="${BOUNDARY_OUTPUT_ROOT:-${BOUNDARY_PROJECT_ROOT}/outputs/diffraction_boundary_$(date -u +%Y%m%dT%H%M%S)_$$}"
# pilot：30 UE 预跑；formal：同一输出目录内完成预跑后，另取 300 UE 正式实验。
BOUNDARY_PHASE="${BOUNDARY_PHASE:-pilot}"
# experiment：运行实验；validate：完整测试和 2+1 UE 的小规模流程检查。
BOUNDARY_MODE="${BOUNDARY_MODE:-experiment}"
# 必填：nvidia-smi 显示的物理编号，如 2,3,4,5,6,7；每次由用户指定。
# 只有 validate 模式允许明确填写 cpu。不会继承或默认开放其他 GPU。
BOUNDARY_GPU_IDS="${BOUNDARY_GPU_IDS:-}"
# 可选：复用旧实验已固定的 UE 和 CSI；新结果仍写入独立输出目录。
BOUNDARY_SOURCE_ROOT="${BOUNDARY_SOURCE_ROOT:-}"
BOUNDARY_CPU_THREADS="${BOUNDARY_CPU_THREADS:-1}"
# 留空时从所选 Python 的基础运行环境解析 lib，避免误用系统 C++ 动态库。
BOUNDARY_ENV_LIB_DIR="${BOUNDARY_ENV_LIB_DIR:-}"
BOUNDARY_CUDA_LIB_DIR="${BOUNDARY_CUDA_LIB_DIR:-/usr/local/cuda-12.2/lib64}"
BOUNDARY_RUN_RECORDS="${BOUNDARY_OUTPUT_ROOT}/run_records"
# 详细采样、噪声、超时和预热参数集中在 BOUNDARY_CONFIG_PATH。
# ====================================================================

BOUNDARY_COMMAND="${1:-start}"
if [[ "${BOUNDARY_COMMAND}" != start ]]; then
  [[ $# -eq 2 ]] || { echo "用法：$0 log|status|stop /实验输出目录" >&2; exit 1; }
  BOUNDARY_OUTPUT_ROOT="$(realpath -- "$2")"
  [[ -f "${BOUNDARY_OUTPUT_ROOT}/latest_run.txt" ]] || { echo '没有运行记录' >&2; exit 1; }
  IFS= read -r BOUNDARY_RUN_DIR <"${BOUNDARY_OUTPUT_ROOT}/latest_run.txt"
  case "${BOUNDARY_COMMAND}" in
    log) echo "日志：${BOUNDARY_RUN_DIR}/task.log；Ctrl+C 只退出查看。"; exec tail -n 100 -F "${BOUNDARY_RUN_DIR}/task.log" ;;
    status|stop) exec "${BOUNDARY_PYTHON_BIN}" "${BOUNDARY_PROJECT_ROOT}/scripts/detached_task.py" "${BOUNDARY_COMMAND}" "${BOUNDARY_RUN_DIR}" ;;
    *) echo '命令必须为 start、log、status 或 stop' >&2; exit 1 ;;
  esac
fi
cd "${BOUNDARY_PROJECT_ROOT}"
[[ -x "${BOUNDARY_PYTHON_BIN}" ]] || { echo "Python 不可执行：${BOUNDARY_PYTHON_BIN}" >&2; exit 1; }
[[ -f "${BOUNDARY_CONFIG_PATH}" ]] || { echo "配置不存在：${BOUNDARY_CONFIG_PATH}" >&2; exit 1; }
case "${BOUNDARY_PHASE}" in pilot|formal) ;; *) echo '实验阶段无效' >&2; exit 1;; esac
case "${BOUNDARY_MODE}" in experiment|validate) ;; *) echo '运行模式无效' >&2; exit 1;; esac
[[ -n "${BOUNDARY_GPU_IDS//[[:space:]]/}" ]] || {
  echo '请先填写 BOUNDARY_GPU_IDS，例如 BOUNDARY_GPU_IDS=2,3,4,5,6,7；不会默认使用全部 GPU。' >&2
  echo '仅进行 CPU 流程检查时，可填写 BOUNDARY_MODE=validate BOUNDARY_GPU_IDS=cpu。' >&2
  exit 1
}
if [[ -n "${BOUNDARY_SOURCE_ROOT}" ]]; then
  [[ "${BOUNDARY_MODE}" == experiment ]] || { echo 'validate 模式不支持复用实验数据。' >&2; exit 1; }
  [[ -d "${BOUNDARY_SOURCE_ROOT}" ]] || { echo "旧实验目录不存在：${BOUNDARY_SOURCE_ROOT}" >&2; exit 1; }
  BOUNDARY_SOURCE_ROOT="$(realpath -- "${BOUNDARY_SOURCE_ROOT}")"
fi
if [[ -z "${BOUNDARY_ENV_LIB_DIR}" ]]; then
  BOUNDARY_ENV_LIB_DIR="$("${BOUNDARY_PYTHON_BIN}" -c 'import sys; from pathlib import Path; print(Path(sys.base_prefix) / "lib")')"
fi
[[ -d "${BOUNDARY_ENV_LIB_DIR}" ]] || { echo "Python 动态库目录不存在：${BOUNDARY_ENV_LIB_DIR}" >&2; exit 1; }
if [[ -d "${BOUNDARY_CUDA_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${BOUNDARY_ENV_LIB_DIR}:${BOUNDARY_CUDA_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export CUDA_PATH="$(dirname -- "${BOUNDARY_CUDA_LIB_DIR}")"
elif [[ "${BOUNDARY_MODE}" == validate && "${BOUNDARY_GPU_IDS}" == cpu ]]; then
  export LD_LIBRARY_PATH="${BOUNDARY_ENV_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  echo "CUDA 动态库目录不存在：${BOUNDARY_CUDA_LIB_DIR}；请在参数区设置正确路径。" >&2
  exit 1
fi
command -v nohup >/dev/null
command -v setsid >/dev/null
command -v flock >/dev/null
mkdir -p -- "${BOUNDARY_OUTPUT_ROOT}"
BOUNDARY_OUTPUT_ROOT="$(realpath -- "${BOUNDARY_OUTPUT_ROOT}")"
# 锁随后台监督进程继承，避免同一实验目录同时启动两项任务。
exec 9>"${BOUNDARY_OUTPUT_ROOT}/.background_task.lock"
flock -n 9 || { echo "同一实验目录已有后台任务；请先查询状态。" >&2; exit 1; }
BOUNDARY_RUN_RECORDS="${BOUNDARY_OUTPUT_ROOT}/run_records"
mkdir -p -- "${BOUNDARY_RUN_RECORDS}"
BOUNDARY_RUN_DIR="${BOUNDARY_RUN_RECORDS}/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -- "${BOUNDARY_RUN_DIR}"
BOUNDARY_GPU_OPTIONS=(--gpu-ids "${BOUNDARY_GPU_IDS}" --record "${BOUNDARY_RUN_DIR}/gpu_allocation.json")
[[ "${BOUNDARY_MODE}" != validate ]] || BOUNDARY_GPU_OPTIONS+=(--allow-cpu)
CUDA_VISIBLE_DEVICES="$("${BOUNDARY_PYTHON_BIN}" "${BOUNDARY_PROJECT_ROOT}/scripts/select_boundary_gpus.py" "${BOUNDARY_GPU_OPTIONS[@]}")"
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID
export BOUNDARY_GPU_ALLOCATION_PATH="${BOUNDARY_RUN_DIR}/gpu_allocation.json"
export BOUNDARY_GPU_IDS
export PYTHONPATH="${BOUNDARY_PROJECT_ROOT}/src" PYTHONUNBUFFERED=1 PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export OPENBLAS_NUM_THREADS="${BOUNDARY_CPU_THREADS}" OMP_NUM_THREADS="${BOUNDARY_CPU_THREADS}" MKL_NUM_THREADS="${BOUNDARY_CPU_THREADS}"
export MPLCONFIGDIR="${BOUNDARY_PROJECT_ROOT}/.cache/matplotlib"
export CUPY_CACHE_DIR="${BOUNDARY_PROJECT_ROOT}/.cache/cupy"
if [[ "${BOUNDARY_MODE}" == validate ]]; then
  BOUNDARY_TASK=("${BOUNDARY_PYTHON_BIN}" -u "${BOUNDARY_PROJECT_ROOT}/scripts/check_boundary_experiment.py" --output "${BOUNDARY_RUN_DIR}/validation")
else
  BOUNDARY_TASK=("${BOUNDARY_PYTHON_BIN}" -u -m time_bias_localization.boundary_experiment --config "${BOUNDARY_CONFIG_PATH}" --output "${BOUNDARY_OUTPUT_ROOT}" --phase "${BOUNDARY_PHASE}")
  [[ -z "${BOUNDARY_SOURCE_ROOT}" ]] || BOUNDARY_TASK+=(--reuse-prepared-from "${BOUNDARY_SOURCE_ROOT}")
fi
nohup setsid "${BOUNDARY_PYTHON_BIN}" -u "${BOUNDARY_PROJECT_ROOT}/scripts/detached_task.py" run "${BOUNDARY_RUN_DIR}" -- "${BOUNDARY_TASK[@]}" \
  </dev/null >"${BOUNDARY_RUN_DIR}/task.log" 2>&1 &
for ((attempt=0; attempt<50; attempt++)); do
  [[ -f "${BOUNDARY_RUN_DIR}/task.json" || -f "${BOUNDARY_RUN_DIR}/completion.json" ]] && break
  sleep 0.1
done
[[ -f "${BOUNDARY_RUN_DIR}/task.json" ]] || { cat "${BOUNDARY_RUN_DIR}/task.log" >&2; exit 1; }
printf '%s\n' "${BOUNDARY_RUN_DIR}" >"${BOUNDARY_OUTPUT_ROOT}/latest_run.txt"
"${BOUNDARY_PYTHON_BIN}" "${BOUNDARY_PROJECT_ROOT}/scripts/detached_task.py" status "${BOUNDARY_RUN_DIR}"
head -n 12 "${BOUNDARY_RUN_DIR}/task.log"
printf '结果目录：%s\n运行记录：%s\n' "${BOUNDARY_OUTPUT_ROOT}" "${BOUNDARY_RUN_DIR}"
printf '用户指定的物理 GPU：%s\nGPU 对应记录：%s\n' "${BOUNDARY_GPU_IDS}" "${BOUNDARY_GPU_ALLOCATION_PATH}"
printf '实时日志：tail -n 100 -F %q\n' "${BOUNDARY_RUN_DIR}/task.log"
printf '查看状态：bash %q status %q\n' "${BOUNDARY_PROJECT_ROOT}/run_boundary_experiment.sh" "${BOUNDARY_OUTPUT_ROOT}"
printf '停止任务：bash %q stop %q\n' "${BOUNDARY_PROJECT_ROOT}/run_boundary_experiment.sh" "${BOUNDARY_OUTPUT_ROOT}"
printf '查看日志时按 Ctrl+C 只退出查看，不会停止后台任务。\n'
