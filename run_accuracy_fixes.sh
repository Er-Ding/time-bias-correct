#!/usr/bin/env bash
set -euo pipefail
# ============================== 参数区 ==============================
ACCURACY_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
ACCURACY_PYTHON_BIN="${ACCURACY_PYTHON_BIN:-${ACCURACY_PROJECT_ROOT}/.sionna-venv/bin/python}"
# validate：完整代码验收；pilot：重新生成 30 UE × 5 噪声的单/多代表对照。
ACCURACY_MODE="${ACCURACY_MODE:-validate}"
# 必填物理 GPU 编号；仅 validate 模式允许 cpu。每次由用户指定。
ACCURACY_GPU_IDS="${ACCURACY_GPU_IDS:-}"
ACCURACY_OUTPUT_ROOT="${ACCURACY_OUTPUT_ROOT:-${ACCURACY_PROJECT_ROOT}/outputs/accuracy_fixes_$(date -u +%Y%m%dT%H%M%S)_$$}"
ACCURACY_CONFIG_PATH="${ACCURACY_CONFIG_PATH:-${ACCURACY_PROJECT_ROOT}/configs/diffraction_boundary_accuracy_v3.yaml}"
ACCURACY_SAVED_EXPERIMENT="${ACCURACY_SAVED_EXPERIMENT:-${ACCURACY_PROJECT_ROOT}/outputs/diffraction_boundary_v2}"
ACCURACY_SCENE_JSON="${ACCURACY_SCENE_JSON:-${ACCURACY_PROJECT_ROOT}/outputs/spectrum_experiment_20260908T091621_2543248/UE003/channel/scene/scene_2d.json}"
ACCURACY_CPU_THREADS="${ACCURACY_CPU_THREADS:-1}"
ACCURACY_CUDA_LIB_DIR="${ACCURACY_CUDA_LIB_DIR:-/usr/local/cuda-12.2/lib64}"
# ====================================================================
ACCURACY_ACTION="${1:-start}"
if [[ "${ACCURACY_ACTION}" != start ]]; then
  [[ $# -eq 2 ]] || { echo "用法：$0 log|status|stop /绝对输出目录" >&2; exit 2; }
  exec bash "${ACCURACY_PROJECT_ROOT}/run_boundary_experiment.sh" "${ACCURACY_ACTION}" "$2"
fi
[[ $# -le 1 ]] || { echo 'start 不接受额外位置参数，请在参数区或环境变量设置。' >&2; exit 2; }
[[ -n "${ACCURACY_GPU_IDS}" ]] || { echo '必须填写 ACCURACY_GPU_IDS，例如 0,1；仅验收可填 cpu。' >&2; exit 2; }
cd "${ACCURACY_PROJECT_ROOT}"
[[ -x "${ACCURACY_PYTHON_BIN}" ]]
if [[ "${ACCURACY_MODE}" == pilot ]]; then
  export BOUNDARY_PYTHON_BIN="${ACCURACY_PYTHON_BIN}" BOUNDARY_GPU_IDS="${ACCURACY_GPU_IDS}"
  export BOUNDARY_CONFIG_PATH="${ACCURACY_CONFIG_PATH}" BOUNDARY_OUTPUT_ROOT="${ACCURACY_OUTPUT_ROOT}"
  export BOUNDARY_PHASE=pilot BOUNDARY_MODE=experiment BOUNDARY_SOURCE_ROOT=''
  export BOUNDARY_CPU_THREADS="${ACCURACY_CPU_THREADS}" BOUNDARY_CUDA_LIB_DIR="${ACCURACY_CUDA_LIB_DIR}"
  exec bash "${ACCURACY_PROJECT_ROOT}/run_boundary_experiment.sh" start
fi
[[ "${ACCURACY_MODE}" == validate ]] || { echo 'ACCURACY_MODE 必须为 validate 或 pilot' >&2; exit 2; }
[[ -f "${ACCURACY_SCENE_JSON}" && -f "${ACCURACY_SAVED_EXPERIMENT}/pilot/trials.jsonl" ]]
command -v nohup >/dev/null
command -v setsid >/dev/null
mkdir -p -- "$(dirname -- "${ACCURACY_OUTPUT_ROOT}")"
mkdir -- "${ACCURACY_OUTPUT_ROOT}"
ACCURACY_OUTPUT_ROOT="$(realpath -- "${ACCURACY_OUTPUT_ROOT}")"
ACCURACY_RUN_DIR="${ACCURACY_OUTPUT_ROOT}/run_records/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p -- "${ACCURACY_RUN_DIR}"
ACCURACY_ENV_LIB_DIR="$("${ACCURACY_PYTHON_BIN}" -c 'import sys; from pathlib import Path; print(Path(sys.base_prefix)/"lib")')"
export LD_LIBRARY_PATH="${ACCURACY_ENV_LIB_DIR}:${ACCURACY_CUDA_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_PATH="$(dirname -- "${ACCURACY_CUDA_LIB_DIR}")"
export CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES="$("${ACCURACY_PYTHON_BIN}" scripts/select_boundary_gpus.py --gpu-ids "${ACCURACY_GPU_IDS}" --allow-cpu --record "${ACCURACY_RUN_DIR}/gpu_allocation.json")"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH="${ACCURACY_PROJECT_ROOT}/src" PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${ACCURACY_CPU_THREADS}" OMP_NUM_THREADS="${ACCURACY_CPU_THREADS}" MKL_NUM_THREADS="${ACCURACY_CPU_THREADS}"
export MPLCONFIGDIR="${ACCURACY_PROJECT_ROOT}/.cache/matplotlib" CUPY_CACHE_DIR="${ACCURACY_PROJECT_ROOT}/.cache/cupy"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
ACCURACY_BACKEND=cuda
[[ "${ACCURACY_GPU_IDS}" != cpu ]] || ACCURACY_BACKEND=numpy
nohup setsid "${ACCURACY_PYTHON_BIN}" -u scripts/detached_task.py run "${ACCURACY_RUN_DIR}" -- \
  "${ACCURACY_PYTHON_BIN}" -u scripts/check_accuracy_fixes.py --output "${ACCURACY_OUTPUT_ROOT}/validation" \
  --backend "${ACCURACY_BACKEND}" --saved-experiment "${ACCURACY_SAVED_EXPERIMENT}" --scene "${ACCURACY_SCENE_JSON}" \
  </dev/null >"${ACCURACY_RUN_DIR}/task.log" 2>&1 &
for ((attempt=0; attempt<50; attempt++)); do
  [[ -f "${ACCURACY_RUN_DIR}/task.json" || -f "${ACCURACY_RUN_DIR}/completion.json" ]] && break
  sleep 0.1
done
[[ -f "${ACCURACY_RUN_DIR}/task.json" ]] || { cat "${ACCURACY_RUN_DIR}/task.log" >&2; exit 1; }
printf '%s\n' "${ACCURACY_RUN_DIR}" >"${ACCURACY_OUTPUT_ROOT}/latest_run.txt"
"${ACCURACY_PYTHON_BIN}" scripts/detached_task.py status "${ACCURACY_RUN_DIR}"
printf '实时日志：tail -n 100 -F %q\n' "${ACCURACY_RUN_DIR}/task.log"
printf '查看状态：bash %q status %q\n' "${ACCURACY_PROJECT_ROOT}/run_accuracy_fixes.sh" "${ACCURACY_OUTPUT_ROOT}"
printf '停止任务：bash %q stop %q\n' "${ACCURACY_PROJECT_ROOT}/run_accuracy_fixes.sh" "${ACCURACY_OUTPUT_ROOT}"
printf 'Ctrl+C 只退出日志查看，不会停止后台任务。\n'
