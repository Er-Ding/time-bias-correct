#!/usr/bin/env bash
set -euo pipefail

# ============================== 参数区 ==============================
GPU_CHECK_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
GPU_CHECK_PYTHON_BIN="${GPU_CHECK_PYTHON_BIN:-${GPU_CHECK_PROJECT_ROOT}/.sionna-venv/bin/python}"
# 必须由用户填写 nvidia-smi 的物理 GPU 编号；例如 2,3,4,5,6,7。
BOUNDARY_GPU_IDS="${BOUNDARY_GPU_IDS:-}"
GPU_CHECK_OUTPUT_ROOT="${GPU_CHECK_OUTPUT_ROOT:-${GPU_CHECK_PROJECT_ROOT}/outputs/gpu_runtime_check_$(date -u +%Y%m%dT%H%M%S)_$$}"
GPU_CHECK_CPU_THREADS="${GPU_CHECK_CPU_THREADS:-1}"
GPU_CHECK_ENV_ROOT="${GPU_CHECK_ENV_ROOT:-/data/zhujun/conda_envs/envrecons-differt}"
GPU_CHECK_CUDA_ROOT="${GPU_CHECK_CUDA_ROOT:-/usr/local/cuda-12.2}"
# ====================================================================

cd "${GPU_CHECK_PROJECT_ROOT}"
[[ -x "${GPU_CHECK_PYTHON_BIN}" ]] || { echo 'Python 路径不可执行' >&2; exit 1; }
[[ -n "${BOUNDARY_GPU_IDS}" ]] || { echo '请填写 BOUNDARY_GPU_IDS，不会默认开放全部 GPU。' >&2; exit 1; }
command -v nohup >/dev/null
command -v setsid >/dev/null
mkdir -p -- "$(dirname -- "${GPU_CHECK_OUTPUT_ROOT}")"
mkdir -- "${GPU_CHECK_OUTPUT_ROOT}"
GPU_CHECK_OUTPUT_ROOT="$(realpath -- "${GPU_CHECK_OUTPUT_ROOT}")"
CUDA_VISIBLE_DEVICES="$("${GPU_CHECK_PYTHON_BIN}" scripts/select_boundary_gpus.py --gpu-ids "${BOUNDARY_GPU_IDS}" --record "${GPU_CHECK_OUTPUT_ROOT}/gpu_allocation.json")"
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1 PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export OPENBLAS_NUM_THREADS="${GPU_CHECK_CPU_THREADS}" OMP_NUM_THREADS="${GPU_CHECK_CPU_THREADS}" MKL_NUM_THREADS="${GPU_CHECK_CPU_THREADS}"
export CUDA_PATH="${GPU_CHECK_CUDA_ROOT}"
export LD_LIBRARY_PATH="${GPU_CHECK_CUDA_ROOT}/lib64:${GPU_CHECK_ENV_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export MPLCONFIGDIR="${GPU_CHECK_PROJECT_ROOT}/.cache/matplotlib" CUPY_CACHE_DIR="${GPU_CHECK_PROJECT_ROOT}/.cache/cupy"
GPU_CHECK_TASK=("${GPU_CHECK_PYTHON_BIN}" -u "${GPU_CHECK_PROJECT_ROOT}/scripts/check_boundary_gpu_runtime.py" --allocation "${GPU_CHECK_OUTPUT_ROOT}/gpu_allocation.json" --output "${GPU_CHECK_OUTPUT_ROOT}/probes")
nohup setsid "${GPU_CHECK_PYTHON_BIN}" -u "${GPU_CHECK_PROJECT_ROOT}/scripts/detached_task.py" run "${GPU_CHECK_OUTPUT_ROOT}" -- "${GPU_CHECK_TASK[@]}" </dev/null >"${GPU_CHECK_OUTPUT_ROOT}/task.log" 2>&1 &
for ((attempt=0; attempt<50; attempt++)); do
  [[ -f "${GPU_CHECK_OUTPUT_ROOT}/task.json" || -f "${GPU_CHECK_OUTPUT_ROOT}/completion.json" ]] && break
  sleep 0.1
done
[[ -f "${GPU_CHECK_OUTPUT_ROOT}/task.json" ]] || { cat "${GPU_CHECK_OUTPUT_ROOT}/task.log" >&2; exit 1; }
"${GPU_CHECK_PYTHON_BIN}" "${GPU_CHECK_PROJECT_ROOT}/scripts/detached_task.py" status "${GPU_CHECK_OUTPUT_ROOT}"
printf '运行记录：%s\n实时日志：tail -n 100 -F %q\n' "${GPU_CHECK_OUTPUT_ROOT}" "${GPU_CHECK_OUTPUT_ROOT}/task.log"
printf '查看状态：%q %q status %q\n' "${GPU_CHECK_PYTHON_BIN}" "${GPU_CHECK_PROJECT_ROOT}/scripts/detached_task.py" "${GPU_CHECK_OUTPUT_ROOT}"
printf '停止任务：%q %q stop %q\n' "${GPU_CHECK_PYTHON_BIN}" "${GPU_CHECK_PROJECT_ROOT}/scripts/detached_task.py" "${GPU_CHECK_OUTPUT_ROOT}"
printf '查看日志时按 Ctrl+C 只退出查看，不会停止后台任务。\n'
