#!/usr/bin/env bash
set -euo pipefail

# ============================== 参数区 ==============================
RT_CUDA_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
RT_CUDA_PYTHON_BIN="${RT_CUDA_PYTHON_BIN:-${RT_CUDA_PROJECT_ROOT}/.sionna-venv/bin/python}"
# 必填：nvidia-smi 中的一张物理卡编号。单卡对照便于判断 CUDA 本身的收益。
RT_CUDA_GPU_ID="${RT_CUDA_GPU_ID:-}"
RT_CUDA_MODE="${RT_CUDA_MODE:-benchmark}" # tests 或 benchmark
RT_CUDA_INPUT_ROOT="${RT_CUDA_INPUT_ROOT:-${RT_CUDA_PROJECT_ROOT}/outputs/diffraction_boundary_v1}"
# 可选：已有完整反向计时的 artifacts 目录，用于继续完整流程核对。
RT_CUDA_REUSE_ROOT="${RT_CUDA_REUSE_ROOT:-}"
RT_CUDA_OUTPUT_ROOT="${RT_CUDA_OUTPUT_ROOT:-${RT_CUDA_PROJECT_ROOT}/outputs/reverse_cuda_$(date -u +%Y%m%dT%H%M%S)_$$}"
RT_CUDA_UE_IDS="${RT_CUDA_UE_IDS:-PILOT_0001,PILOT_0002,PILOT_0006,PILOT_0017,PILOT_0022,PILOT_0030}"
RT_CUDA_NOISE_REPEATS="${RT_CUDA_NOISE_REPEATS:-2}"
RT_CUDA_TIMING_REPEATS="${RT_CUDA_TIMING_REPEATS:-3}"
RT_CUDA_PIPELINE_CASES="${RT_CUDA_PIPELINE_CASES:-2}"
RT_CUDA_TIMEOUT_SECONDS="${RT_CUDA_TIMEOUT_SECONDS:-2400}"
RT_CUDA_CPU_THREADS="${RT_CUDA_CPU_THREADS:-1}"
RT_CUDA_TOOLKIT_ROOT="${RT_CUDA_TOOLKIT_ROOT:-/usr/local/cuda-12.2}"
# ====================================================================

cd "${RT_CUDA_PROJECT_ROOT}"
[[ -x "${RT_CUDA_PYTHON_BIN}" ]] || { echo 'Python 路径不可执行' >&2; exit 1; }
RT_CUDA_ACTION="${1:-start}"
case "${RT_CUDA_ACTION}" in
  status|stop)
    [[ $# -eq 2 ]] || { echo '请在管理命令后填写运行目录的绝对路径' >&2; exit 1; }
    exec "${RT_CUDA_PYTHON_BIN}" "${RT_CUDA_PROJECT_ROOT}/scripts/detached_task.py" "${RT_CUDA_ACTION}" "$2"
    ;;
  log)
    [[ $# -eq 2 ]] || { echo '请在 log 后填写运行目录的绝对路径' >&2; exit 1; }
    echo 'Ctrl+C 只退出日志查看，不会停止后台任务。'
    exec tail -n 100 -F "$2/task.log"
    ;;
  start) [[ $# -le 1 ]] || { echo '启动参数请在参数区或环境变量中设置' >&2; exit 1; } ;;
  *) echo '用法：脚本 [start | log 运行目录 | status 运行目录 | stop 运行目录]' >&2; exit 1 ;;
esac
[[ "${RT_CUDA_GPU_ID}" =~ ^[0-9]+$ ]] || { echo '必须填写 RT_CUDA_GPU_ID，例如 3；不默认选择 GPU。' >&2; exit 1; }
for RT_CUDA_NUMBER in "${RT_CUDA_NOISE_REPEATS}" "${RT_CUDA_TIMING_REPEATS}" "${RT_CUDA_TIMEOUT_SECONDS}" "${RT_CUDA_CPU_THREADS}"; do
  [[ "${RT_CUDA_NUMBER}" =~ ^[1-9][0-9]*$ ]] || { echo '次数、线程数和超时必须为正整数' >&2; exit 1; }
done
[[ "${RT_CUDA_PIPELINE_CASES}" =~ ^[0-9]+$ ]] || { echo 'RT_CUDA_PIPELINE_CASES 必须为非负整数' >&2; exit 1; }
case "${RT_CUDA_MODE}" in
  tests) RT_CUDA_TASK=("${RT_CUDA_PYTHON_BIN}" -u -m pytest tests/test_reverse_cuda.py tests/test_benchmark_reverse_cuda.py tests/test_diffraction_prefixes.py tests/test_reverse_compute.py) ;;
  benchmark)
    [[ -f "${RT_CUDA_INPUT_ROOT}/experiment.json" && -f scripts/benchmark_reverse_cuda.py ]]
    RT_CUDA_TASK=("${RT_CUDA_PYTHON_BIN}" -u scripts/benchmark_reverse_cuda.py
      --experiment "${RT_CUDA_INPUT_ROOT}" --output "${RT_CUDA_OUTPUT_ROOT}/artifacts"
      --ue-ids "${RT_CUDA_UE_IDS}" --noise-repeats "${RT_CUDA_NOISE_REPEATS}"
      --timing-repeats "${RT_CUDA_TIMING_REPEATS}" --pipeline-cases "${RT_CUDA_PIPELINE_CASES}"
      --timeout-seconds "${RT_CUDA_TIMEOUT_SECONDS}")
    if [[ -n "${RT_CUDA_REUSE_ROOT}" ]]; then
      [[ -f "${RT_CUDA_REUSE_ROOT}/timings.jsonl" && -f "${RT_CUDA_REUSE_ROOT}/protocol.json" ]]
      RT_CUDA_TASK+=(--reuse-reverse-from "${RT_CUDA_REUSE_ROOT}")
    fi
    ;;
  *) echo 'RT_CUDA_MODE 必须为 tests 或 benchmark' >&2; exit 1 ;;
esac
command -v nohup >/dev/null
command -v setsid >/dev/null
RT_CUDA_ENV_LIB="$("${RT_CUDA_PYTHON_BIN}" -c 'import sys; print(sys.base_prefix + "/lib")')"
export LD_LIBRARY_PATH="${RT_CUDA_ENV_LIB}:${RT_CUDA_TOOLKIT_ROOT}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_PATH="${RT_CUDA_TOOLKIT_ROOT}"
mkdir -p -- "$(dirname -- "${RT_CUDA_OUTPUT_ROOT}")"
mkdir -- "${RT_CUDA_OUTPUT_ROOT}"
RT_CUDA_OUTPUT_ROOT="$(realpath -- "${RT_CUDA_OUTPUT_ROOT}")"
CUDA_VISIBLE_DEVICES="$("${RT_CUDA_PYTHON_BIN}" scripts/select_boundary_gpus.py --gpu-ids "${RT_CUDA_GPU_ID}" --record "${RT_CUDA_OUTPUT_ROOT}/gpu_allocation.json")"
export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="${RT_CUDA_PROJECT_ROOT}/src" PYTHONUNBUFFERED=1 PYTHONUTF8=1
export LANG=C.UTF-8 LC_ALL=C.UTF-8
export OPENBLAS_NUM_THREADS="${RT_CUDA_CPU_THREADS}" OMP_NUM_THREADS="${RT_CUDA_CPU_THREADS}" MKL_NUM_THREADS="${RT_CUDA_CPU_THREADS}"
export CUPY_CACHE_DIR="${RT_CUDA_OUTPUT_ROOT}/cupy_cache" MPLCONFIGDIR="${RT_CUDA_OUTPUT_ROOT}/matplotlib_cache"
export TBC_RUN_CUDA_REVERSE_TESTS=1
nohup setsid "${RT_CUDA_PYTHON_BIN}" -u scripts/detached_task.py run "${RT_CUDA_OUTPUT_ROOT}" -- \
  "${RT_CUDA_TASK[@]}" </dev/null >"${RT_CUDA_OUTPUT_ROOT}/task.log" 2>&1 &
for ((RT_CUDA_ATTEMPT=0; RT_CUDA_ATTEMPT<50; RT_CUDA_ATTEMPT++)); do
  [[ -f "${RT_CUDA_OUTPUT_ROOT}/task.json" || -f "${RT_CUDA_OUTPUT_ROOT}/completion.json" ]] && break
  sleep 0.1
done
[[ -f "${RT_CUDA_OUTPUT_ROOT}/task.json" ]] || { cat "${RT_CUDA_OUTPUT_ROOT}/task.log" >&2; exit 1; }
"${RT_CUDA_PYTHON_BIN}" scripts/detached_task.py status "${RT_CUDA_OUTPUT_ROOT}"
printf '实时日志：tail -n 100 -F %q\n' "${RT_CUDA_OUTPUT_ROOT}/task.log"
printf '查看状态：bash %q status %q\n' "${RT_CUDA_PROJECT_ROOT}/run_reverse_cuda_experiment.sh" "${RT_CUDA_OUTPUT_ROOT}"
printf '停止任务：bash %q stop %q\n' "${RT_CUDA_PROJECT_ROOT}/run_reverse_cuda_experiment.sh" "${RT_CUDA_OUTPUT_ROOT}"
printf 'Ctrl+C 只退出日志查看，不会停止后台任务。\n'
