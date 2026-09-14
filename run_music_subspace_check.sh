#!/usr/bin/env bash
set -euo pipefail
# ============================== 参数区 ==============================
MUSIC_SUBSPACE_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
MUSIC_SUBSPACE_PYTHON_BIN="${MUSIC_SUBSPACE_PYTHON_BIN:-${MUSIC_SUBSPACE_PROJECT_ROOT}/.sionna-venv/bin/python}"
# 输入为新的阈值配置；仅执行 CPU 回归测试，不重跑历史实验。
MUSIC_SUBSPACE_CONFIG="${MUSIC_SUBSPACE_CONFIG:-${MUSIC_SUBSPACE_PROJECT_ROOT}/configs/diffraction_boundary_generation_v4.yaml}"
MUSIC_SUBSPACE_TEST_SCOPE="${MUSIC_SUBSPACE_TEST_SCOPE:-all}"
MUSIC_SUBSPACE_OUTPUT_ROOT="${MUSIC_SUBSPACE_OUTPUT_ROOT:-${MUSIC_SUBSPACE_PROJECT_ROOT}/outputs/music_subspace_check_$(date -u +%Y%m%dT%H%M%S)_$$}"
MUSIC_SUBSPACE_CPU_THREADS="${MUSIC_SUBSPACE_CPU_THREADS:-1}"
# 本脚本仅使用 CPU，不申请 GPU。
# ====================================================================
MUSIC_SUBSPACE_ACTION="${1:-start}"
if [[ "${MUSIC_SUBSPACE_ACTION}" != start ]]; then
  [[ $# -eq 2 ]] || { echo "用法：$0 log|status|stop /绝对输出目录" >&2; exit 2; }
  MUSIC_SUBSPACE_MANAGEMENT_ROOT="$(realpath -- "$2")"
  [[ -f "${MUSIC_SUBSPACE_MANAGEMENT_ROOT}/latest_run.txt" ]] || { echo "没有运行记录" >&2; exit 1; }
  IFS= read -r MUSIC_SUBSPACE_MANAGEMENT_RUN <"${MUSIC_SUBSPACE_MANAGEMENT_ROOT}/latest_run.txt"
  case "${MUSIC_SUBSPACE_ACTION}" in
    log) exec tail -n 100 -F "${MUSIC_SUBSPACE_MANAGEMENT_RUN}/task.log" ;;
    status|stop) exec "${MUSIC_SUBSPACE_PYTHON_BIN}" "${MUSIC_SUBSPACE_PROJECT_ROOT}/scripts/detached_task.py" "${MUSIC_SUBSPACE_ACTION}" "${MUSIC_SUBSPACE_MANAGEMENT_RUN}" ;;
    *) echo "动作只能是 start、log、status 或 stop" >&2; exit 2 ;;
  esac
fi
[[ $# -le 1 ]] || { echo 'start 不接受额外位置参数，请在参数区或环境变量设置。' >&2; exit 2; }
cd "${MUSIC_SUBSPACE_PROJECT_ROOT}"
[[ -x "${MUSIC_SUBSPACE_PYTHON_BIN}" ]]
[[ -f "${MUSIC_SUBSPACE_CONFIG}" ]]
[[ "${MUSIC_SUBSPACE_TEST_SCOPE}" == all || "${MUSIC_SUBSPACE_TEST_SCOPE}" == focused ]]
command -v nohup >/dev/null
command -v setsid >/dev/null
mkdir -p -- "$(dirname -- "${MUSIC_SUBSPACE_OUTPUT_ROOT}")"
mkdir -- "${MUSIC_SUBSPACE_OUTPUT_ROOT}"
MUSIC_SUBSPACE_OUTPUT_ROOT="$(realpath -- "${MUSIC_SUBSPACE_OUTPUT_ROOT}")"
MUSIC_SUBSPACE_RUN_DIR="${MUSIC_SUBSPACE_OUTPUT_ROOT}/run_records/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p -- "${MUSIC_SUBSPACE_RUN_DIR}"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH="${MUSIC_SUBSPACE_PROJECT_ROOT}/src" PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${MUSIC_SUBSPACE_CPU_THREADS}" OMP_NUM_THREADS="${MUSIC_SUBSPACE_CPU_THREADS}" MKL_NUM_THREADS="${MUSIC_SUBSPACE_CPU_THREADS}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export MPLCONFIGDIR="${MUSIC_SUBSPACE_OUTPUT_ROOT}/matplotlib_cache"
nohup setsid "${MUSIC_SUBSPACE_PYTHON_BIN}" -u scripts/detached_task.py run "${MUSIC_SUBSPACE_RUN_DIR}" -- \
  "${MUSIC_SUBSPACE_PYTHON_BIN}" -u scripts/check_music_subspace.py \
  --output "${MUSIC_SUBSPACE_OUTPUT_ROOT}/validation" --config "${MUSIC_SUBSPACE_CONFIG}" --scope "${MUSIC_SUBSPACE_TEST_SCOPE}" \
  </dev/null >"${MUSIC_SUBSPACE_RUN_DIR}/task.log" 2>&1 &
for ((attempt=0; attempt<50; attempt++)); do
  [[ -f "${MUSIC_SUBSPACE_RUN_DIR}/task.json" || -f "${MUSIC_SUBSPACE_RUN_DIR}/completion.json" ]] && break
  sleep 0.1
done
[[ -f "${MUSIC_SUBSPACE_RUN_DIR}/task.json" ]] || { cat "${MUSIC_SUBSPACE_RUN_DIR}/task.log" >&2; exit 1; }
printf '%s\n' "${MUSIC_SUBSPACE_RUN_DIR}" >"${MUSIC_SUBSPACE_OUTPUT_ROOT}/latest_run.txt"
"${MUSIC_SUBSPACE_PYTHON_BIN}" scripts/detached_task.py status "${MUSIC_SUBSPACE_RUN_DIR}"
head -n 8 "${MUSIC_SUBSPACE_RUN_DIR}/task.log"
printf '实时日志：tail -n 100 -F %q\n' "${MUSIC_SUBSPACE_RUN_DIR}/task.log"
printf '查看状态：bash %q status %q\n' "${MUSIC_SUBSPACE_PROJECT_ROOT}/run_music_subspace_check.sh" "${MUSIC_SUBSPACE_OUTPUT_ROOT}"
printf '停止任务：bash %q stop %q\n' "${MUSIC_SUBSPACE_PROJECT_ROOT}/run_music_subspace_check.sh" "${MUSIC_SUBSPACE_OUTPUT_ROOT}"
printf 'Ctrl+C 只退出日志查看，不会停止后台任务。\n'
