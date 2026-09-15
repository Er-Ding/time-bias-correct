#!/usr/bin/env bash
set -euo pipefail
# ============================== 参数区 ==============================
RANSAC_CHECK_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
RANSAC_CHECK_PYTHON_BIN="${RANSAC_CHECK_PYTHON_BIN:-${RANSAC_CHECK_PROJECT_ROOT}/.sionna-venv/bin/python}"
# 输入为新的 RANSAC 配置；仅执行 CPU 回归测试，不重跑历史实验。
RANSAC_CHECK_CONFIG="${RANSAC_CHECK_CONFIG:-${RANSAC_CHECK_PROJECT_ROOT}/configs/diffraction_boundary_generation_v6.yaml}"
RANSAC_CHECK_TEST_SCOPE="${RANSAC_CHECK_TEST_SCOPE:-all}"
RANSAC_CHECK_OUTPUT_ROOT="${RANSAC_CHECK_OUTPUT_ROOT:-${RANSAC_CHECK_PROJECT_ROOT}/outputs/ransac_check_$(date -u +%Y%m%dT%H%M%S)_$$}"
RANSAC_CHECK_CPU_THREADS="${RANSAC_CHECK_CPU_THREADS:-1}"
# 本脚本仅使用 CPU，不申请 GPU。
# ====================================================================
RANSAC_CHECK_ACTION="${1:-start}"
if [[ "${RANSAC_CHECK_ACTION}" != start ]]; then
  [[ $# -eq 2 ]] || { echo "用法：$0 log|status|stop /绝对输出目录" >&2; exit 2; }
  RANSAC_CHECK_MANAGEMENT_ROOT="$(realpath -- "$2")"
  [[ -f "${RANSAC_CHECK_MANAGEMENT_ROOT}/latest_run.txt" ]] || { echo "没有运行记录" >&2; exit 1; }
  IFS= read -r RANSAC_CHECK_MANAGEMENT_RUN <"${RANSAC_CHECK_MANAGEMENT_ROOT}/latest_run.txt"
  case "${RANSAC_CHECK_ACTION}" in
    log) exec tail -n 100 -F "${RANSAC_CHECK_MANAGEMENT_RUN}/task.log" ;;
    status|stop) exec "${RANSAC_CHECK_PYTHON_BIN}" "${RANSAC_CHECK_PROJECT_ROOT}/scripts/detached_task.py" "${RANSAC_CHECK_ACTION}" "${RANSAC_CHECK_MANAGEMENT_RUN}" ;;
    *) echo "动作只能是 start、log、status 或 stop" >&2; exit 2 ;;
  esac
fi
[[ $# -le 1 ]] || { echo 'start 不接受额外位置参数，请在参数区或环境变量设置。' >&2; exit 2; }
cd "${RANSAC_CHECK_PROJECT_ROOT}"
[[ -x "${RANSAC_CHECK_PYTHON_BIN}" ]]
[[ -f "${RANSAC_CHECK_CONFIG}" ]]
[[ "${RANSAC_CHECK_TEST_SCOPE}" == all || "${RANSAC_CHECK_TEST_SCOPE}" == focused ]]
command -v nohup >/dev/null
command -v setsid >/dev/null
mkdir -p -- "$(dirname -- "${RANSAC_CHECK_OUTPUT_ROOT}")"
mkdir -- "${RANSAC_CHECK_OUTPUT_ROOT}"
RANSAC_CHECK_OUTPUT_ROOT="$(realpath -- "${RANSAC_CHECK_OUTPUT_ROOT}")"
RANSAC_CHECK_RUN_DIR="${RANSAC_CHECK_OUTPUT_ROOT}/run_records/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p -- "${RANSAC_CHECK_RUN_DIR}"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH="${RANSAC_CHECK_PROJECT_ROOT}/src" PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${RANSAC_CHECK_CPU_THREADS}" OMP_NUM_THREADS="${RANSAC_CHECK_CPU_THREADS}" MKL_NUM_THREADS="${RANSAC_CHECK_CPU_THREADS}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export MPLCONFIGDIR="${RANSAC_CHECK_OUTPUT_ROOT}/matplotlib_cache"
nohup setsid "${RANSAC_CHECK_PYTHON_BIN}" -u scripts/detached_task.py run "${RANSAC_CHECK_RUN_DIR}" -- \
  "${RANSAC_CHECK_PYTHON_BIN}" -u scripts/check_ransac.py \
  --output "${RANSAC_CHECK_OUTPUT_ROOT}/validation" --config "${RANSAC_CHECK_CONFIG}" --scope "${RANSAC_CHECK_TEST_SCOPE}" \
  </dev/null >"${RANSAC_CHECK_RUN_DIR}/task.log" 2>&1 &
for ((attempt=0; attempt<50; attempt++)); do
  [[ -f "${RANSAC_CHECK_RUN_DIR}/task.json" || -f "${RANSAC_CHECK_RUN_DIR}/completion.json" ]] && break
  sleep 0.1
done
[[ -f "${RANSAC_CHECK_RUN_DIR}/task.json" ]] || { cat "${RANSAC_CHECK_RUN_DIR}/task.log" >&2; exit 1; }
printf '%s\n' "${RANSAC_CHECK_RUN_DIR}" >"${RANSAC_CHECK_OUTPUT_ROOT}/latest_run.txt"
"${RANSAC_CHECK_PYTHON_BIN}" scripts/detached_task.py status "${RANSAC_CHECK_RUN_DIR}"
head -n 8 "${RANSAC_CHECK_RUN_DIR}/task.log"
printf '实时日志：tail -n 100 -F %q\n' "${RANSAC_CHECK_RUN_DIR}/task.log"
printf '查看状态：bash %q status %q\n' "${RANSAC_CHECK_PROJECT_ROOT}/run_ransac_check.sh" "${RANSAC_CHECK_OUTPUT_ROOT}"
printf '停止任务：bash %q stop %q\n' "${RANSAC_CHECK_PROJECT_ROOT}/run_ransac_check.sh" "${RANSAC_CHECK_OUTPUT_ROOT}"
printf 'Ctrl+C 只退出日志查看，不会停止后台任务。\n'
