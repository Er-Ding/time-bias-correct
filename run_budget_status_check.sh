#!/usr/bin/env bash
set -euo pipefail
# ============================== 参数区 ==============================
BUDGET_CHECK_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
BUDGET_CHECK_PYTHON_BIN="${BUDGET_CHECK_PYTHON_BIN:-${BUDGET_CHECK_PROJECT_ROOT}/.sionna-venv/bin/python}"
# 仅读取已完成的实验；输出为新的验收及状态重分类报告，不重跑定位。
BUDGET_CHECK_EXPERIMENT_ROOT="${BUDGET_CHECK_EXPERIMENT_ROOT:-${BUDGET_CHECK_PROJECT_ROOT}/outputs/diffraction_boundary_v3}"
BUDGET_CHECK_OUTPUT_ROOT="${BUDGET_CHECK_OUTPUT_ROOT:-${BUDGET_CHECK_PROJECT_ROOT}/outputs/budget_status_check_$(date -u +%Y%m%dT%H%M%S)_$$}"
BUDGET_CHECK_CPU_THREADS="${BUDGET_CHECK_CPU_THREADS:-1}"
# 本脚本仅使用 CPU，不申请 GPU。
# ====================================================================
BUDGET_CHECK_ACTION="${1:-start}"
if [[ "${BUDGET_CHECK_ACTION}" != start ]]; then
  [[ $# -eq 2 ]] || { echo "用法：$0 log|status|stop /绝对输出目录" >&2; exit 2; }
  exec bash "${BUDGET_CHECK_PROJECT_ROOT}/run_boundary_experiment.sh" "${BUDGET_CHECK_ACTION}" "$2"
fi
[[ $# -le 1 ]] || { echo 'start 不接受额外位置参数，请在参数区或环境变量设置。' >&2; exit 2; }
cd "${BUDGET_CHECK_PROJECT_ROOT}"
[[ -x "${BUDGET_CHECK_PYTHON_BIN}" ]]
[[ -f "${BUDGET_CHECK_EXPERIMENT_ROOT}/pilot/trials.jsonl" ]]
[[ -f "${BUDGET_CHECK_EXPERIMENT_ROOT}/pilot/plan.json" ]]
command -v nohup >/dev/null
command -v setsid >/dev/null
mkdir -p -- "$(dirname -- "${BUDGET_CHECK_OUTPUT_ROOT}")"
mkdir -- "${BUDGET_CHECK_OUTPUT_ROOT}"
BUDGET_CHECK_OUTPUT_ROOT="$(realpath -- "${BUDGET_CHECK_OUTPUT_ROOT}")"
BUDGET_CHECK_RUN_DIR="${BUDGET_CHECK_OUTPUT_ROOT}/run_records/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p -- "${BUDGET_CHECK_RUN_DIR}"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH="${BUDGET_CHECK_PROJECT_ROOT}/src" PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${BUDGET_CHECK_CPU_THREADS}" OMP_NUM_THREADS="${BUDGET_CHECK_CPU_THREADS}" MKL_NUM_THREADS="${BUDGET_CHECK_CPU_THREADS}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export MPLCONFIGDIR="${BUDGET_CHECK_PROJECT_ROOT}/.cache/matplotlib"
nohup setsid "${BUDGET_CHECK_PYTHON_BIN}" -u scripts/detached_task.py run "${BUDGET_CHECK_RUN_DIR}" -- \
  "${BUDGET_CHECK_PYTHON_BIN}" -u scripts/check_budget_status.py \
  --output "${BUDGET_CHECK_OUTPUT_ROOT}/validation" --experiment "${BUDGET_CHECK_EXPERIMENT_ROOT}" \
  </dev/null >"${BUDGET_CHECK_RUN_DIR}/task.log" 2>&1 &
for ((attempt=0; attempt<50; attempt++)); do
  [[ -f "${BUDGET_CHECK_RUN_DIR}/task.json" || -f "${BUDGET_CHECK_RUN_DIR}/completion.json" ]] && break
  sleep 0.1
done
[[ -f "${BUDGET_CHECK_RUN_DIR}/task.json" ]] || { cat "${BUDGET_CHECK_RUN_DIR}/task.log" >&2; exit 1; }
printf '%s\n' "${BUDGET_CHECK_RUN_DIR}" >"${BUDGET_CHECK_OUTPUT_ROOT}/latest_run.txt"
"${BUDGET_CHECK_PYTHON_BIN}" scripts/detached_task.py status "${BUDGET_CHECK_RUN_DIR}"
printf '实时日志：tail -n 100 -F %q\n' "${BUDGET_CHECK_RUN_DIR}/task.log"
printf '查看状态：bash %q status %q\n' "${BUDGET_CHECK_PROJECT_ROOT}/run_budget_status_check.sh" "${BUDGET_CHECK_OUTPUT_ROOT}"
printf '停止任务：bash %q stop %q\n' "${BUDGET_CHECK_PROJECT_ROOT}/run_budget_status_check.sh" "${BUDGET_CHECK_OUTPUT_ROOT}"
printf 'Ctrl+C 只退出日志查看，不会停止后台任务。\n'
