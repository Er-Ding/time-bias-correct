#!/usr/bin/env bash
set -euo pipefail

# ============================== 参数区 ==============================
DIFFRACTION_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
DIFFRACTION_PYTHON_BIN="${DIFFRACTION_PYTHON_BIN:-${DIFFRACTION_PROJECT_ROOT}/.sionna-venv/bin/python}"
# tests：完整测试；demo：绕射闭环及报告；replay：原 CSI 对照；all：依次全部执行。
DIFFRACTION_CHECK_MODE="${DIFFRACTION_CHECK_MODE:-all}"
DIFFRACTION_CONFIG_PATH="${DIFFRACTION_CONFIG_PATH:-${DIFFRACTION_PROJECT_ROOT}/configs/diffraction_demo.yaml}"
DIFFRACTION_SOURCE_RUN="${DIFFRACTION_SOURCE_RUN:-${DIFFRACTION_PROJECT_ROOT}/outputs/fine_dbscan_experiment_20260909T041220_4156404/UE001/repeat_000}"
DIFFRACTION_RUN_DIR="${DIFFRACTION_RUN_DIR:-${DIFFRACTION_PROJECT_ROOT}/outputs/diffraction_check_$(date -u +%Y%m%dT%H%M%S)_$$}"
DIFFRACTION_CPU_THREADS="${DIFFRACTION_CPU_THREADS:-1}"
# ====================================================================

cd "${DIFFRACTION_PROJECT_ROOT}"
[[ -x "${DIFFRACTION_PYTHON_BIN}" ]] || { echo "Python 不可执行：${DIFFRACTION_PYTHON_BIN}" >&2; exit 1; }
[[ -f "${DIFFRACTION_CONFIG_PATH}" ]] || { echo "配置文件不存在：${DIFFRACTION_CONFIG_PATH}" >&2; exit 1; }
case "${DIFFRACTION_CHECK_MODE}" in tests|demo|replay|all) ;; *) echo "检查模式无效" >&2; exit 1;; esac
command -v nohup >/dev/null
command -v setsid >/dev/null
mkdir -p -- "$(dirname -- "${DIFFRACTION_RUN_DIR}")"
mkdir -- "${DIFFRACTION_RUN_DIR}"
export PYTHONPATH="${DIFFRACTION_PROJECT_ROOT}/src" PYTHONUNBUFFERED=1
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export OPENBLAS_NUM_THREADS="${DIFFRACTION_CPU_THREADS}" OMP_NUM_THREADS="${DIFFRACTION_CPU_THREADS}"
export MKL_NUM_THREADS="${DIFFRACTION_CPU_THREADS}" MPLCONFIGDIR="${DIFFRACTION_PROJECT_ROOT}/.cache/matplotlib"
nohup setsid "${DIFFRACTION_PYTHON_BIN}" -u scripts/detached_task.py run "${DIFFRACTION_RUN_DIR}" -- \
  "${DIFFRACTION_PYTHON_BIN}" -u scripts/check_diffraction_workflow.py \
  --mode "${DIFFRACTION_CHECK_MODE}" --config "${DIFFRACTION_CONFIG_PATH}" \
  --source-run "${DIFFRACTION_SOURCE_RUN}" --output "${DIFFRACTION_RUN_DIR}/artifacts" \
  </dev/null >"${DIFFRACTION_RUN_DIR}/task.log" 2>&1 &
for ((attempt=0; attempt<50; attempt++)); do
  [[ -f "${DIFFRACTION_RUN_DIR}/task.json" ]] && break
  [[ -f "${DIFFRACTION_RUN_DIR}/completion.json" ]] && break
  sleep 0.1
done
[[ -f "${DIFFRACTION_RUN_DIR}/task.json" ]] || { cat "${DIFFRACTION_RUN_DIR}/task.log" >&2; exit 1; }
"${DIFFRACTION_PYTHON_BIN}" scripts/detached_task.py status "${DIFFRACTION_RUN_DIR}"
printf '查看日志：tail -n 100 -F %q\n' "${DIFFRACTION_RUN_DIR}/task.log"
printf '查看状态：%q %q status %q\n' "${DIFFRACTION_PYTHON_BIN}" "${DIFFRACTION_PROJECT_ROOT}/scripts/detached_task.py" "${DIFFRACTION_RUN_DIR}"
printf '停止任务：%q %q stop %q\n' "${DIFFRACTION_PYTHON_BIN}" "${DIFFRACTION_PROJECT_ROOT}/scripts/detached_task.py" "${DIFFRACTION_RUN_DIR}"
printf '查看日志时按 Ctrl+C 只退出查看，不会停止后台任务。\n'
