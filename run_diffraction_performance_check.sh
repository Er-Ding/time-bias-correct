#!/usr/bin/env bash
set -euo pipefail
# ============================== 参数区 ==============================
PERFORMANCE_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
PERFORMANCE_PYTHON_BIN="${PERFORMANCE_PYTHON_BIN:-${PERFORMANCE_PROJECT_ROOT}/.sionna-venv/bin/python}"
PERFORMANCE_EXPERIMENT_ROOT="${PERFORMANCE_EXPERIMENT_ROOT:-${PERFORMANCE_PROJECT_ROOT}/outputs/diffraction_boundary_v1}"
PERFORMANCE_RUN_DIR="${PERFORMANCE_RUN_DIR:-${PERFORMANCE_PROJECT_ROOT}/outputs/diffraction_performance_$(date -u +%Y%m%dT%H%M%S)_$$}"
PERFORMANCE_TIMEOUT_SECONDS="${PERFORMANCE_TIMEOUT_SECONDS:-300}"
# candidates：同一 CSI 的冷/热反向耗时；prefixes：真实地图全首墙枚举核对。
PERFORMANCE_CHECK_MODE="${PERFORMANCE_CHECK_MODE:-candidates}"
# 此脚本只测 CPU 几何，不使用任何 GPU。
# ====================================================================
cd "${PERFORMANCE_PROJECT_ROOT}"
[[ -x "${PERFORMANCE_PYTHON_BIN}" ]]
[[ -f "${PERFORMANCE_EXPERIMENT_ROOT}/experiment.json" ]]
[[ "${PERFORMANCE_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]
case "${PERFORMANCE_CHECK_MODE}" in
  candidates) PERFORMANCE_CHECK_SCRIPT="scripts/check_diffraction_performance.py" ;;
  prefixes) PERFORMANCE_CHECK_SCRIPT="scripts/check_diffraction_prefix_equivalence.py" ;;
  *) echo "PERFORMANCE_CHECK_MODE 必须为 candidates 或 prefixes" >&2; exit 1 ;;
esac
[[ -f "${PERFORMANCE_CHECK_SCRIPT}" ]]
command -v nohup >/dev/null
command -v setsid >/dev/null
mkdir -p -- "$(dirname -- "${PERFORMANCE_RUN_DIR}")"
mkdir -- "${PERFORMANCE_RUN_DIR}"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH="${PERFORMANCE_PROJECT_ROOT}/src" PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
nohup setsid "${PERFORMANCE_PYTHON_BIN}" -u scripts/detached_task.py run "${PERFORMANCE_RUN_DIR}" -- \
  "${PERFORMANCE_PYTHON_BIN}" -u "${PERFORMANCE_CHECK_SCRIPT}" \
  --experiment "${PERFORMANCE_EXPERIMENT_ROOT}" --output "${PERFORMANCE_RUN_DIR}/artifacts" \
  --timeout-seconds "${PERFORMANCE_TIMEOUT_SECONDS}" \
  </dev/null >"${PERFORMANCE_RUN_DIR}/task.log" 2>&1 &
for ((attempt=0; attempt<50; attempt++)); do
  [[ -f "${PERFORMANCE_RUN_DIR}/task.json" ]] && break
  [[ -f "${PERFORMANCE_RUN_DIR}/completion.json" ]] && break
  sleep 0.1
done
[[ -f "${PERFORMANCE_RUN_DIR}/task.json" ]] || { cat "${PERFORMANCE_RUN_DIR}/task.log" >&2; exit 1; }
"${PERFORMANCE_PYTHON_BIN}" scripts/detached_task.py status "${PERFORMANCE_RUN_DIR}"
printf '查看日志：tail -n 100 -F %q\n' "${PERFORMANCE_RUN_DIR}/task.log"
printf '查看状态：%q %q status %q\n' "${PERFORMANCE_PYTHON_BIN}" "${PERFORMANCE_PROJECT_ROOT}/scripts/detached_task.py" "${PERFORMANCE_RUN_DIR}"
printf '停止任务：%q %q stop %q\n' "${PERFORMANCE_PYTHON_BIN}" "${PERFORMANCE_PROJECT_ROOT}/scripts/detached_task.py" "${PERFORMANCE_RUN_DIR}"
printf '查看日志时按 Ctrl+C 只退出查看，不会停止后台任务。\n'
