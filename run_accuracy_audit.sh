#!/usr/bin/env bash
set -euo pipefail
# ============================== 参数区 ==============================
AUDIT_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
AUDIT_PYTHON_BIN="${AUDIT_PYTHON_BIN:-${AUDIT_PROJECT_ROOT}/.sionna-venv/bin/python}"
AUDIT_EXPERIMENT_ROOT="${AUDIT_EXPERIMENT_ROOT:-${AUDIT_PROJECT_ROOT}/outputs/diffraction_boundary_v2}"
AUDIT_RUN_DIR="${AUDIT_RUN_DIR:-${AUDIT_PROJECT_ROOT}/outputs/accuracy_audit_$(date -u +%Y%m%dT%H%M%S)_$$}"
AUDIT_COVERAGE_RADIUS_M="${AUDIT_COVERAGE_RADIUS_M:-2.0}"
AUDIT_UE_IDS="${AUDIT_UE_IDS:-}"
AUDIT_TRUTH_SEED_DIAGNOSTIC="${AUDIT_TRUTH_SEED_DIAGNOSTIC:-1}"
# stages：全量阶段回溯；reference_bias：保存观测不变，仅评估参考偏差影响。
AUDIT_MODE="${AUDIT_MODE:-stages}"
AUDIT_SCENE_JSON="${AUDIT_SCENE_JSON:-${AUDIT_PROJECT_ROOT}/outputs/spectrum_experiment_20260908T091621_2543248/UE003/channel/scene/scene_2d.json}"
# 仅在评估侧读取保存的结果；CPU 执行，不使用 GPU，不修改原实验。
# ====================================================================
AUDIT_ACTION="${1:-start}"
if [[ "${AUDIT_ACTION}" != start ]]; then
  [[ $# -eq 2 ]] || { echo "用法：$0 log|status|stop /绝对运行目录" >&2; exit 2; }
  case "${AUDIT_ACTION}" in
    log) exec tail -n 100 -F "${2}/task.log" ;;
    status|stop) exec "${AUDIT_PYTHON_BIN}" "${AUDIT_PROJECT_ROOT}/scripts/detached_task.py" "${AUDIT_ACTION}" "$2" ;;
    *) echo "未知操作：${AUDIT_ACTION}" >&2; exit 2 ;;
  esac
fi
cd "${AUDIT_PROJECT_ROOT}"
[[ -x "${AUDIT_PYTHON_BIN}" && -f "${AUDIT_EXPERIMENT_ROOT}/pilot/trials.jsonl" ]]
[[ "${AUDIT_TRUTH_SEED_DIAGNOSTIC}" =~ ^[01]$ ]]
command -v nohup >/dev/null
command -v setsid >/dev/null
mkdir -p -- "$(dirname -- "${AUDIT_RUN_DIR}")"
mkdir -- "${AUDIT_RUN_DIR}"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH="${AUDIT_PROJECT_ROOT}/src" PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
AUDIT_ARGS=(--experiment "${AUDIT_EXPERIMENT_ROOT}" --output "${AUDIT_RUN_DIR}/artifacts"
            --coverage-radius-m "${AUDIT_COVERAGE_RADIUS_M}" --ue-ids "${AUDIT_UE_IDS}")
[[ "${AUDIT_TRUTH_SEED_DIAGNOSTIC}" == 0 ]] || AUDIT_ARGS+=(--truth-seed-diagnostic)
AUDIT_SCRIPT="scripts/audit_boundary_accuracy.py"
case "${AUDIT_MODE}" in
  stages) ;;
  reference_bias)
    [[ -f "${AUDIT_SCENE_JSON}" ]]
    AUDIT_SCRIPT="scripts/check_accuracy_reference_bias.py"
    AUDIT_ARGS=(--experiment "${AUDIT_EXPERIMENT_ROOT}" --output "${AUDIT_RUN_DIR}/artifacts" --scene "${AUDIT_SCENE_JSON}")
    [[ -z "${AUDIT_UE_IDS}" ]] || AUDIT_ARGS+=(--ue-ids "${AUDIT_UE_IDS}")
    ;;
  *) echo "AUDIT_MODE 必须为 stages 或 reference_bias" >&2; exit 2 ;;
esac
nohup setsid "${AUDIT_PYTHON_BIN}" -u scripts/detached_task.py run "${AUDIT_RUN_DIR}" -- \
  "${AUDIT_PYTHON_BIN}" -u "${AUDIT_SCRIPT}" "${AUDIT_ARGS[@]}" \
  </dev/null >"${AUDIT_RUN_DIR}/task.log" 2>&1 &
for ((attempt=0; attempt<50; attempt++)); do
  [[ -f "${AUDIT_RUN_DIR}/task.json" || -f "${AUDIT_RUN_DIR}/completion.json" ]] && break
  sleep 0.1
done
[[ -f "${AUDIT_RUN_DIR}/task.json" ]] || { cat "${AUDIT_RUN_DIR}/task.log" >&2; exit 1; }
"${AUDIT_PYTHON_BIN}" scripts/detached_task.py status "${AUDIT_RUN_DIR}"
printf '查看日志：tail -n 100 -F %q\n' "${AUDIT_RUN_DIR}/task.log"
printf '查看状态：bash %q status %q\n' "${AUDIT_PROJECT_ROOT}/run_accuracy_audit.sh" "${AUDIT_RUN_DIR}"
printf '停止任务：bash %q stop %q\n' "${AUDIT_PROJECT_ROOT}/run_accuracy_audit.sh" "${AUDIT_RUN_DIR}"
printf '查看日志时按 Ctrl+C 只退出查看，不会停止后台任务。\n'
