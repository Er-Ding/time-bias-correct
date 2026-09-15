#!/usr/bin/env bash
set -euo pipefail
# ============================== 参数区 ==============================
CONTINUOUS_CHECK_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
CONTINUOUS_CHECK_PYTHON_BIN="${CONTINUOUS_CHECK_PYTHON_BIN:-${CONTINUOUS_CHECK_PROJECT_ROOT}/.sionna-venv/bin/python}"
# 输入为连续传播定位专用配置；仅执行 CPU 检查。
CONTINUOUS_CHECK_CONFIG="${CONTINUOUS_CHECK_CONFIG:-${CONTINUOUS_CHECK_PROJECT_ROOT}/configs/continuous_model_v1.yaml}"
CONTINUOUS_CHECK_TEST_SCOPE="${CONTINUOUS_CHECK_TEST_SCOPE:-focused}"
CONTINUOUS_CHECK_OUTPUT_ROOT="${CONTINUOUS_CHECK_OUTPUT_ROOT:-${CONTINUOUS_CHECK_PROJECT_ROOT}/outputs/continuous_model_check_$(date -u +%Y%m%dT%H%M%S)_$$}"
CONTINUOUS_CHECK_CPU_THREADS="${CONTINUOUS_CHECK_CPU_THREADS:-1}"
# 本脚本仅使用 CPU，不申请 GPU。
# ====================================================================
CONTINUOUS_CHECK_ACTION="${1:-start}"
if [[ "${CONTINUOUS_CHECK_ACTION}" != start ]]; then
  [[ $# -eq 2 ]] || { echo "用法：$0 log|status|stop /绝对输出目录" >&2; exit 2; }
  CONTINUOUS_CHECK_MANAGEMENT_ROOT="$(realpath -- "$2")"
  [[ -f "${CONTINUOUS_CHECK_MANAGEMENT_ROOT}/latest_run.txt" ]] || { echo "没有运行记录" >&2; exit 1; }
  IFS= read -r CONTINUOUS_CHECK_MANAGEMENT_RUN <"${CONTINUOUS_CHECK_MANAGEMENT_ROOT}/latest_run.txt"
  case "${CONTINUOUS_CHECK_ACTION}" in
    log) exec tail -n 100 -F "${CONTINUOUS_CHECK_MANAGEMENT_RUN}/task.log" ;;
    status|stop) exec "${CONTINUOUS_CHECK_PYTHON_BIN}" "${CONTINUOUS_CHECK_PROJECT_ROOT}/scripts/detached_task.py" "${CONTINUOUS_CHECK_ACTION}" "${CONTINUOUS_CHECK_MANAGEMENT_RUN}" ;;
    *) echo "动作只能是 start、log、status 或 stop" >&2; exit 2 ;;
  esac
fi
[[ $# -le 1 ]] || { echo 'start 不接受额外位置参数，请在参数区或环境变量设置。' >&2; exit 2; }
cd "${CONTINUOUS_CHECK_PROJECT_ROOT}"
[[ -x "${CONTINUOUS_CHECK_PYTHON_BIN}" ]]
[[ "${CONTINUOUS_CHECK_CPU_THREADS}" =~ ^[1-9][0-9]*$ ]]
[[ -f "${CONTINUOUS_CHECK_CONFIG}" ]]
[[ "${CONTINUOUS_CHECK_TEST_SCOPE}" == all || "${CONTINUOUS_CHECK_TEST_SCOPE}" == focused ]]
command -v nohup >/dev/null
command -v setsid >/dev/null
mkdir -p -- "$(dirname -- "${CONTINUOUS_CHECK_OUTPUT_ROOT}")"
mkdir -- "${CONTINUOUS_CHECK_OUTPUT_ROOT}"
CONTINUOUS_CHECK_OUTPUT_ROOT="$(realpath -- "${CONTINUOUS_CHECK_OUTPUT_ROOT}")"
CONTINUOUS_CHECK_RUN_DIR="${CONTINUOUS_CHECK_OUTPUT_ROOT}/run_records/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p -- "${CONTINUOUS_CHECK_RUN_DIR}"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH="${CONTINUOUS_CHECK_PROJECT_ROOT}/src" PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${CONTINUOUS_CHECK_CPU_THREADS}" OMP_NUM_THREADS="${CONTINUOUS_CHECK_CPU_THREADS}" MKL_NUM_THREADS="${CONTINUOUS_CHECK_CPU_THREADS}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export MPLCONFIGDIR="${CONTINUOUS_CHECK_OUTPUT_ROOT}/matplotlib_cache"
nohup setsid "${CONTINUOUS_CHECK_PYTHON_BIN}" -u scripts/detached_task.py run "${CONTINUOUS_CHECK_RUN_DIR}" -- \
  "${CONTINUOUS_CHECK_PYTHON_BIN}" -u scripts/check_continuous_model.py \
  --output "${CONTINUOUS_CHECK_OUTPUT_ROOT}/validation" --config "${CONTINUOUS_CHECK_CONFIG}" --scope "${CONTINUOUS_CHECK_TEST_SCOPE}" \
  </dev/null >"${CONTINUOUS_CHECK_RUN_DIR}/task.log" 2>&1 &
for ((attempt=0; attempt<50; attempt++)); do
  [[ -f "${CONTINUOUS_CHECK_RUN_DIR}/task.json" || -f "${CONTINUOUS_CHECK_RUN_DIR}/completion.json" ]] && break
  sleep 0.1
done
[[ -f "${CONTINUOUS_CHECK_RUN_DIR}/task.json" ]] || { cat "${CONTINUOUS_CHECK_RUN_DIR}/task.log" >&2; exit 1; }
printf '%s\n' "${CONTINUOUS_CHECK_RUN_DIR}" >"${CONTINUOUS_CHECK_OUTPUT_ROOT}/latest_run.txt"
"${CONTINUOUS_CHECK_PYTHON_BIN}" scripts/detached_task.py status "${CONTINUOUS_CHECK_RUN_DIR}"
head -n 8 "${CONTINUOUS_CHECK_RUN_DIR}/task.log"
"${CONTINUOUS_CHECK_PYTHON_BIN}" - "${CONTINUOUS_CHECK_RUN_DIR}" <<'PYVERIFY'
import json
from pathlib import Path
import sys
from scripts.detached_task import identity
record_dir = Path(sys.argv[1])
record = json.loads((record_dir / "task.json").read_text())
completion = record_dir / "completion.json"
if completion.is_file():
    result = json.loads(completion.read_text())
    if result["exit_code"] != 0:
        raise SystemExit("后台任务启动后失败；请查看 task.log。")
    print("任务已经完成，退出码为 0。")
elif identity(record["task"]["pid"]) == record["task"]:
    print(f"已核对实际任务 PID={record['task']['pid']}，任务正在后台运行。")
else:
    raise SystemExit("实际任务已退出，结束记录尚未写入；请查询状态，不能判为成功。")
PYVERIFY
printf '实时日志：tail -n 100 -F %q\n' "${CONTINUOUS_CHECK_RUN_DIR}/task.log"
printf '查看状态：bash %q status %q\n' "${CONTINUOUS_CHECK_PROJECT_ROOT}/run_continuous_model_check.sh" "${CONTINUOUS_CHECK_OUTPUT_ROOT}"
printf '停止任务：bash %q stop %q\n' "${CONTINUOUS_CHECK_PROJECT_ROOT}/run_continuous_model_check.sh" "${CONTINUOUS_CHECK_OUTPUT_ROOT}"
printf 'Ctrl+C 只退出日志查看，不会停止后台任务。\n'
