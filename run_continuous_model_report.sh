#!/usr/bin/env bash
set -euo pipefail
# ============================== 参数区 ==============================
CONTINUOUS_REPORT_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
CONTINUOUS_REPORT_PYTHON_BIN="${CONTINUOUS_REPORT_PYTHON_BIN:-${CONTINUOUS_REPORT_PROJECT_ROOT}/.sionna-venv/bin/python}"
# comparison：含 comparison_plan.json 的对照目录；run：含 localization 的单次目录。
CONTINUOUS_REPORT_INPUT_KIND="${CONTINUOUS_REPORT_INPUT_KIND:-comparison}"
CONTINUOUS_REPORT_INPUT_ROOT="${CONTINUOUS_REPORT_INPUT_ROOT:-}"
CONTINUOUS_REPORT_OUTPUT_ROOT="${CONTINUOUS_REPORT_OUTPUT_ROOT:-${CONTINUOUS_REPORT_PROJECT_ROOT}/outputs/continuous_model_report_$(date -u +%Y%m%dT%H%M%S)_$$}"
# 1：仅读取公开数据和定位结果，完全不打开评价真值；0：核对已有可选评价。
CONTINUOUS_REPORT_SKIP_EVALUATION="${CONTINUOUS_REPORT_SKIP_EVALUATION:-1}"
CONTINUOUS_REPORT_SUMMARY_ONLY="${CONTINUOUS_REPORT_SUMMARY_ONLY:-0}"
CONTINUOUS_REPORT_CPU_THREADS="${CONTINUOUS_REPORT_CPU_THREADS:-1}"
# ====================================================================
CONTINUOUS_REPORT_ACTION="${1:-start}"
if [[ "${CONTINUOUS_REPORT_ACTION}" != start ]]; then
  [[ $# -eq 2 ]] || { echo "用法：$0 log|status|stop /绝对输出目录" >&2; exit 2; }
  CONTINUOUS_REPORT_MANAGEMENT_ROOT="$(realpath -- "$2")"
  [[ -f "${CONTINUOUS_REPORT_MANAGEMENT_ROOT}/latest_run.txt" ]] || { echo "没有运行记录" >&2; exit 1; }
  IFS= read -r CONTINUOUS_REPORT_MANAGEMENT_RUN <"${CONTINUOUS_REPORT_MANAGEMENT_ROOT}/latest_run.txt"
  case "${CONTINUOUS_REPORT_ACTION}" in
    log) exec tail -n 100 -F "${CONTINUOUS_REPORT_MANAGEMENT_RUN}/task.log" ;;
    status|stop) exec "${CONTINUOUS_REPORT_PYTHON_BIN}" "${CONTINUOUS_REPORT_PROJECT_ROOT}/scripts/detached_task.py" "${CONTINUOUS_REPORT_ACTION}" "${CONTINUOUS_REPORT_MANAGEMENT_RUN}" ;;
    *) echo "动作只能是 start、log、status 或 stop" >&2; exit 2 ;;
  esac
fi
[[ $# -le 1 ]] || { echo "start 不接受额外位置参数，请在参数区或环境变量设置。" >&2; exit 2; }
[[ -n "${CONTINUOUS_REPORT_INPUT_ROOT}" ]] || { echo "请在参数区或 CONTINUOUS_REPORT_INPUT_ROOT 指定已有结果的绝对路径。" >&2; exit 2; }
cd "${CONTINUOUS_REPORT_PROJECT_ROOT}"
[[ -x "${CONTINUOUS_REPORT_PYTHON_BIN}" ]]
[[ "${CONTINUOUS_REPORT_CPU_THREADS}" =~ ^[1-9][0-9]*$ ]]
[[ "${CONTINUOUS_REPORT_SKIP_EVALUATION}" =~ ^[01]$ && "${CONTINUOUS_REPORT_SUMMARY_ONLY}" =~ ^[01]$ ]]
CONTINUOUS_REPORT_INPUT_ROOT="$(realpath -- "${CONTINUOUS_REPORT_INPUT_ROOT}")"
CONTINUOUS_REPORT_ARGUMENTS=()
case "${CONTINUOUS_REPORT_INPUT_KIND}" in
  comparison)
    [[ -f "${CONTINUOUS_REPORT_INPUT_ROOT}/comparison_plan.json" && -f "${CONTINUOUS_REPORT_INPUT_ROOT}/trials.json" ]]
    CONTINUOUS_REPORT_ARGUMENTS+=(--experiment "${CONTINUOUS_REPORT_INPUT_ROOT}") ;;
  run)
    [[ -f "${CONTINUOUS_REPORT_INPUT_ROOT}/localization/frozen_input_manifest.json" ]]
    CONTINUOUS_REPORT_ARGUMENTS+=(--run-root "${CONTINUOUS_REPORT_INPUT_ROOT}") ;;
  *) echo "输入类型只能是 comparison 或 run" >&2; exit 2 ;;
esac
[[ "${CONTINUOUS_REPORT_SKIP_EVALUATION}" == 0 ]] || CONTINUOUS_REPORT_ARGUMENTS+=(--skip-evaluation)
[[ "${CONTINUOUS_REPORT_SUMMARY_ONLY}" == 0 ]] || CONTINUOUS_REPORT_ARGUMENTS+=(--summary-only)
command -v nohup >/dev/null
command -v setsid >/dev/null
"${CONTINUOUS_REPORT_PYTHON_BIN}" - "${CONTINUOUS_REPORT_INPUT_ROOT}" "${CONTINUOUS_REPORT_OUTPUT_ROOT}" <<'PYPATHS'
from pathlib import Path
import sys
source, output = (Path(value).resolve() for value in sys.argv[1:])
if source == output or source in output.parents or output in source.parents:
    raise SystemExit("报告输出必须与只读输入分开，拒绝创建目录。")
PYPATHS
mkdir -p -- "$(dirname -- "${CONTINUOUS_REPORT_OUTPUT_ROOT}")"
mkdir -- "${CONTINUOUS_REPORT_OUTPUT_ROOT}"
CONTINUOUS_REPORT_OUTPUT_ROOT="$(realpath -- "${CONTINUOUS_REPORT_OUTPUT_ROOT}")"
CONTINUOUS_REPORT_RUN_DIR="${CONTINUOUS_REPORT_OUTPUT_ROOT}/run_records/$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p -- "${CONTINUOUS_REPORT_RUN_DIR}"
export CUDA_VISIBLE_DEVICES='' PYTHONPATH="${CONTINUOUS_REPORT_PROJECT_ROOT}/src" PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS="${CONTINUOUS_REPORT_CPU_THREADS}" OMP_NUM_THREADS="${CONTINUOUS_REPORT_CPU_THREADS}" MKL_NUM_THREADS="${CONTINUOUS_REPORT_CPU_THREADS}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export MPLCONFIGDIR="${CONTINUOUS_REPORT_OUTPUT_ROOT}/matplotlib_cache"
nohup setsid "${CONTINUOUS_REPORT_PYTHON_BIN}" -u scripts/detached_task.py run "${CONTINUOUS_REPORT_RUN_DIR}" -- \
  "${CONTINUOUS_REPORT_PYTHON_BIN}" -u -m time_bias_localization.visualization \
  "${CONTINUOUS_REPORT_ARGUMENTS[@]}" --output "${CONTINUOUS_REPORT_OUTPUT_ROOT}/report" \
  </dev/null >"${CONTINUOUS_REPORT_RUN_DIR}/task.log" 2>&1 &
for ((attempt=0; attempt<50; attempt++)); do
  [[ -f "${CONTINUOUS_REPORT_RUN_DIR}/task.json" || -f "${CONTINUOUS_REPORT_RUN_DIR}/completion.json" ]] && break
  sleep 0.1
done
[[ -f "${CONTINUOUS_REPORT_RUN_DIR}/task.json" ]] || { cat "${CONTINUOUS_REPORT_RUN_DIR}/task.log" >&2; exit 1; }
printf '%s\n' "${CONTINUOUS_REPORT_RUN_DIR}" >"${CONTINUOUS_REPORT_OUTPUT_ROOT}/latest_run.txt"
"${CONTINUOUS_REPORT_PYTHON_BIN}" scripts/detached_task.py status "${CONTINUOUS_REPORT_RUN_DIR}"
head -n 8 "${CONTINUOUS_REPORT_RUN_DIR}/task.log"
"${CONTINUOUS_REPORT_PYTHON_BIN}" - "${CONTINUOUS_REPORT_RUN_DIR}" <<'PYVERIFY'
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
        raise SystemExit("后台报告任务启动后失败；请查看 task.log。")
    print("报告任务已经完成，退出码为 0。")
elif identity(record["task"]["pid"]) == record["task"]:
    print(f"已核对实际任务 PID={record['task']['pid']}，报告正在后台生成。")
else:
    raise SystemExit("实际任务已退出但没有结束记录，不能判为成功。")
PYVERIFY
printf '报告目录：%s\n' "${CONTINUOUS_REPORT_OUTPUT_ROOT}/report"
printf '实时日志：tail -n 100 -F %q\n' "${CONTINUOUS_REPORT_RUN_DIR}/task.log"
printf '查看状态：bash %q status %q\n' "${CONTINUOUS_REPORT_PROJECT_ROOT}/run_continuous_model_report.sh" "${CONTINUOUS_REPORT_OUTPUT_ROOT}"
printf '停止任务：bash %q stop %q\n' "${CONTINUOUS_REPORT_PROJECT_ROOT}/run_continuous_model_report.sh" "${CONTINUOUS_REPORT_OUTPUT_ROOT}"
printf 'Ctrl+C 只退出日志查看，不会停止后台任务。\n'
