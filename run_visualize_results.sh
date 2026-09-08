#!/usr/bin/env bash
set -euo pipefail

# ============================ 用户参数区 ============================
PROJECT_ROOT="${PROJECT_ROOT:-/data/zhujun/differt_projects/time-bias-correct}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhujun/miniconda3/bin/python3}"
# 每个 UE 单独输出到 samples/UE编号/repeat编号/00–08步骤。
# 汇总 CDF、Med、P90 在 summary/。EXPERIMENT_ROOT 非空时读取整个批量实验。
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/deepmimo_sionna_smoke_run11}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-}"
# 1：只生成汇总图表；0：同时导出全部 sample 的逐步骤图。
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"
# 默认每次新建目录；显式指定的目录也必须尚不存在。
REPORT_ROOT="${REPORT_ROOT:-${PROJECT_ROOT}/outputs/step_report_$(date -u +%Y%m%dT%H%M%S)_$$}"
# ===================================================================

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
cd "${PROJECT_ROOT}"
if [[ -n "${EXPERIMENT_ROOT}" ]]; then
  INPUT_ARGS=(--experiment "${EXPERIMENT_ROOT}")
else
  INPUT_ARGS=(--run-root "${RUN_ROOT}")
fi
if [[ "${SUMMARY_ONLY}" == "1" ]]; then
  INPUT_ARGS+=(--summary-only)
elif [[ "${SUMMARY_ONLY}" != "0" ]]; then
  echo "SUMMARY_ONLY 只能为 0 或 1" >&2
  exit 2
fi
exec "${PYTHON_BIN}" -m time_bias_localization.visualization \
  "${INPUT_ARGS[@]}" --output "${REPORT_ROOT}"
