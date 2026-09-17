#!/usr/bin/env bash
set -euo pipefail
# ============================== 参数区 ==============================
MC_CHECK_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
MC_CHECK_OUTPUT_ROOT="${MC_CHECK_OUTPUT_ROOT:-${MC_CHECK_PROJECT_ROOT}/outputs/monte_carlo_check_$(date -u +%Y%m%dT%H%M%S)_$$}"
# ====================================================================
export MC_MODE=check MC_OUTPUT_ROOT="${MC_CHECK_OUTPUT_ROOT}"
exec bash "${MC_CHECK_PROJECT_ROOT}/run_monte_carlo_experiment.sh" "$@"
