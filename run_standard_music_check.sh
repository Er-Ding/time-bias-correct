#!/usr/bin/env bash
set -euo pipefail

# ============================== 参数区 ==============================
STANDARD_MUSIC_PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
STANDARD_MUSIC_PYTHON_BIN="${STANDARD_MUSIC_PYTHON_BIN:-${STANDARD_MUSIC_PROJECT_ROOT}/.sionna-venv/bin/python}"
# v5：自动信号维数 + 标准 MUSIC 直接读峰；关闭后置 CSI 残差验收。
STANDARD_MUSIC_CONFIG="${STANDARD_MUSIC_CONFIG:-${STANDARD_MUSIC_PROJECT_ROOT}/configs/diffraction_boundary_generation_v5.yaml}"
# all 运行完整 CPU 回归；focused 只运行标准 MUSIC 及其直接相关回归。
STANDARD_MUSIC_TEST_SCOPE="${STANDARD_MUSIC_TEST_SCOPE:-all}"
STANDARD_MUSIC_OUTPUT_ROOT="${STANDARD_MUSIC_OUTPUT_ROOT:-${STANDARD_MUSIC_PROJECT_ROOT}/outputs/standard_music_check_$(date -u +%Y%m%dT%H%M%S)_$$}"
STANDARD_MUSIC_CPU_THREADS="${STANDARD_MUSIC_CPU_THREADS:-1}"
# 日志：<输出目录>/run_records/<启动时间_编号>/task.log。
# 同目录保存实际任务 PID、启动参数、开始/结束时间及退出码。
# 本任务仅使用 CPU，不使用 GPU；每次必须使用新的输出目录。
# ====================================================================

export MUSIC_SUBSPACE_PYTHON_BIN="${STANDARD_MUSIC_PYTHON_BIN}"
export MUSIC_SUBSPACE_CONFIG="${STANDARD_MUSIC_CONFIG}"
export MUSIC_SUBSPACE_TEST_SCOPE="${STANDARD_MUSIC_TEST_SCOPE}"
export MUSIC_SUBSPACE_OUTPUT_ROOT="${STANDARD_MUSIC_OUTPUT_ROOT}"
export MUSIC_SUBSPACE_CPU_THREADS="${STANDARD_MUSIC_CPU_THREADS}"
# 共用现有 nohup + setsid 启动、日志、状态与整组任务停止逻辑。
exec bash "${STANDARD_MUSIC_PROJECT_ROOT}/run_music_subspace_check.sh" "$@"
