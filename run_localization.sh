#!/usr/bin/env bash
set -euo pipefail

# ======================== 用户参数区 ========================
PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
PYTHON_BIN="${PYTHON_BIN:-/home/zhujun/miniconda3/bin/python3}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/offline_demo_localization.yaml}"
SCENE_JSON="${SCENE_JSON:-${PROJECT_ROOT}/outputs/offline_demo/scene/scene_2d.json}"
ONLINE_INPUT="${ONLINE_INPUT:-${PROJECT_ROOT}/outputs/offline_demo/data/online/measurement.npz}"
GENERATION_MANIFEST="${GENERATION_MANIFEST:-${PROJECT_ROOT}/outputs/offline_demo/data/generation_manifest.json}"
# 留空时自动使用本次时间和进程号生成唯一文件名；也可显式指定新路径。
# RUN_RECEIPT_JSON 与评估脚本同名，便于先导出一次再顺序执行两个脚本。
RUN_RECEIPT_JSON="${RUN_RECEIPT_JSON:-}"
RECEIPT_JSON="${RECEIPT_JSON:-${RUN_RECEIPT_JSON}}"
# ===========================================================

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "找不到可执行的 Python：${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${SCENE_JSON}" ]]; then
  echo "缺少二维场景：${SCENE_JSON}" >&2
  exit 2
fi
if [[ ! -f "${ONLINE_INPUT}" ]]; then
  echo "缺少在线 CSI：${ONLINE_INPUT}" >&2
  exit 2
fi
if [[ ! -f "${GENERATION_MANIFEST}" ]]; then
  echo "缺少生成清单：${GENERATION_MANIFEST}" >&2
  exit 2
fi
OUTPUT_ROOT="$("${PYTHON_BIN}" -c \
  'import sys; from time_bias_localization.config import load_localization_config; from time_bias_localization.pipeline import resolve_output_root; print(resolve_output_root(load_localization_config(sys.argv[1])))' \
  "${CONFIG_PATH}")"
if [[ -z "${RECEIPT_JSON}" ]]; then
  RECEIPT_JSON="${OUTPUT_ROOT}/receipts/localization_$(date -u +%Y%m%dT%H%M%S.%NZ)_$$.json"
elif [[ "${RECEIPT_JSON}" != /* ]]; then
  RECEIPT_JSON="${PROJECT_ROOT}/${RECEIPT_JSON}"
fi
RECEIPT_JSON="$(readlink -m "${RECEIPT_JSON}")"
if [[ -n "${RUN_RECEIPT_JSON}" ]]; then
  if [[ "${RUN_RECEIPT_JSON}" != /* ]]; then
    RUN_RECEIPT_JSON="${PROJECT_ROOT}/${RUN_RECEIPT_JSON}"
  fi
  RUN_RECEIPT_JSON="$(readlink -m "${RUN_RECEIPT_JSON}")"
  if [[ "${RUN_RECEIPT_JSON}" != "${RECEIPT_JSON}" ]]; then
    echo "RECEIPT_JSON 与 RUN_RECEIPT_JSON 指向不同文件，已停止。" >&2
    exit 2
  fi
fi
if [[ -e "${RECEIPT_JSON}" ]]; then
  echo "运行回执目标已存在，拒绝覆盖：${RECEIPT_JSON}" >&2
  exit 2
fi
echo "生成清单：${GENERATION_MANIFEST}"
echo "本次运行回执：${RECEIPT_JSON}"
echo "后续评估请设置：RUN_RECEIPT_JSON=${RECEIPT_JSON}"
exec "${PYTHON_BIN}" -m time_bias_localization.cli localize \
  --config "${CONFIG_PATH}" \
  --scene-json "${SCENE_JSON}" \
  --online-input "${ONLINE_INPUT}" \
  --generation-manifest "${GENERATION_MANIFEST}" \
  --run-receipt "${RECEIPT_JSON}"
