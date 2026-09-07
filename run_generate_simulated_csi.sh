#!/usr/bin/env bash
set -euo pipefail

# ======================== 用户参数区 ========================
PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
PYTHON_BIN="${PYTHON_BIN:-/home/zhujun/miniconda3/bin/python3}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/offline_demo.yaml}"
SCENE_JSON="${SCENE_JSON:-${PROJECT_ROOT}/outputs/offline_demo/scene/scene_2d.json}"
# ===========================================================

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"
if [[ ! -f "${SCENE_JSON}" ]]; then
  echo "缺少二维场景：${SCENE_JSON}" >&2
  echo "请先执行 ${PROJECT_ROOT}/run_preprocess_scene.sh" >&2
  exit 2
fi
exec "${PYTHON_BIN}" -m time_bias_localization.cli generate-data \
  --config "${CONFIG_PATH}" \
  --scene-json "${SCENE_JSON}"

