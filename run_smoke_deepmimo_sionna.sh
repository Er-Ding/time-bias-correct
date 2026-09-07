#!/usr/bin/env bash
set -euo pipefail

# ============================ 用户参数区 ============================
PROJECT_ROOT="${PROJECT_ROOT:-/data/zhujun/differt_projects/time-bias-correct}"
SIONNA_BASE_ENV="${SIONNA_BASE_ENV:-/data/zhujun/conda_envs/envrecons-differt}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.sionna-venv/bin/python}"
GENERATION_CONFIG_PATH="${GENERATION_CONFIG_PATH:-${PROJECT_ROOT}/configs/deepmimo_sionna_smoke.yaml}"
LOCALIZATION_CONFIG_PATH="${LOCALIZATION_CONFIG_PATH:-${PROJECT_ROOT}/configs/deepmimo_sionna_smoke_localization.yaml}"

# 以下路径留空时，统一从两份配置共同的 output.root 推导。若手工填写
# OUTPUT_ROOT，它必须与两份配置都一致，避免阶段间串数据。
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
SCENE_JSON="${SCENE_JSON:-}"
ONLINE_INPUT="${ONLINE_INPUT:-}"
GENERATION_MANIFEST="${GENERATION_MANIFEST:-}"
TRUTH_NPZ="${TRUTH_NPZ:-}"
RESULT_JSON="${RESULT_JSON:-}"
METRICS_JSON="${METRICS_JSON:-}"
# 留空时按本次时间和进程号创建唯一回执；显式路径也必须是新文件。
RUN_RECEIPT_JSON="${RUN_RECEIPT_JSON:-}"

# 1：已有且通过检查的生成产物可复用；0：重新执行 Sionna/DeepMIMO 生成。
# 无论取何值，定位和评估都会重新执行。
REUSE_GENERATED="${REUSE_GENERATED:-1}"
# ===================================================================

PROJECT_ROOT="$(readlink -m "${PROJECT_ROOT}")"
GENERATION_CONFIG_PATH="$(readlink -m "${GENERATION_CONFIG_PATH}")"
LOCALIZATION_CONFIG_PATH="$(readlink -m "${LOCALIZATION_CONFIG_PATH}")"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "找不到 Sionna Python：${PYTHON_BIN}" >&2
  echo "请先运行 ${PROJECT_ROOT}/setup_sionna_environment.sh" >&2
  exit 2
fi
if [[ ! -f "${GENERATION_CONFIG_PATH}" ]]; then
  echo "找不到数据生成配置：${GENERATION_CONFIG_PATH}" >&2
  exit 2
fi
if [[ ! -f "${LOCALIZATION_CONFIG_PATH}" ]]; then
  echo "找不到定位专用配置：${LOCALIZATION_CONFIG_PATH}" >&2
  exit 2
fi
if [[ "${REUSE_GENERATED}" != "0" && "${REUSE_GENERATED}" != "1" ]]; then
  echo "REUSE_GENERATED 只能是 0 或 1，当前为：${REUSE_GENERATED}" >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${PROJECT_ROOT}/.cache/matplotlib"
export XDG_CACHE_HOME="${PROJECT_ROOT}/.cache/xdg"
export LD_LIBRARY_PATH="${SIONNA_BASE_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

GENERATION_OUTPUT_ROOT="$("${PYTHON_BIN}" -c \
  'import sys; from time_bias_localization.config import load_config; print(load_config(sys.argv[1])["output"]["root"])' \
  "${GENERATION_CONFIG_PATH}")"
LOCALIZATION_OUTPUT_ROOT="$("${PYTHON_BIN}" -c \
  'import sys; from time_bias_localization.config import load_localization_config; print(load_localization_config(sys.argv[1])["output"]["root"])' \
  "${LOCALIZATION_CONFIG_PATH}")"
if [[ "${GENERATION_OUTPUT_ROOT}" != /* ]]; then
  GENERATION_OUTPUT_ROOT="${PROJECT_ROOT}/${GENERATION_OUTPUT_ROOT}"
fi
if [[ "${LOCALIZATION_OUTPUT_ROOT}" != /* ]]; then
  LOCALIZATION_OUTPUT_ROOT="${PROJECT_ROOT}/${LOCALIZATION_OUTPUT_ROOT}"
fi
GENERATION_OUTPUT_ROOT="$(readlink -m "${GENERATION_OUTPUT_ROOT}")"
LOCALIZATION_OUTPUT_ROOT="$(readlink -m "${LOCALIZATION_OUTPUT_ROOT}")"
if [[ "${GENERATION_OUTPUT_ROOT}" != "${LOCALIZATION_OUTPUT_ROOT}" ]]; then
  echo "生成配置与定位配置的 output.root 不一致，已停止。" >&2
  echo "生成输出：${GENERATION_OUTPUT_ROOT}" >&2
  echo "定位输出：${LOCALIZATION_OUTPUT_ROOT}" >&2
  exit 2
fi

"${PYTHON_BIN}" - "${GENERATION_CONFIG_PATH}" "${LOCALIZATION_CONFIG_PATH}" <<'PY'
import sys

from time_bias_localization.config import load_config, load_localization_config

generation = load_config(sys.argv[1])
localization = load_localization_config(sys.argv[2])
for section in ("project", "music", "output"):
    if generation[section] != localization[section]:
        raise SystemExit(f"生成配置与定位配置的 {section} 参数不一致")
for name, value in localization["localization"].items():
    if generation["localization"].get(name) != value:
        raise SystemExit(f"生成配置与定位配置的 localization.{name} 参数不一致")
if "snr_db" in localization["radio"]:
    raise SystemExit("定位配置不得包含用于仿真注噪的 radio.snr_db")
for name, value in localization["radio"].items():
    if name == "bs_position_m":
        continue
    if generation["radio"].get(name) != value:
        raise SystemExit(f"生成配置与定位配置的 radio.{name} 参数不一致")
if localization["radio"].get("bs_position_m") != generation["simulation"].get(
    "bs_position_m"
):
    raise SystemExit(
        "定位配置的 radio.bs_position_m 与生成配置的 simulation.bs_position_m 不一致"
    )
for name in (
    "source",
    "name",
    "fixed_height_m",
    "bev_resolution_m",
    "max_reflections",
    "localization_bounds_m",
):
    if generation["scene"].get(name) != localization["scene"].get(name):
        raise SystemExit(f"生成配置与定位配置的 scene.{name} 不一致")
if "simulation" in localization:
    raise SystemExit("定位配置不得包含 simulation 真值段")
PY

if [[ -z "${OUTPUT_ROOT}" ]]; then
  OUTPUT_ROOT="${GENERATION_OUTPUT_ROOT}"
elif [[ "${OUTPUT_ROOT}" != /* ]]; then
  OUTPUT_ROOT="${PROJECT_ROOT}/${OUTPUT_ROOT}"
fi
OUTPUT_ROOT="$(readlink -m "${OUTPUT_ROOT}")"
if [[ "${OUTPUT_ROOT}" != "${GENERATION_OUTPUT_ROOT}" ]]; then
  echo "OUTPUT_ROOT 与两份配置中的 output.root 不一致，已停止以避免串用产物。" >&2
  echo "OUTPUT_ROOT: ${OUTPUT_ROOT}" >&2
  echo "配置路径: ${GENERATION_OUTPUT_ROOT}" >&2
  exit 2
fi

SCENE_JSON="${SCENE_JSON:-${OUTPUT_ROOT}/scene/scene_2d.json}"
ONLINE_INPUT="${ONLINE_INPUT:-${OUTPUT_ROOT}/data/online/measurement.npz}"
GENERATION_MANIFEST="${GENERATION_MANIFEST:-${OUTPUT_ROOT}/generation_manifest.json}"
TRUTH_NPZ="${TRUTH_NPZ:-${OUTPUT_ROOT}/data/truth/ground_truth.npz}"
RESULT_JSON="${RESULT_JSON:-${OUTPUT_ROOT}/localization/localization_result.json}"
METRICS_JSON="${METRICS_JSON:-${OUTPUT_ROOT}/evaluation/metrics.json}"
if [[ -z "${RUN_RECEIPT_JSON}" ]]; then
  RUN_RECEIPT_JSON="${OUTPUT_ROOT}/receipts/smoke_localization_$(date -u +%Y%m%dT%H%M%S.%NZ)_$$.json"
fi

resolve_project_path() {
  local path_value="$1"
  if [[ "${path_value}" != /* ]]; then
    path_value="${PROJECT_ROOT}/${path_value}"
  fi
  readlink -m "${path_value}"
}

SCENE_JSON="$(resolve_project_path "${SCENE_JSON}")"
ONLINE_INPUT="$(resolve_project_path "${ONLINE_INPUT}")"
GENERATION_MANIFEST="$(resolve_project_path "${GENERATION_MANIFEST}")"
TRUTH_NPZ="$(resolve_project_path "${TRUTH_NPZ}")"
RESULT_JSON="$(resolve_project_path "${RESULT_JSON}")"
METRICS_JSON="$(resolve_project_path "${METRICS_JSON}")"
RUN_RECEIPT_JSON="$(resolve_project_path "${RUN_RECEIPT_JSON}")"
if [[ -e "${RUN_RECEIPT_JSON}" ]]; then
  echo "本次定位回执目标已存在，拒绝覆盖：${RUN_RECEIPT_JSON}" >&2
  exit 2
fi

# localize 子命令固定写入 output.root/localization。这里拒绝另指旧结果，
# 避免本次定位完成后，评估阶段却读到别处残留的 JSON。
EXPECTED_RESULT_JSON="${OUTPUT_ROOT}/localization/localization_result.json"
if [[ "${RESULT_JSON}" != "${EXPECTED_RESULT_JSON}" ]]; then
  echo "RESULT_JSON 必须指向本次配置实际生成的定位结果，已停止以避免读取旧文件。" >&2
  echo "RESULT_JSON: ${RESULT_JSON}" >&2
  echo "应为: ${EXPECTED_RESULT_JSON}" >&2
  exit 2
fi
EXPECTED_METRICS_JSON="${OUTPUT_ROOT}/evaluation/metrics.json"
if [[ "${METRICS_JSON}" != "${EXPECTED_METRICS_JSON}" ]]; then
  echo "METRICS_JSON 必须指向本次运行的固定评估文件，已停止以避免覆盖或串用产物。" >&2
  echo "METRICS_JSON: ${METRICS_JSON}" >&2
  echo "应为: ${EXPECTED_METRICS_JSON}" >&2
  exit 2
fi
require_file() {
  local file_path="$1"
  local description="$2"
  if [[ ! -f "${file_path}" ]]; then
    echo "缺少${description}：${file_path}" >&2
    exit 3
  fi
}

verify_generated_bundle() {
  "${PYTHON_BIN}" - \
    "${GENERATION_MANIFEST}" \
    "${SCENE_JSON}" \
    "${ONLINE_INPUT}" \
    "${TRUTH_NPZ}" \
    "${GENERATION_CONFIG_PATH}" <<'PY'
import json
import math
from pathlib import Path
import sys

import numpy as np

from time_bias_localization.config import load_config
from time_bias_localization.provenance import (
    file_sha256,
    generation_bundle_id,
    localization_config_snapshot,
)

manifest_path, scene_path, online_path, truth_path, config_path = map(
    lambda value: Path(value).expanduser().resolve(), sys.argv[1:]
)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
config = load_config(config_path)
if manifest.get("schema_version") != 2:
    raise SystemExit("生成清单不是当前 schema_version=2，不能安全复用")
declared_bundle_id = manifest.get("bundle_id")
computed_bundle_id = generation_bundle_id(manifest)
if declared_bundle_id != computed_bundle_id:
    raise SystemExit("生成清单的 bundle_id 与三个核心产物哈希不一致")
if manifest.get("localization_scene_geometry_source") != "sionna_exported_triangle_mesh":
    raise SystemExit("生成清单不是原始 Sionna 三角网格版本，不能安全复用")
if manifest.get("absolute_delay_normalization") is not False:
    raise SystemExit("生成清单没有确认保留绝对时延，不能安全复用")
if manifest.get("link_direction") != "uplink_ue_to_bs":
    raise SystemExit("生成清单的链路方向不是 UE 发射、BS 接收")
if manifest.get("scene_consistency", {}).get("passed") is not True:
    raise SystemExit("生成清单没有通过二维场景与反向射线一致性检查")
if manifest.get("sionna_scene") != str(config["scene"]["name"]):
    raise SystemExit("生成清单的 Sionna 场景与当前配置不一致")
if manifest.get("deepmimo_scenario") != str(
    config["simulation"]["deepmimo_scenario_name"]
):
    raise SystemExit("生成清单的 DeepMIMO 场景名与当前配置不一致")
expected_bounds = [float(value) for value in config["scene"]["localization_bounds_m"]]
if manifest.get("bounds_source") != "fixed_config_not_ue_or_truth_paths":
    raise SystemExit("生成清单没有确认定位地图使用固定范围")
if not np.allclose(
    manifest.get("localization_scene_bounds_m", []), expected_bounds, atol=1e-12
):
    raise SystemExit("生成清单的固定定位范围与当前配置不一致")

expected_config = localization_config_snapshot(config)
snapshot_record = manifest.get("config_snapshot", {})
snapshot_path_value = snapshot_record.get("path")
if snapshot_record.get("canonical_sha256") != expected_config["canonical_sha256"]:
    raise SystemExit("生成清单的完整配置哈希与当前配置不一致")
if snapshot_path_value is None:
    raise SystemExit("生成清单没有记录完整配置快照")
snapshot_path = Path(snapshot_path_value).expanduser().resolve()
if not snapshot_path.is_file():
    raise SystemExit(f"生成配置快照不存在：{snapshot_path}")
if snapshot_record.get("file_sha256") != file_sha256(snapshot_path):
    raise SystemExit("生成配置快照的文件哈希不一致")
rt_params = manifest.get("rt_params", {})
expected_rt_params = {
    "max_depth": int(config["scene"]["max_reflections"]),
    "samples_per_src": int(config["simulation"]["samples_per_source"]),
    "max_num_paths_per_src": int(config["simulation"]["max_num_paths_per_source"]),
    "seed": int(config["project"]["random_seed"]),
}
for name, expected_value in expected_rt_params.items():
    if rt_params.get(name) != expected_value:
        raise SystemExit(f"生成清单的射线参数 {name} 与当前配置不一致")

selection = manifest.get("path_selection", {})
expected_selection = {
    "front_facing_only": bool(config["radio"].get("front_facing_only", True)),
    "bs_boresight_rad": math.radians(float(config["radio"]["bs_boresight_deg"])),
    "local_angle_min_rad": math.radians(float(config["music"]["angle_min_deg"])),
    "local_angle_max_rad": math.radians(float(config["music"]["angle_max_deg"])),
}
for name, expected_value in expected_selection.items():
    actual_value = selection.get(name)
    if isinstance(expected_value, bool):
        matches = actual_value is expected_value
    else:
        matches = actual_value is not None and math.isclose(
            float(actual_value), expected_value, rel_tol=0.0, abs_tol=1e-12
        )
    if not matches:
        raise SystemExit(f"生成清单的路径筛选参数 {name} 与当前配置不一致")

recorded = {
    "scene": manifest.get("scene_artifacts", {}).get("scene_json"),
    "online": manifest.get("data_artifacts", {}).get("online_npz"),
    "truth": manifest.get("data_artifacts", {}).get("truth_npz"),
}
expected = {"scene": scene_path, "online": online_path, "truth": truth_path}
for name, expected_path in expected.items():
    recorded_value = recorded[name]
    if recorded_value is None or Path(recorded_value).expanduser().resolve() != expected_path:
        raise SystemExit(f"生成清单中的 {name} 路径与本次参数不一致")
    if not expected_path.is_file():
        raise SystemExit(f"生成清单记录的 {name} 文件不存在：{expected_path}")

hash_records = manifest.get("artifact_hashes", {})
hash_names = {
    "scene": "scene_json",
    "online": "online_measurement",
    "truth": "ground_truth",
}
for name, expected_path in expected.items():
    record = hash_records.get(hash_names[name], {})
    recorded_hash_path = record.get("path")
    if recorded_hash_path is None or Path(recorded_hash_path).expanduser().resolve() != expected_path:
        raise SystemExit(f"生成清单中的 {name} 哈希路径与本次参数不一致")
    if record.get("sha256") != file_sha256(expected_path):
        raise SystemExit(f"生成产物 {name} 的文件哈希不一致，不能安全复用")

scene = json.loads(scene_path.read_text(encoding="utf-8"))
if scene.get("source") != "sionna_exported_triangle_mesh":
    raise SystemExit("scene_2d.json 不是从原始 Sionna 三角网格生成，不能安全复用")
if not np.allclose(scene.get("bounds_m", []), expected_bounds, atol=1e-12):
    raise SystemExit("二维场景范围不是当前配置预先固定的研究区域")
if not math.isclose(
    float(scene["fixed_height_m"]),
    float(config["scene"]["fixed_height_m"]),
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit("二维场景高度与当前配置不一致")
if not math.isclose(
    float(scene["bev_resolution_m"]),
    float(config["scene"]["bev_resolution_m"]),
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit("二维场景分辨率与当前配置不一致")

with np.load(online_path) as online:
    expected_fields = {
        "csi_observed",
        "subcarrier_frequencies_hz",
        "carrier_frequency_hz",
        "antenna_spacing_m",
        "bs_position_m",
        "bs_boresight_rad",
    }
    actual_fields = set(online.files)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unexpected = sorted(actual_fields - expected_fields)
        raise SystemExit(
            "在线 CSI 字段不符合严格六字段契约："
            f"缺少={missing}，额外={unexpected}"
        )
    csi = online["csi_observed"]
    frequencies = online["subcarrier_frequencies_hz"]
    if csi.shape[-2:] != (
        int(config["radio"]["num_bs_antennas"]),
        int(config["radio"]["num_subcarriers"]),
    ):
        raise SystemExit("在线 CSI 的阵元数或子载波数与当前配置不一致")
    expected_spacing = float(config["radio"]["bandwidth_hz"]) / frequencies.size
    if frequencies.size < 2 or not np.allclose(
        np.diff(frequencies), expected_spacing, rtol=1e-12, atol=1e-9
    ):
        raise SystemExit("在线 CSI 的子载波间隔与当前配置不一致")
    if not np.allclose(
        online["bs_position_m"], config["simulation"]["bs_position_m"], atol=1e-9
    ):
        raise SystemExit("在线 CSI 的 BS 坐标与当前配置不一致")
    if not math.isclose(
        float(online["carrier_frequency_hz"]),
        float(config["radio"]["carrier_hz"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise SystemExit("在线 CSI 的载频与当前配置不一致")
    expected_antenna_spacing_m = (
        299792458.0
        / float(config["radio"]["carrier_hz"])
        * float(config["radio"]["antenna_spacing_wavelength"])
    )
    if not math.isclose(
        float(online["antenna_spacing_m"]),
        expected_antenna_spacing_m,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SystemExit("在线 CSI 的阵元间距与当前配置不一致")
    if not math.isclose(
        float(online["bs_boresight_rad"]),
        math.radians(float(config["radio"]["bs_boresight_deg"])),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SystemExit("在线 CSI 的 BS 朝向与当前配置不一致")

with np.load(truth_path) as truth:
    if not np.allclose(
        truth["ue_position_m"], config["simulation"]["ue_position_m"], atol=1e-9
    ):
        raise SystemExit("评估真值中的 UE 坐标与当前配置不一致")
    if not math.isclose(
        float(truth["clock_bias_s"]),
        float(config["simulation"]["clock_bias_s"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise SystemExit("评估真值中的公共时延偏差与当前配置不一致")
PY
}

verify_or_stop() {
  if ! verify_generated_bundle; then
    echo "已有生成产物不能与当前配置安全复用。" >&2
    echo "请保留旧目录，在配置中换一个新的 output.root 和 DeepMIMO 场景名后重跑。" >&2
    exit 4
  fi
}

cd "${PROJECT_ROOT}"
echo "项目目录：${PROJECT_ROOT}"
echo "Python：${PYTHON_BIN}"
echo "数据生成配置：${GENERATION_CONFIG_PATH}"
echo "定位专用配置：${LOCALIZATION_CONFIG_PATH}"
echo "生成清单：${GENERATION_MANIFEST}"
echo "输出目录：${OUTPUT_ROOT}"
echo "复用生成产物：${REUSE_GENERATED}"

echo "[1/3] 准备 Sionna RT、DeepMIMO V4、二维场景和在线 CSI"
if [[ "${REUSE_GENERATED}" == "1" && -f "${GENERATION_MANIFEST}" ]]; then
  verify_or_stop
  echo "生成产物检查通过，本次安全复用：${OUTPUT_ROOT}"
else
  if [[ "${REUSE_GENERATED}" == "1" ]]; then
    echo "没有找到生成清单，将执行一次新的生成。"
  else
    echo "REUSE_GENERATED=0，将执行一次新的生成；脚本不会删除或覆盖旧目录。"
  fi
  "${PYTHON_BIN}" -m time_bias_localization.cli prepare-sionna-scene \
    --config "${GENERATION_CONFIG_PATH}"
  require_file "${GENERATION_MANIFEST}" "生成清单"
  verify_or_stop
fi

require_file "${SCENE_JSON}" "二维场景"
require_file "${ONLINE_INPUT}" "在线 CSI"
require_file "${TRUTH_NPZ}" "独立评估真值"

echo "[2/3] 重新执行定位（此阶段不读取真值）"
"${PYTHON_BIN}" -m time_bias_localization.cli localize \
  --config "${LOCALIZATION_CONFIG_PATH}" \
  --scene-json "${SCENE_JSON}" \
  --online-input "${ONLINE_INPUT}" \
  --generation-manifest "${GENERATION_MANIFEST}" \
  --run-receipt "${RUN_RECEIPT_JSON}"
require_file "${RESULT_JSON}" "定位结果"
require_file "${RUN_RECEIPT_JSON}" "本次定位运行回执"

LOCALIZATION_RUN_ID="$("${PYTHON_BIN}" - \
  "${RUN_RECEIPT_JSON}" "${RESULT_JSON}" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

receipt_path = Path(sys.argv[1]).resolve()
expected_result_path = Path(sys.argv[2]).resolve()
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if not isinstance(receipt, dict) or set(receipt) != {"run_id", "result", "manifest"}:
    raise SystemExit("本次定位运行回执字段不完整或含有额外字段")
run_id = receipt["run_id"]
if not isinstance(run_id, str) or not run_id.strip():
    raise SystemExit("本次定位运行回执中的 run_id 无效")
expected_paths = {
    "result": expected_result_path,
    "manifest": expected_result_path.parent / "localization_manifest.json",
}
for name, expected_path in expected_paths.items():
    record = receipt.get(name)
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise SystemExit(f"本次定位运行回执中的 {name} 记录无效")
    if Path(record["path"]).expanduser().resolve() != expected_path:
        raise SystemExit(f"本次定位运行回执中的 {name} 路径不一致")
    if not expected_path.is_file():
        raise SystemExit(f"本次定位运行回执中的 {name} 文件不存在")
    if record["sha256"] != sha256(expected_path.read_bytes()).hexdigest():
        raise SystemExit(f"本次定位运行回执中的 {name} 哈希不一致")
print(run_id)
PY
)"

echo "[3/3] 重新执行独立评估"
"${PYTHON_BIN}" -m time_bias_localization.cli evaluate \
  --result-json "${RESULT_JSON}" \
  --truth-npz "${TRUTH_NPZ}" \
  --output-json "${METRICS_JSON}" \
  --expected-run-id "${LOCALIZATION_RUN_ID}"
require_file "${METRICS_JSON}" "评估结果"

echo "一键流程完成。"
echo "定位结果：${RESULT_JSON}"
echo "定位运行编号：${LOCALIZATION_RUN_ID}"
echo "定位运行回执：${RUN_RECEIPT_JSON}"
echo "评估结果：${METRICS_JSON}"
