#!/usr/bin/env bash
set -euo pipefail

# ======================== 用户参数区 ========================
PROJECT_ROOT="/data/zhujun/differt_projects/time-bias-correct"
PYTHON_BIN="${PYTHON_BIN:-/home/zhujun/miniconda3/bin/python3}"
RESULT_JSON="${RESULT_JSON:-${PROJECT_ROOT}/outputs/offline_demo/localization/localization_result.json}"
TRUTH_NPZ="${TRUTH_NPZ:-${PROJECT_ROOT}/outputs/offline_demo/data/truth/ground_truth.npz}"
METRICS_JSON="${METRICS_JSON:-${PROJECT_ROOT}/outputs/offline_demo/evaluation/metrics.json}"
# 二选一：优先填写定位阶段生成的回执；或显式填写定位运行编号。
RUN_RECEIPT_JSON="${RUN_RECEIPT_JSON:-}"
EXPECTED_RUN_ID="${EXPECTED_RUN_ID:-}"
# ===========================================================

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${PROJECT_ROOT}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "找不到可执行的 Python：${PYTHON_BIN}" >&2
  exit 2
fi

resolve_project_path() {
  local path_value="$1"
  if [[ "${path_value}" != /* ]]; then
    path_value="${PROJECT_ROOT}/${path_value}"
  fi
  readlink -m "${path_value}"
}

RESULT_JSON="$(resolve_project_path "${RESULT_JSON}")"
TRUTH_NPZ="$(resolve_project_path "${TRUTH_NPZ}")"
METRICS_JSON="$(resolve_project_path "${METRICS_JSON}")"
OUTPUT_ROOT="$(dirname "$(dirname "${RESULT_JSON}")")"
REQUIRED_METRICS_JSON="$(readlink -m "${OUTPUT_ROOT}/evaluation/metrics.json")"

if [[ "${METRICS_JSON}" != "${REQUIRED_METRICS_JSON}" ]]; then
  echo "METRICS_JSON 必须固定为 ${REQUIRED_METRICS_JSON}；当前为 ${METRICS_JSON}。" >&2
  exit 2
fi

if [[ -n "${RUN_RECEIPT_JSON}" ]]; then
  RUN_RECEIPT_JSON="$(resolve_project_path "${RUN_RECEIPT_JSON}")"
  if [[ ! -f "${RUN_RECEIPT_JSON}" ]]; then
    echo "找不到定位运行回执：${RUN_RECEIPT_JSON}" >&2
    exit 2
  fi
  RECEIPT_RUN_ID="$("${PYTHON_BIN}" - \
    "${RUN_RECEIPT_JSON}" "${RESULT_JSON}" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

receipt_path = Path(sys.argv[1]).resolve()
expected_result_path = Path(sys.argv[2]).resolve()
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if not isinstance(receipt, dict) or set(receipt) != {"run_id", "result", "manifest"}:
    raise SystemExit("定位运行回执字段不完整或含有额外字段")
run_id = receipt["run_id"]
if not isinstance(run_id, str) or not run_id.strip():
    raise SystemExit("定位运行回执中的 run_id 不是非空字符串")

expected_paths = {
    "result": expected_result_path,
    "manifest": expected_result_path.parent / "localization_manifest.json",
}
records = {}
for name, expected_path in expected_paths.items():
    record = receipt.get(name)
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise SystemExit(f"定位运行回执中的 {name} 记录必须只含 path 和 sha256")
    recorded_path = Path(record["path"]).expanduser().resolve()
    if recorded_path != expected_path:
        raise SystemExit(
            f"定位运行回执中的 {name} 路径与本次评估输入不一致："
            f"回执={recorded_path}，本次={expected_path}"
        )
    if not expected_path.is_file():
        raise SystemExit(f"定位运行回执记录的 {name} 文件不存在：{expected_path}")
    records[name] = record

result_sha256 = sha256(expected_result_path.read_bytes()).hexdigest()
if records["result"]["sha256"] != result_sha256:
    raise SystemExit("定位运行回执记录的 result 文件哈希已不匹配")

manifest_path = expected_paths["manifest"]
manifest_bytes = manifest_path.read_bytes()
manifest_sha256 = sha256(manifest_bytes).hexdigest()
if records["manifest"]["sha256"] != manifest_sha256:
    # 首次评估成功后，清单只会把 evaluation_pending 改成 false，并增加
    # evaluation。恢复这两处后，必须逐字节重建出回执记录的评估前清单。
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("run_id") != run_id:
        raise SystemExit("当前定位清单既不匹配回执哈希，也不属于回执运行编号")
    if manifest.get("schema_version") not in (3, 4):
        raise SystemExit("只有 schema_version=3 或 4 的定位清单允许核对评估写回")
    current_result = manifest.get("artifacts", {}).get("result")
    if not isinstance(current_result, dict):
        raise SystemExit("当前定位清单缺少结果来源记录")
    if (
        Path(current_result.get("path", "")).expanduser().resolve()
        != expected_result_path
        or current_result.get("sha256") != records["result"]["sha256"]
    ):
        raise SystemExit("当前定位清单中的结果来源与运行回执不一致")
    evaluation = manifest.get("evaluation")
    if (
        manifest.get("evaluation_pending") is not False
        or not isinstance(evaluation, dict)
        or set(evaluation) != {"path", "sha256"}
    ):
        raise SystemExit("当前定位清单的变化不是一次完整评估写回")
    evaluation_path = Path(evaluation.get("path", "")).expanduser().resolve()
    if not evaluation_path.is_file():
        raise SystemExit("当前定位清单记录的既有评估文件不存在")
    if evaluation.get("sha256") != sha256(evaluation_path.read_bytes()).hexdigest():
        raise SystemExit("当前定位清单记录的既有评估文件哈希不一致")
    pre_evaluation_manifest = dict(manifest)
    pre_evaluation_manifest.pop("evaluation")
    pre_evaluation_manifest["evaluation_pending"] = True
    pre_evaluation_bytes = json.dumps(
        pre_evaluation_manifest, ensure_ascii=False, indent=2
    ).encode("utf-8")
    if sha256(pre_evaluation_bytes).hexdigest() != records["manifest"]["sha256"]:
        raise SystemExit(
            "当前定位清单除评估写回外还发生了变化，与运行回执不一致"
        )
print(run_id)
PY
  )"
  if [[ -n "${EXPECTED_RUN_ID}" && "${EXPECTED_RUN_ID}" != "${RECEIPT_RUN_ID}" ]]; then
    echo "EXPECTED_RUN_ID 与运行回执中的编号不一致，已停止。" >&2
    exit 2
  fi
  EXPECTED_RUN_ID="${RECEIPT_RUN_ID}"
elif [[ -z "${EXPECTED_RUN_ID}" ]]; then
  echo "必须设置 RUN_RECEIPT_JSON 或明确的 EXPECTED_RUN_ID；不会从当前定位清单猜测。" >&2
  exit 2
fi

exec "${PYTHON_BIN}" -m time_bias_localization.cli evaluate \
  --result-json "${RESULT_JSON}" \
  --truth-npz "${TRUTH_NPZ}" \
  --output-json "${METRICS_JSON}" \
  --expected-run-id "${EXPECTED_RUN_ID}"
