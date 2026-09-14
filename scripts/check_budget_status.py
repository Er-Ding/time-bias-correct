"""执行状态分类回归，并在独立目录重分类旧记录；不重跑定位。"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re
import subprocess
import sys

from time_bias_localization.boundary_experiment import source_fingerprint
from time_bias_localization.boundary_report import create_boundary_report
from time_bias_localization.provenance import file_sha256


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = source_fingerprint()
    write(args.output / "source.json", source)
    print("[代码检查] 执行完整回归测试", flush=True)
    subprocess.run([sys.executable, "-u", "-m", "pytest", "-q",
                    "--junitxml=" + str(args.output / "pytest.xml")], check=True)
    trials_path = args.experiment / "pilot/trials.jsonl"
    plan_path = args.experiment / "pilot/plan.json"
    input_hashes = {str(p): file_sha256(p) for p in (trials_path, plan_path)}
    records = [json.loads(line) for line in trials_path.read_text().splitlines()]
    plan = json.loads(plan_path.read_text())
    revised = deepcopy(records)
    changes = []
    for old, row in zip(records, revised):
        if old["status"] != "unlocalizable":
            continue
        reason = old.get("unlocalizable_reason")
        if reason not in {"path_detection_budget_exhausted", "no_solvable_joint_candidate"}:
            continue
        progress_path = Path(old["progress_path"])
        input_hashes[str(progress_path)] = file_sha256(progress_path)
        progress = json.loads(progress_path.read_text())
        entry = progress["artifacts"]["result"]
        path = Path(entry["path"])
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"保存结果摘要不符：{path}")
        input_hashes[str(path)] = entry["sha256"]
        result = json.loads(path.read_text())
        if result["status"] != "unlocalizable" or result["reason"] != reason:
            raise ValueError("逐次记录与对应结果不一致")
        if reason == "path_detection_budget_exhausted":
            if result["diagnostics"]["path_detection"]["stop_reason"] != "path_limit_reached":
                raise ValueError("没有明确的路径检测预算耗尽证据")
            status = "detection_incomplete"
        else:
            match = re.match(r"跨观测候选对数量 (\d+) 超过 max_seed_pairs=(\d+)；",
                             result["diagnostics"].get("solver_reason", ""))
            if not match:
                continue  # 不能把其他求解错误推断成预算问题。
            if int(match[1]) <= int(match[2]):
                raise ValueError("候选对预算诊断不一致")
            status, reason = "solver_budget_exhausted", "solver_pair_budget_exhausted"
        if row["position_error_m"] is not None or row["clock_bias_error_ns"] is not None:
            raise ValueError("无坐标记录却包含精度结果")
        row.update(status=status, stop_reason=reason, unlocalizable_reason=None,
                   recorded_status=old["status"], recorded_reason=old["unlocalizable_reason"])
        changes.append({"ue_id": row["ue_id"], "repeat_index": row["repeat_index"],
                        "strategy": row["strategy"], "new_status": status, "progress_path": str(progress_path)})
    print(f"[历史记录复核] {len(records)} 条记录，{len(changes)} 条明确的预算退出重新分类", flush=True)
    report = create_boundary_report(args.output / "reclassified_report", revised, plan["points"],
                                   metadata={"plots": False, "purpose": "status_reclassification_only_not_new_localization"})
    for old, row in zip(records, revised):
        for key in ("input_sha256", "noise_seed", "mc_seed", "position_error_m", "clock_bias_error_ns",
                    "localization_seconds", "processing_seconds", "stage_timings"):
            if old.get(key) != row.get(key):
                raise AssertionError(f"重分类意外改变 {key}")
    for path, expected in input_hashes.items():
        if file_sha256(path) != expected:
            raise AssertionError("原始记录在检查期间发生变化")
    if source_fingerprint() != source:
        raise AssertionError("验收过程中代码发生变化")
    (args.output / "reclassified_trials.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n" for r in revised))
    write(args.output / "validation_summary.json", {
        "status": "passed", "scope": "tests_and_saved_record_reclassification_only",
        "original_results_modified": False, "new_localization_executed": False,
        "input_hashes": input_hashes, "reclassified_count": len(changes), "changes": changes,
        "planned_count": report["planned_count"],
        "status_counts_by_strategy": {s: dict(Counter(r["status"] for r in revised if r["strategy"] == s))
                                      for s in ("single", "coverage")},
    })
    print(f"[完成] {args.output}/validation_summary.json", flush=True)


if __name__ == "__main__":
    main()
