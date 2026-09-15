"""冻结旧实验的接收数据与 MUSIC 峰，独立运行连续模型；所有请求留在总分母内。"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from time_bias_localization.boundary_experiment import source_fingerprint
from time_bias_localization.config import (
    load_localization_config, localization_config_view, validate_localization_config,
)
from time_bias_localization.provenance import file_sha256


def read_json(path: Path):
    return json.loads(path.read_text())


def implementation_fingerprint() -> dict:
    project = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), project / "scripts" / "detached_task.py",
             project / "run_continuous_model_experiment.sh"]
    return {**source_fingerprint(), "continuous_runner_files": {
        str(path.relative_to(project)): file_sha256(path) for path in files}}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def artifact(path: Path, expected_hash: str | None = None) -> dict:
    path = path.resolve(strict=True)
    digest = file_sha256(path)
    if expected_hash and digest != expected_hash:
        raise ValueError(f"冻结输入内容已经改变：{path}")
    return {"path": str(path), "sha256": digest}


def load_inventory(source: Path, cohort: str, strategy: str) -> list[dict]:
    """用计划构造总分母，避免只读取有位置输出的旧运行。"""
    plan = read_json(source / cohort / "plan.json")
    csv.field_size_limit(sys.maxsize)
    with (source / cohort / "report" / "trials.csv").open(encoding="utf-8-sig", newline="") as stream:
        baseline_rows = [row for row in csv.DictReader(stream) if row["strategy"] == strategy]
    baseline = {}
    for row in baseline_rows:
        key = (row["ue_id"], int(row["repeat_index"]))
        if key in baseline:
            raise ValueError(f"旧结果重复，必须明确唯一来源：{key}")
        baseline[key] = row
    jobs = []
    for point in plan["points"]:
        for repeat, observation in enumerate(point["observations"]):
            key = (point["ue_id"], repeat)
            row = baseline.get(key)
            if row is None:
                raise ValueError(f"计划中的观测缺少旧结果记录：{key}")
            root = Path(row["result_dir"])
            successful_root = root / "localization"
            manifest = successful_root / "localization_manifest.json"
            if not manifest.is_file():
                progress = row.get("progress_path", "")
                manifest = Path(progress) if progress else None
            frozen_screen = None
            if row["status"] == "excluded_observation":
                frozen_screen = "excluded_observation"
            elif row.get("stop_reason") == "insufficient_music_peaks":
                frozen_screen = "insufficient_observations"
            jobs.append({
                "ue_id": key[0], "repeat_index": repeat,
                "online_input": observation["online_npz"],
                "online_input_sha256": observation.get("input_sha256"),
                "scene_json": observation["scene_json"],
                "scene_sha256": observation.get("scene_sha256"),
                "evaluation_input": {"truth_npz": observation["truth_npz"],
                                     "truth_sha256": observation.get("truth_sha256")},
                "source_manifest": str(manifest) if manifest is not None else None,
                "frozen_screen_status": frozen_screen,
                "baseline": {
                    "historical_status": row["status"],
                    "historical_stop_reason": row.get("stop_reason", ""),
                    "historical_position_error_m": _optional_float(row.get("position_error_m")),
                    "historical_clock_bias_error_ns": _optional_float(row.get("clock_bias_error_ns")),
                    "historical_forward_valid": row.get("forward_valid") == "True",
                    "result_dir": str(root),
                },
            })
    if len(jobs) != len(baseline):
        raise ValueError("旧结果与冻结计划的观测集合不一致")
    return sorted(jobs, key=lambda job: (job["ue_id"], job["repeat_index"]))


def _optional_float(value):
    if value is None or value == "":
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def prepare_frozen_input(job: dict, requested_config: dict, output_root: Path) -> tuple[dict, dict]:
    """仅读取公开定位配置，保留原始地图、传播阶数、偏差范围和峰筛选。"""
    if not job["source_manifest"]:
        raise FileNotFoundError("旧运行缺少定位清单或失败进度记录，无法核验冻结峰")
    manifest_path = Path(job["source_manifest"])
    manifest = read_json(manifest_path)
    peaks_record = manifest["artifacts"]["music_peaks"]
    config_record = manifest["config_snapshot"]
    config_record_hash = config_record.get("file_sha256", config_record.get("sha256"))
    records = {
        "manifest": artifact(manifest_path),
        "music_peaks": artifact(Path(peaks_record["path"]), peaks_record["sha256"]),
        "online_input": artifact(Path(job["online_input"]), job["online_input_sha256"]),
        "scene": artifact(Path(job["scene_json"]), job["scene_sha256"]),
        "source_config": artifact(Path(config_record["path"]), config_record_hash),
    }
    for key, own_key in (("scene", "scene"), ("online_measurement", "online_input")):
        old = manifest["inputs"][key]
        if old["sha256"] != records[own_key]["sha256"]:
            raise ValueError(f"旧定位清单与计划中的 {key} 不一致")
    snapshot = read_json(Path(config_record["path"]))
    config = localization_config_view(snapshot.get("resolved_config", snapshot))
    # 只替换求解方法、连续求解参数和运行环境。既有峰不能因新前端参数重算。
    config["localization"]["solver_method"] = "continuous"
    config["localization"]["continuous"] = deepcopy(requested_config["localization"]["continuous"])
    config["compute"] = deepcopy(requested_config["compute"])
    config["project"]["random_seed"] = requested_config["project"]["random_seed"]
    config["output"]["root"] = str(output_root)
    config.pop("_config_path", None)
    validate_localization_config(config)
    return config, records


def evaluate_numeric_result(result: dict, evaluation_input: dict) -> dict:
    """仅在求解已结束后打开真值；结果状态不受误差大小影响。"""
    estimate = np.asarray(result.get("mu_m"), dtype=float)
    bias = result.get("clock_bias_s")
    if estimate.shape != (2,) or not np.all(np.isfinite(estimate)) or bias is None:
        return {"status": "no_numeric_output"}
    record = artifact(Path(evaluation_input["truth_npz"]), evaluation_input["truth_sha256"])
    with np.load(record["path"], allow_pickle=False) as truth:
        position = np.asarray(truth["ue_position_m"], dtype=float)
        true_bias = np.asarray(truth["clock_bias_s"], dtype=float)
    if position.shape != (2,) or true_bias.shape != ():
        raise ValueError("评估真值的形状不正确")
    return {"status": "evaluated_after_online_solve", "truth_input": record,
            "position_error_m": float(np.linalg.norm(estimate - position)),
            "clock_bias_error_ns": abs(float(bias) - float(true_bias)) * 1e9,
            "truth_used_in_online_solve": False}


def summarize(rows: list[dict], *, planned: int, completed: bool) -> dict:
    counts = dict(sorted(Counter(row["status"] for row in rows).items()))
    numeric = [row for row in rows if row.get("position_error_m") is not None]
    eligible = [row for row in rows if not row.get("frozen_screen_status")]
    baseline_counts = Counter(row.get("baseline", {}).get("historical_status", "unknown") for row in rows)
    common_checks = [row["baseline"]["common_state_validation"] for row in rows
                     if "common_state_validation" in row.get("baseline", {})]
    paired = [row for row in numeric if row.get("baseline", {}).get("historical_position_error_m") is not None]
    return {
        "planned_observation_count": planned,
        "recorded_observation_count": len(rows), "status_counts": counts,
        "pending_count": sum(row["status"] == "pending" for row in rows),
        "attempted_continuous_solve_count": sum(bool(row.get("attempted")) for row in rows),
        "frozen_screen_eligible_count": len(eligible), "numeric_output_count": len(numeric),
        "all_requested_attempts_finished": completed,
        "full_frozen_cohort_finished": completed and counts.get("pending", 0) == 0,
        "success_rate_total_denominator": counts.get("success", 0) / planned if planned else None,
        "position_error_mean_m_numeric_only": float(np.mean([row["position_error_m"] for row in numeric])) if numeric else None,
        "position_error_median_m_numeric_only": float(np.median([row["position_error_m"] for row in numeric])) if numeric else None,
        "baseline_historical_status_counts": dict(sorted(baseline_counts.items())),
        "baseline_common_state_checked_count": len(common_checks),
        "baseline_common_state_acceptable_count": sum(item.get("acceptable") is True for item in common_checks),
        "baseline_common_check_scope": "old_position_bias_reassociated_to_new_bank_without_refinement",
        "paired_numeric_count": len(paired),
        "paired_position_error_change_mean_m": float(np.mean([
            row["position_error_m"] - row["baseline"]["historical_position_error_m"]
            for row in paired])) if paired else None,
        "full_path_set_scientific_acceptance": "not_established",
        "same_music_peaks_reused": True, "old_solver_used_for_initialization": False,
        "historical_baseline_outputs_read_only": True,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", default="pilot", choices=("pilot", "formal"))
    parser.add_argument("--baseline-strategy", default="coverage", choices=("coverage", "single"))
    parser.add_argument("--limit", type=int, default=0,
                        help="最多运行多少份可求解观测；0 表示全部。其余记录保留 pending。")
    parser.add_argument("--ue-ids", default="", help="逗号分隔；未选的观测仍保留 pending")
    parser.add_argument("--repeat-indices", default="", help="逗号分隔，从 0 开始")
    parser.add_argument("--skip-evaluation", action="store_true", help="完全不打开评估真值")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit 不能为负数")
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents or output in source.parents:
        parser.error("新输出目录必须与旧实验目录分开")
    requested_config = load_localization_config(args.config)
    if requested_config["localization"]["solver_method"] != "continuous":
        parser.error("配置必须选择 continuous 求解器")
    jobs = load_inventory(source, args.cohort, args.baseline_strategy)
    chosen_ues = set(filter(None, args.ue_ids.split(",")))
    chosen_repeats = {int(value) for value in args.repeat_indices.split(",") if value}
    if chosen_ues - {job["ue_id"] for job in jobs}:
        parser.error("--ue-ids 包含计划外的编号")
    if chosen_repeats - {job["repeat_index"] for job in jobs}:
        parser.error("--repeat-indices 包含计划外的重复编号")
    output.mkdir(parents=True, exist_ok=False)
    source_before = implementation_fingerprint()
    write_json(output / "source.json", source_before)
    write_json(output / "comparison_plan.json", {
        "started_at": datetime.now(timezone.utc).isoformat(), "source": str(source),
        "source_plan": artifact(source / args.cohort / "plan.json"),
        "source_trials": artifact(source / args.cohort / "report" / "trials.csv"),
        "requested_config": artifact(args.config), "limit": args.limit,
        "ue_ids": sorted(chosen_ues), "repeat_indices": sorted(chosen_repeats),
        "planned_observation_count": len(jobs), "baseline_strategy": args.baseline_strategy,
        "frozen_frontend_and_screening": True, "evaluation_enabled": not args.skip_evaluation,
        "truth_read_policy": "only_after_online_solve_for_numeric_metrics",
        "configuration_policy": "frozen_public_config_plus_requested_continuous_compute_and_search_seed",
    })
    rows = [{"ue_id": job["ue_id"], "repeat_index": job["repeat_index"],
             "status": job["frozen_screen_status"] or "pending",
             "frozen_screen_status": job["frozen_screen_status"], "attempted": False,
             "baseline": job["baseline"]} for job in jobs]
    # 在首个长任务开始前保存完整分母，包括尚未运行的请求。
    write_json(output / "trials.json", rows)
    write_json(output / "summary.json", summarize(rows, planned=len(jobs), completed=False))
    from time_bias_localization.continuous_pipeline import localize_saved_music
    from time_bias_localization.continuous_solver import evaluate_continuous_state
    attempted = 0
    fatal_failures = 0
    for index, (job, row) in enumerate(zip(jobs, rows)):
        selected = (not chosen_ues or job["ue_id"] in chosen_ues) and (
            not chosen_repeats or job["repeat_index"] in chosen_repeats)
        if not selected or (args.limit and attempted >= args.limit):
            continue
        root = output / "trials" / job["ue_id"] / f"repeat_{job['repeat_index']:03d}"
        started = time.monotonic()
        if not job["frozen_screen_status"]:
            attempted += 1
            row["attempted"] = True
            row["started_at"] = datetime.now(timezone.utc).isoformat()
        try:
            current, records = prepare_frozen_input(job, requested_config, root)
            root.mkdir(parents=True, exist_ok=False)
            write_json(root / "frozen_input_records.json", records)
            row["input_records"] = str(root / "frozen_input_records.json")
            if job["frozen_screen_status"]:
                row["frozen_screen_inputs_verified"] = True
                write_json(root / "attempt.json", row)
                continue
            print(f"[{index + 1}/{len(jobs)}] {job['ue_id']} repeat_{job['repeat_index']:03d}：连续求解开始，输出={root}", flush=True)
            result, bank, solver_config = localize_saved_music(
                current, scene_json=records["scene"]["path"],
                music_peaks_json=records["music_peaks"]["path"], output_root=root,
                source_online_input=records["online_input"]["path"],
                source_manifest=records["manifest"]["path"],
                return_problem=True,
            )
            row.update(status=result["status"], result_dir=str(root),
                       localization_seconds=time.monotonic() - started,
                       diagnostic_summary=result.get("diagnostics", {}))
            paths = result.get("selected_paths", [])
            row["selected_path_count"] = len(paths)
            row["selected_diffraction_path_count"] = sum(
                any(kind == "diffraction" for kind, _ in path.get("propagation_interactions", []))
                for path in paths)
            row["forward_selected_paths_valid"] = result.get("forward_check", {}).get("all_selected_paths_valid", False)
            # 旧位置只在新求解完成后独立评分，不用于新方法初值或接受规则。
            baseline_result_path = Path(job["baseline"]["result_dir"]) / "localization" / "localization_result.json"
            if baseline_result_path.is_file():
                try:
                    baseline_manifest = read_json(Path(records["manifest"]["path"]))
                    original_result_record = baseline_manifest["artifacts"]["result"]
                    verified_result = artifact(baseline_result_path, original_result_record["sha256"])
                    baseline_result = read_json(baseline_result_path)
                    state = np.asarray([*baseline_result["mu_m"], baseline_result["distance_bias_m"]], dtype=float)
                    checked = evaluate_continuous_state(bank, solver_config, state)
                    checked["source_result"] = verified_result
                    write_json(root / "baseline_common_state_validation.json", checked)
                    row["baseline"]["common_state_validation"] = {
                        "acceptable": checked["acceptable"], "rejection_reasons": checked["rejection_reasons"],
                        "physical_rank": checked["physical_rank"], "objective": checked["objective"],
                        "path": str(root / "baseline_common_state_validation.json"),
                        "scope": "fixed_old_position_bias_reassociated_to_same_continuous_bank",
                        "old_estimate_refined": False,
                    }
                except Exception as error:
                    fatal_failures += 1
                    row["baseline"]["common_state_validation"] = {
                        "status": "validation_failed", "error": f"{type(error).__name__}: {error}"}
                    (root / "baseline_validation_failure.txt").write_text(traceback.format_exc())
            if not args.skip_evaluation:
                try:
                    evaluation = evaluate_numeric_result(result, job["evaluation_input"])
                    write_json(root / "evaluation" / "metrics.json", evaluation)
                    row.update({key: evaluation[key] for key in ("position_error_m", "clock_bias_error_ns") if key in evaluation})
                except Exception as error:
                    # 真值或评估坏了不能把已完成的在线状态改写为求解失败。
                    fatal_failures += 1
                    row["evaluation_status"] = "evaluation_failed"
                    row["evaluation_error"] = f"{type(error).__name__}: {error}"
                    (root / "evaluation_failure.txt").write_text(traceback.format_exc())
        except Exception as error:
            fatal_failures += 1
            row.update(status="execution_failed", error=f"{type(error).__name__}: {error}",
                       elapsed_seconds=time.monotonic() - started)
            root.mkdir(parents=True, exist_ok=True)
            (root / "failure.txt").write_text(traceback.format_exc())
        row["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "attempt.json", row)
        write_json(output / "trials.json", rows)
        write_json(output / "summary.json", summarize(rows, planned=len(jobs), completed=False))
        print(f"  状态={row['status']}；已尝试={attempted}，总请求={len(jobs)}", flush=True)
    summary = summarize(rows, planned=len(jobs), completed=True)
    summary["source_unchanged_during_run"] = implementation_fingerprint() == source_before
    summary["execution_failure_count"] = fatal_failures
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "trials.json", rows)
    write_json(output / "summary.json", summary)
    print(f"对照任务结束：{summary['status_counts']}；汇总={output / 'summary.json'}", flush=True)
    return 1 if fatal_failures or not summary["source_unchanged_during_run"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
