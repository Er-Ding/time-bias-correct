"""只读重放：镜像角歧义组按角度分支求解，统计原本被整点排除的样本能否恢复。

对每个 ``excluded_observation`` 样本复用已保存的 MUSIC 峰、地图和观测配置，
不重新生成 CSI、不修改原实验。真值只在离线比较中使用。

注意：原样本在筛选处提前退出（约 2 秒），重放会真正跑完求解器，耗时由
观测条数和连续模型预算决定，远大于原耗时。
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

from time_bias_localization.continuous_pipeline import run_continuous_with_angle_branches
from time_bias_localization.observation_screen import ambiguity_groups
from time_bias_localization.scene import Scene2D


ANGLE_MATCH_DEG = 3.0
DELAY_MATCH_NS = 5.0


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                     default=_json_default) + "\n")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"无法序列化 {type(value).__name__}")


def wrap_degrees(value):
    return (value + 180.0) % 360.0 - 180.0


class Evidence:
    def __init__(self):
        self.files: dict[str, str] = {}

    def read(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.files:
            self.files[key] = digest(path)
        if path.suffix == ".npz":
            return np.load(path, allow_pickle=False)
        return json.loads(path.read_text())

    def recheck(self):
        for path, expected in self.files.items():
            if digest(path) != expected:
                raise ValueError(f"重放期间输入发生变化：{path}")


def truth_paths(archive, period_ns, boresight_deg):
    rows = []
    for index in np.flatnonzero(archive["retained_mask"]):
        rows.append({
            "path_index": int(index),
            "angle_deg": wrap_degrees(float(np.rad2deg(archive["aoa_local_rad"][index])) + boresight_deg),
            "observed_delay_ns": float((archive["absolute_delays_s"][index]
                                        + archive["clock_bias_s"]) * 1e9),
            "reflections": int(archive["reflection_order"][index]),
            "diffractions": int(archive["diffraction_order"][index]),
        })
    for row in rows:
        row["delay_period_index"] = period_ns and int(round(row["observed_delay_ns"] / period_ns))
    return rows


def matches_truth(angle_deg, delay_s, paths, period_ns):
    """峰是否对上任意一条真实路径；时延按子载波周期折回后比较。"""
    delay_ns = delay_s * 1e9
    for path in paths:
        if abs(wrap_degrees(angle_deg - path["angle_deg"])) > ANGLE_MATCH_DEG:
            continue
        period_index = int(round((path["observed_delay_ns"] - delay_ns) / period_ns))
        folded = path["observed_delay_ns"] - period_index * period_ns
        if abs(delay_ns - folded) <= DELAY_MATCH_NS:
            return {"matched": True, "path_index": path["path_index"],
                    "angle_error_deg": wrap_degrees(angle_deg - path["angle_deg"]),
                    "delay_error_ns": delay_ns - folded}
    return {"matched": False}


def replay_sample(evidence, source, scene, row):
    sample_dir = source / "samples" / row["sample_id"]
    attempt = Path(row["attempt_dir"])
    folder, = (attempt / "localization_unavailable").glob("*")
    peaks_document = evidence.read(folder / "music_peaks.json")
    config = evidence.read(folder / "localization_config.json")["resolved_config"]
    observation = evidence.read(sample_dir / "observation.json")
    online = evidence.read(observation["artifacts"]["online_npz"]["path"])
    truth = evidence.read(observation["artifacts"]["truth_npz"]["path"])
    try:
        frequency = online["subcarrier_frequencies_hz"]
        period_ns = 1e9 / float(np.median(np.diff(frequency)))
        boresight_deg = float(config["radio"]["bs_boresight_deg"])
        bs = np.asarray(config["radio"]["bs_position_m"], float)
        peaks = [SimpleNamespace(aoa_rad=float(value["aoa_rad"]), delay_s=float(value["delay_s"]))
                 for value in peaks_document["nominal"]]
        indices = list(peaks_document["nominal_source_indices"])
        screening = peaks_document["observation_screen"]
        groups = ambiguity_groups(screening["excluded_pairs"], len(peaks))
        recorded = screening.get("ambiguity_groups")
        if recorded is not None and recorded != groups:
            raise ValueError(f"记录歧义组与重算不一致：{row['sample_id']}")
        config["music"]["observation_screen"]["ambiguity_policy"] = "enumerate_branches"
        true_position = np.asarray(truth["ue_position_m"], float)
        true_bias_ns = float(truth["clock_bias_s"]) * 1e9
        paths = truth_paths(truth, period_ns, boresight_deg)
        began = time.perf_counter()
        selection = run_continuous_with_angle_branches(
            config, scene, peaks, indices, groups, bs, math.radians(boresight_deg))
        elapsed = time.perf_counter() - began
    finally:
        online.close()
        truth.close()

    report = selection.report
    result = selection.result
    kept_rows = []
    for branch in report["branches"]:
        checked = [matches_truth(wrap_degrees(math.degrees(peaks[index].aoa_rad) + boresight_deg),
                                 peaks[index].delay_s, paths, period_ns)
                   for index in branch["kept_peak_positions"]]
        kept_rows.append({"branch_index": branch["branch_index"],
                          "kept_peak_positions": branch["kept_peak_positions"],
                          "kept_peak_truth_matches": checked,
                          "kept_truth_match_count": sum(item["matched"] for item in checked),
                          "status": branch["status"], "mu_m": branch["mu_m"],
                          "objective": branch["objective"],
                          "matched_observation_count": branch["matched_observation_count"],
                          "physical_rank": branch["physical_rank"],
                          "skipped_reason": branch["skipped_reason"]})
    selected = report["selected_branch"]
    selected_row = kept_rows[selected] if selected is not None else None
    critical_groups = []
    for members in groups:
        flags = [matches_truth(wrap_degrees(math.degrees(peaks[index].aoa_rad) + boresight_deg),
                               peaks[index].delay_s, paths, period_ns)["matched"] for index in members]
        if sum(flags) == 1:
            critical_groups.append({"peak_indices": list(members),
                                    "true_member_position": members[flags.index(True)]})
    selection_kept_true_member = (None if not critical_groups or selected_row is None else
                                 all(item["true_member_position"] in selected_row["kept_peak_positions"]
                                     for item in critical_groups))
    entry = {
        "sample_id": row["sample_id"], "status_original": row["status"],
        "reason_original": row["reason"], "rt_path_count": row["rt_path_count"],
        "peak_count": len(peaks), "music_peak_count": row["music_observation_count"],
        "ambiguity_groups": groups, "branch_count": report["branch_count"],
        "true_position_m": true_position.tolist(), "true_bias_ns": true_bias_ns,
        "truth_paths": paths, "delay_period_ns": period_ns,
        "branches": kept_rows, "selected_branch": selected,
        "all_branches_failed": report["all_branches_failed"],
        "failure_reason": report["failure_reason"],
        "recovered": result is not None and result.get("mu_m") is not None,
        "selected_status": None if result is None else result["status"],
        "selected_kept_true_member": selection_kept_true_member,
        "critical_groups": critical_groups,
        "replay_seconds": elapsed,
        "bias_bound_ns": [float(config["localization"]["bias_min_s"]) * 1e9,
                          float(config["localization"]["bias_max_s"]) * 1e9],
        "evidence_paths": {"music_peaks": str(folder / "music_peaks.json"),
                           "localization_config": str(folder / "localization_config.json"),
                           "observation": str(sample_dir / "observation.json")},
    }
    if entry["recovered"]:
        position = np.asarray(result["mu_m"], float)
        bias_ns = float(result["distance_bias_m"]) / 299792458.0 * 1e9
        low, high = entry["bias_bound_ns"]
        entry.update(position_error_m=float(np.linalg.norm(position - true_position)),
                     bias_error_ns=abs(bias_ns - true_bias_ns), estimated_bias_ns=bias_ns,
                     bias_at_bound=bool(abs(bias_ns - low) < 1e-6 or abs(bias_ns - high) < 1e-6),
                     output_type=result["output_type"])
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", default="", help="逗号分隔的样本编号，空表示全部")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少样本，0 表示不限")
    parser.add_argument("--time-budget-s", type=float, default=0.0, help="总时限，0 表示不限")
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("输出必须与只读输入分开")
    output.mkdir(parents=True, exist_ok=False)
    evidence = Evidence()
    scene_path, = (source / "channel_setups").glob("*/scene/scene_2d.json")
    scene = Scene2D.from_dict(evidence.read(scene_path))
    rows = []
    for path in sorted((source / "samples").glob("SAMPLE_*/result.json")):
        row = evidence.read(path)
        if row["status"] != "excluded_observation":
            continue
        row["sample_id"] = path.parent.name
        rows.append(row)
    wanted = {int(value) for value in args.samples.split(",") if value.strip()}
    if wanted:
        rows = [row for row in rows if int(row["sample_id"].split("_")[-1]) in wanted]
    if args.limit:
        rows = rows[:args.limit]
    cases, skipped = [], []
    began = time.perf_counter()
    for index, row in enumerate(rows, 1):
        if args.time_budget_s and time.perf_counter() - began > args.time_budget_s:
            skipped = [item["sample_id"] for item in rows[index - 1:]]
            break
        case = replay_sample(evidence, source, scene, row)
        cases.append(case)
        print(f"[重放] {index}/{len(rows)} {case['sample_id']}：分支 {case['branch_count']}，"
              f"选中 {case['selected_branch']}，恢复 {case['recovered']}，"
              f"耗时 {case['replay_seconds']:.1f} s", flush=True)
        write_json(output / "replay.json", {"cases": cases, "pending": skipped})
    recovered = [case for case in cases if case["recovered"]]
    errors = [case["position_error_m"] for case in recovered]
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(source), "sample_count": len(cases),
        "pending_sample_ids": skipped,
        "recovered_count": len(recovered),
        "position_error_within_1m": sum(value <= 1.0 for value in errors),
        "position_error_within_2m": sum(value <= 2.0 for value in errors),
        "position_error_within_5m": sum(value <= 5.0 for value in errors),
        "bias_at_bound_count": sum(case.get("bias_at_bound", False) for case in recovered),
        "recovered_accurate_and_bias_free": sum(
            case["position_error_m"] <= 1.0 and not case.get("bias_at_bound", False)
            for case in recovered),
        "branch_count_histogram": dict(Counter(case["branch_count"] for case in cases)),
        "original_peak_count_histogram": dict(Counter(case["peak_count"] for case in cases)),
        "selected_branch_histogram": dict(Counter(case["selected_branch"] for case in cases)),
        "recovered_with_true_member": sum(case["selected_kept_true_member"] is True
                                          for case in recovered),
        "recovered_without_true_member": sum(case["selected_kept_true_member"] is False
                                             for case in recovered),
        "failure_reason_histogram": dict(Counter(case["failure_reason"] for case in cases
                                                 if not case["recovered"])),
        "position_error_m": _describe(errors),
        "bias_error_ns": _describe([case["bias_error_ns"] for case in recovered]),
        "total_replay_seconds": time.perf_counter() - began,
        "reused_saved_music_peaks": True, "csi_regenerated": False,
        "online_policy_before_replay": "exclude_sample",
        "replay_policy": "enumerate_branches",
        "truth_used_for_offline_comparison_only": True,
    }
    write_json(output / "summary.json", summary)
    write_markdown(output / "summary.md", summary, cases)
    evidence.recheck()
    write_json(output / "provenance.json", {
        "input_files_sha256": evidence.files, "input_unchanged_after_replay": True,
        "code_changed": False, "full_experiment_rerun": False})
    print(f"[完成] {output / 'summary.md'}", flush=True)


def _describe(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return {"count": 0}
    array = np.asarray(values)
    return {"count": int(array.size), "min": float(array.min()), "median": float(np.median(array)),
            "mean": float(array.mean()), "max": float(array.max()),
            "p90": float(np.percentile(array, 90))}


def write_markdown(path, summary, cases):
    lines = ["# 镜像角歧义组分支重放", "",
             f"输入：`{summary['input_root']}`。复用已保存的 MUSIC 峰与地图，未重新生成 CSI。", "",
             "原样本在观测筛选处整点退出；本次按歧义组枚举角度分支，选有位置且残差代价最低者。", "",
             "## 汇总", "",
             f"- 已重放样本：{summary['sample_count']}"
             + (f"（未处理 {len(summary['pending_sample_ids'])} 个：时限到达）"
                if summary["pending_sample_ids"] else ""),
             f"- 恢复为有位置输出：{summary['recovered_count']}",
             f"  - 其中位置误差 ≤ 1 m：{summary['position_error_within_1m']}，"
             f"≤ 2 m：{summary['position_error_within_2m']}，≤ 5 m：{summary['position_error_within_5m']}",
             f"  - 其中偏置被压到区间边界（退化解）：{summary['bias_at_bound_count']}",
             f"  - 位置 ≤ 1 m 且偏置未触边界：{summary['recovered_accurate_and_bias_free']}",
             f"- 恢复且选中了含真实成员的镜像分支：{summary['recovered_with_true_member']}",
             f"- 恢复但选中的分支不含真实成员：{summary['recovered_without_true_member']}",
             f"- 分支数分布：{summary['branch_count_histogram']}",
             f"- 未恢复原因的选中分支失败原因：{summary['failure_reason_histogram']}", ""]
    if summary["position_error_m"]["count"]:
        error = summary["position_error_m"]
        lines.extend([f"- 恢复样本位置误差（米）：中位 {error['median']:.4f}，均值 {error['mean']:.4f}，"
                      f"p90 {error['p90']:.4f}，最大 {error['max']:.4f}", ""])
    lines.extend(["## 逐样本", "",
                  "| 样本 | 峰数 | 分支 | 选中 | 恢复 | 位置误差/m | 偏置误差/ns | 偏置触边界 | 选中含真实成员 | 原真实路径数 |",
                  "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- | ---: |"])
    for case in cases:
        if case["recovered"]:
            error = format(case["position_error_m"], ".4f")
            bias = format(case["bias_error_ns"], ".4f")
            at_bound = "是" if case.get("bias_at_bound") else "否"
        else:
            error = bias = at_bound = ""
        lines.append(f"| {case['sample_id']} | {case['peak_count']} | {case['branch_count']} | "
                     f"{case['selected_branch']} | {'是' if case['recovered'] else '否'} | "
                     f"{error} | {bias} | {at_bound} | "
                     f"{case['selected_kept_true_member']} | {case['rt_path_count']} |")
    lines.extend(["", "## 边界", "",
                  "恢复数量只统计本次重放；未重新生成 CSI，未修改原实验，未更新正式成功率。",
                  "真值仅用于离线比较，不参与求解与选择。",
                  "偏置触边界表示解落在 ±bias_max 上，属于退化拟合，不应计入恢复成功。", ""])
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
