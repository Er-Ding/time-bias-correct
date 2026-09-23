"""逐点核对已有实验的无位置样本；真值只用于离线解释，不重新执行定位。"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from types import SimpleNamespace
import csv

import numpy as np

from time_bias_localization.constants import SPEED_OF_LIGHT_M_S as C
from time_bias_localization.continuous_solver import ContinuousSolverConfig, evaluate_continuous_state
from time_bias_localization.observation_screen import screen_music_observation
from time_bias_localization.propagation_hypotheses import HypothesisBank
from time_bias_localization.propagation_model import ContinuousObservation, PropagationHypothesis
from time_bias_localization.scene import Scene2D


PATH_NAMES = ["直达", "一次反射", "两次反射", "单次绕射", "一次反射后绕射", "两次反射后绕射"]
CATEGORIES = {
    "single_path": "只有一条有效路径",
    "refinement_merge": "两条观测在细化去重时合并",
    "response_screen": "阵列两端的相似响应触发整点排除",
    "missing_route": "已构造的路线库缺少真实绕射路线",
    "shared_corner": "候选只匹配同一拐角的重复信息",
    "not_converged": "候选未通过局部优化收敛检查",
    "ambiguous": "多个不同位置的评分接近",
    "timeout": "联合优化达到处理时限",
}
TERMINATIONS = {"iteration_budget": "达到局部迭代次数上限", "damping_stalled": "步长压小后停滞"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Evidence:
    def __init__(self):
        self.files = {}
        self.verified_references = 0

    def capture(self, path):
        path = Path(path).resolve()
        if str(path) not in self.files:
            self.files[str(path)] = digest(path)
        return path

    def read(self, path):
        return json.loads(self.capture(path).read_text())

    def verify(self, record):
        path = self.capture(record["path"])
        expected = record.get("sha256", record.get("file_sha256"))
        if not expected or self.files[str(path)] != expected:
            raise ValueError(f"来源校验失败：{path}")
        self.verified_references += 1
        return path

    def recheck(self):
        for path, expected in self.files.items():
            if digest(path) != expected:
                raise ValueError(f"分析期间输入发生变化：{path}")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def path_type(reflections, diffractions):
    return PATH_NAMES[reflections + 3 if diffractions else reflections]


def maximum_separable_subset(count, pairs):
    edges = {tuple(sorted((p["first_peak_index"], p["second_peak_index"]))) for p in pairs}
    for size in range(count, 0, -1):
        for subset in combinations(range(count), size):
            if all(tuple(pair) not in edges for pair in combinations(subset, 2)):
                return list(subset)
    return []


def truth_paths(archive, period_ns, delay_bounds_ns):
    rows = []
    for index in np.flatnonzero(archive["retained_mask"]):
        delay = float((archive["absolute_delays_s"][index] + archive["clock_bias_s"]) * 1e9)
        period_indices = list(range(int(np.ceil((delay - delay_bounds_ns[1]) / period_ns)),
                                    int(np.floor((delay - delay_bounds_ns[0]) / period_ns)) + 1))
        kinds = archive["interactions"][:, index]
        r, d = int(archive["reflection_order"][index]), int(archive["diffraction_order"][index])
        rows.append({"truth_path_index": int(index), "path_type": path_type(r, d),
            "reflection_count": r, "diffraction_count": d,
            "angle_local_deg": float(np.rad2deg(archive["aoa_local_rad"][index])),
            "observed_delay_ns": delay,
            "delay_representatives_in_search_window_ns": [delay - k * period_ns for k in period_indices],
            "delay_period_indices": period_indices,
            "diffraction_corners_m": archive["vertices_m"][:, index, :2][kinds == 8].tolist()})
    return rows


def peak_truth_comparison(peaks, truth, period_ns):
    """仅提供每个峰最接近的真值证据；不是一对一的路径真伪判定。"""
    rows = []
    for i, peak in enumerate(peaks):
        angle, delay = float(np.rad2deg(peak["aoa_rad"])), peak["delay_s"] * 1e9
        matches = []
        for t in truth:
            angle_error = abs((angle - t["angle_local_deg"] + 180) % 360 - 180)
            k = int(round((t["observed_delay_ns"] - delay) / period_ns))
            error = delay - (t["observed_delay_ns"] - k * period_ns)
            matches.append({"truth_path_index": t["truth_path_index"], "path_type": t["path_type"],
                "angle_error_deg": angle_error, "delay_error_modulo_period_ns": error,
                "delay_period_index": k,
                "distance_in_engineering_scales": float(np.hypot(angle_error, error * C * 1e-9 / .75))})
        nearest = min(matches, key=lambda x: x["distance_in_engineering_scales"])
        rows.append({"peak_index": i, "angle_local_deg": angle, "delay_ns": delay,
                     "spectrum_value": peak["spectrum_value"], "nearest_truth_with_period": nearest})
    return rows


def candidate_summary(candidate, true_position, true_bias):
    if not candidate:
        return None
    fields = ("position_m", "clock_bias_s", "objective", "matched_observation_count", "physical_rank",
              "condition_number", "rejection_reasons", "termination", "converged", "acceptable",
              "selected_paths", "unmatched_observation_ids")
    row = {key: candidate[key] for key in fields}
    row.update(position_error_offline_m=float(np.linalg.norm(np.asarray(candidate["position_m"]) - true_position)),
               bias_error_offline_ns=abs(candidate["clock_bias_s"] * 1e9 - true_bias))
    return row


def check_true_state(evidence, folder, scene, receiver, truth_position, truth_bias):
    """固定在事后真值处检查旧观测和旧路线库；不搜索、不输出位置估计。"""
    bank_data = evidence.read(folder / "propagation_hypotheses.json")
    observed = evidence.read(folder / "continuous_observations.json")
    search = evidence.read(folder / "continuous_search.json")
    hypotheses = []
    for saved in bank_data["hypotheses"]:
        row = dict(saved)
        row["interactions"] = tuple(tuple(x) for x in row["interactions"])
        hypotheses.append(PropagationHypothesis(**row))
    bank = HypothesisBank(scene, tuple(receiver),
        tuple(ContinuousObservation(**r) for r in observed["observations"]), tuple(hypotheses),
        tuple(tuple(x) for x in bank_data["observation_hypothesis_indices"]), bank_data["search_report"])
    settings = ContinuousSolverConfig(**search["diagnostics"]["config"])
    state = [*truth_position, truth_bias * 1e-9 * C]
    result = evaluate_continuous_state(bank, settings, state)
    # evaluate_continuous_state 默认设置 converged=True 以复用局部检查；这里
    # 并未执行优化，不能把该默认值作为已经收敛的证据写进离线结果。
    result["local_checks_passed"] = result.pop("acceptable")
    result.pop("converged")
    result["termination"] = "offline_fixed_state_check_no_optimization"
    result["scope"] = "offline_truth_state_checks_only_not_a_solver_result_or_uniqueness_check"
    result["convergence_was_tested"] = False
    return result, bank_data, search


def analyze_case(evidence, row, experiment, scene):
    sid, attempt = row["sample_id"], Path(row["attempt_dir"])
    sample_root = attempt.parent.parent
    evidence.verify(row["solve_record"])
    evidence.verify(row["evaluation_record"])
    observation = evidence.read(sample_root / "observation.json")
    for record in observation["artifacts"].values():
        evidence.verify(record)
    config = experiment["generation_config"]
    with np.load(observation["artifacts"]["online_npz"]["path"], allow_pickle=False) as online:
        frequency = online["subcarrier_frequencies_hz"]
        period_ns = 1e9 / float(np.median(np.diff(frequency)))
        radio = {"num_antennas": int(online["csi_observed"].shape[-2]),
            "antenna_spacing_m": float(online["antenna_spacing_m"]),
            "carrier_frequency_hz": float(online["carrier_frequency_hz"]), "frequencies_hz": frequency}
        receiver = online["bs_position_m"].tolist()
    with np.load(observation["artifacts"]["truth_npz"]["path"], allow_pickle=False) as truth:
        paths = truth_paths(truth, period_ns,
            [config["music"]["delay_min_s"] * 1e9, config["music"]["delay_max_s"] * 1e9])
        true_position, true_bias = truth["ue_position_m"].tolist(), float(truth["clock_bias_s"]) * 1e9
    assert len(paths) == row["rt_path_count"] == sum(row["path_type_counts"])
    assert np.allclose(true_position, row["true_position_m"], rtol=0, atol=1e-10)
    assert abs(true_bias - row["true_clock_bias_ns"]) < 1e-10
    case = {"sample_id": sid, "status": row["status"], "original_reason": row["reason"],
        "true_position_m_for_offline_analysis": true_position, "true_bias_ns_for_offline_analysis": true_bias,
        "rt_path_count": row["rt_path_count"], "path_type_counts": row["path_type_counts"],
        "processing_seconds": row["processing_seconds"], "truth_paths": paths,
        "delay_period_ns": period_ns, "original_attempt_dir": str(attempt),
        "evidence_paths": {"result": str(sample_root / "result.json"),
                           "truth_archive": observation["artifacts"]["truth_npz"]["path"]},
        "supporting_findings": []}
    if row["status"] == "timeout":
        log_path = evidence.capture(attempt / "online.log")
        log = log_path.read_text()
        progress = re.findall(r"已完成初值 (\d+)/(\d+).*?满足接受条件的初值结果 (\d+) 个", log)
        assert progress, f"缺少超时进度：{sid}"
        done, total, acceptable = map(int, progress[-1])
        observed = re.search(r"观测 (\d+) 条，传播函数 (\d+) 个", log)
        case.update(category="timeout", music_peak_count=int(observed[1]),
            last_confirmed_start_count=done, planned_start_count=total,
            acceptable_start_results_before_timeout=acceptable)
        case["evidence_paths"]["online_log"] = str(log_path)
        case["reason_zh"] = (f"处理到 {row['processing_seconds']:.3f} 秒达到时限；日志最后确认完成 {done}/{total} 个初值，"
            f"已有 {acceptable} 次初值结果满足局部接受条件，但候选尚未保存，无法核对最终位置或排除多解。")
        return case

    folders = list(attempt.glob("localization_unavailable/*"))
    assert len(folders) == 1, f"无法唯一定位失败记录：{sid}"
    folder = folders[0]
    progress = evidence.read(folder / "progress.json")
    for record in progress["artifacts"].values():
        evidence.verify(record)
    evidence.verify(progress["config_snapshot"])
    localization = evidence.read(folder / "localization_result.json")
    assert localization["status"] == row["status"] and localization["mu_m"] is None
    peaks_document = evidence.read(folder / "music_peaks.json")
    peaks = peaks_document.get("nominal") or peaks_document.get("coarse", [])
    case.update(music_peak_count=len(peaks), failed_step=progress["failed_step"],
                signal_rank=peaks_document["subspace_selection"]["signal_rank"],
                peaks=peak_truth_comparison(peaks, paths, period_ns))
    case["evidence_paths"].update(localization_result=str(folder / "localization_result.json"),
        music_peaks=str(folder / "music_peaks.json"), progress=str(folder / "progress.json"))
    assert len(peaks) == row["music_observation_count"]
    case["all_true_paths_share_one_diffraction_corner"] = bool(
        all(p["diffraction_count"] == 1 for p in paths)
        and all(np.linalg.norm(np.asarray(p["diffraction_corners_m"][0]) - paths[0]["diffraction_corners_m"][0]) < 1e-3 for p in paths))
    if case["all_true_paths_share_one_diffraction_corner"]:
        case["supporting_findings"].append("真实路径全部经过同一绕射拐角；在当前只使用角度和时延的模型下，重复提供距离加公共偏置的信息。")
    outside = [p for p in paths if not (config["music"]["delay_min_s"]*1e9 <= p["observed_delay_ns"] <= config["music"]["delay_max_s"]*1e9)]
    if outside:
        case["supporting_findings"].append("有真实带偏置时延超出搜索窗；逐路径记录已列明周期折回后的时延及是否仍落在窗外。这是并存现象，不自动视为本次停机原因。")

    if row["status"] == "unlocalizable":
        if row["rt_path_count"] == 1:
            assert len(peaks) == 1 and case["signal_rank"] == 1
            case.update(category="single_path", reason_zh=(f"进入 CSI 的有效路线只有一条（{paths[0]['path_type']}），"
                "从观测识别的信号成分和提取的粗峰也各一条；未知平面位置与公共时间偏置共三个量，单条路线提供的独立信息不足，程序在联合求解前退出。"))
        else:
            refine = peaks_document["refinement"]
            suppressed = refine["suppressed_refined_peaks"]
            assert len(paths) == 2 and len(peaks) == 1 and len(suppressed) == 1
            removed = suppressed[0]
            da = abs(np.rad2deg(removed["refined_aoa_local_rad"] - peaks[0]["aoa_rad"]))
            dt = abs(removed["refined_delay_s"] - peaks[0]["delay_s"]) * 1e9
            case.update(category="refinement_merge", refinement=refine,
                reason_zh=(f"原有 {len(paths)} 条真实路线，信号成分和粗峰均为 {case['signal_rank']}；"
                    f"细化后的两峰相差 {da:.3f}°、{dt:.3f} ns，同时小于当前 {config['music']['min_angle_separation_deg']:g}°、"
                    f"{config['music']['min_delay_separation_s']*1e9:g} ns 去重门槛，较弱峰被删除，只剩一峰而退出。"))
        return case

    if row["status"] == "excluded_observation":
        original = peaks_document["observation_screen"]
        recalculated = screen_music_observation([SimpleNamespace(**p) for p in peaks], **radio,
            settings=config["music"]["observation_screen"])
        assert len(original["excluded_pairs"]) == len(recalculated["excluded_pairs"])
        for left, right in zip(original["excluded_pairs"], recalculated["excluded_pairs"], strict=True):
            assert (left["first_peak_index"], left["second_peak_index"]) == (right["first_peak_index"], right["second_peak_index"])
            assert abs(left["response_correlation"] - right["response_correlation"]) < 1e-10
        pairs = original["excluded_pairs"]
        assert all(abs(p["angles_deg"][0]-p["angles_deg"][1]) > 150 for p in pairs)
        subset = maximum_separable_subset(len(peaks), pairs)
        descriptions = [f"{p['angles_deg'][0]:.3f}°/{p['angles_deg'][1]:.3f}°，"
            f"时延 {p['delays_ns'][0]:.3f}/{p['delays_ns'][1]:.3f} ns，相似度 {p['response_correlation']:.6f}" for p in pairs]
        case.update(category="response_screen", screening=original,
            maximum_separable_peak_count=len(subset), example_separable_peak_indices=subset,
            reason_zh=(f"{len(peaks)} 个峰中有 {len(pairs)} 对阵列两端响应触发整点排除：" + "；".join(descriptions)
                + f"。按同一门槛最多可保留 {len(subset)} 条互不触发筛选的观测；尚未验证其定位结果。"))
        return case

    true_check, bank_data, search = check_true_state(evidence, folder, scene, receiver, true_position, true_bias)
    best, diagnostics = search["best_candidate"], search["diagnostics"]
    case.update(offline_true_state_check=true_check,
        best_candidate=candidate_summary(best, true_position, true_bias),
        alternatives=[candidate_summary(x, true_position, true_bias) for x in search["alternatives"]],
        starts_attempted=diagnostics["starts_attempted"], hypothesis_count=diagnostics["hypothesis_count"],
        hypothesis_search_incomplete=diagnostics["hypothesis_search_incomplete"],
        seed_combinations_checked=diagnostics["seed_search"]["continuous_seed_combinations_checked"],
        start_termination_counts=dict(Counter(x["termination"] for x in diagnostics["starts"])),
        maximum_matched_count_across_starts=max(x["matched_observation_count"] for x in diagnostics["starts"]))
    closest_start = min(diagnostics["starts"], key=lambda x: np.linalg.norm(
        np.subtract(x["final_state_xy_beta_m"][:2], true_position)))
    case["closest_saved_start_offline"] = {**closest_start,
        "position_error_offline_m": float(np.linalg.norm(np.subtract(closest_start["final_state_xy_beta_m"][:2], true_position)))}
    case["evidence_paths"].update(continuous_search=str(folder / "continuous_search.json"),
                               propagation_hypotheses=str(folder / "propagation_hypotheses.json"))
    case["supporting_findings"].append(f"事后固定在真实位置和真实偏置处，用原观测及原路线库检查：匹配 {true_check['matched_observation_count']}/{len(peaks)} 条，"
        f"独立信息维数 {true_check['physical_rank']}/3，{'通过' if true_check['local_checks_passed'] else '未通过'}局部路径与残差条件。此检查没有执行优化，也没有验证唯一性。")
    if true_check["local_checks_passed"] and true_check["objective"] < best["objective"] - 1e-6:
        case["supporting_findings"].append(f"真实位置处的合法解释评分为 {true_check['objective']:.6f}，低于保存的最佳候选 {best['objective']:.6f}；"
            "说明有限搜索尚未找到评分更好的可行区域，不能仅依据现有候选判断整个问题无解或必然多解。")
    if row["status"] == "ambiguous":
        competitive = [x for x in search["alternatives"]
                       if x["objective"] <= best["objective"] + diagnostics["config"]["ambiguity_cost_tolerance"]]
        assert competitive
        other = competitive[0]
        gap = float(np.linalg.norm(np.subtract(other["position_m"], best["position_m"])))
        bias_gap = abs(other["clock_bias_s"] - best["clock_bias_s"]) * 1e9
        case.update(category="ambiguous", competitive_alternative_count=diagnostics["competitive_alternative_count"],
            first_competitive_position_gap_m=gap, first_competitive_bias_gap_ns=bias_gap,
            reason_zh=(f"最佳候选匹配 {best['matched_observation_count']}/{len(peaks)} 条观测，另一不同候选与其相距 {gap:.4f} m，"
                f"偏置相差 {bias_gap:.4f} ns；评分为 {best['objective']:.6f} 和 {other['objective']:.6f}，"
                f"差值未超过 {diagnostics['config']['ambiguity_cost_tolerance']:g} 的容差，因此拒绝选定一个位置。"))
    elif best["matched_observation_count"] < 2:
        missing = []
        for t in paths:
            if not t["diffraction_count"]:
                continue
            same_type = [h for h in bank_data["hypotheses"] if sum(k == "reflection" for k, _ in h["interactions"]) == t["reflection_count"]
                         and sum(k == "diffraction" for k, _ in h["interactions"]) == 1]
            same_corner = [h for h in same_type if np.linalg.norm(np.subtract(h["anchor_m"], t["diffraction_corners_m"][0])) < 1e-3]
            if not same_corner:
                missing.append({"truth_path_index": t["truth_path_index"], "path_type": t["path_type"],
                    "corner_m": t["diffraction_corners_m"][0], "same_type_library_count": len(same_type),
                    "same_type_and_corner_library_count": 0})
        assert missing and diagnostics["hypothesis_search_incomplete"]
        case.update(category="missing_route", missing_diffraction_routes=missing,
            reason_zh=(f"提取出 {len(peaks)} 条观测，但 {diagnostics['starts_attempted']} 个初值最多只匹配 "
                f"{case['maximum_matched_count_across_starts']} 条；路线库达到 {diagnostics['hypothesis_count']} 条上限。"
                + "；".join(f"真实{m['path_type']}经过拐角 {np.round(m['corner_m'],4).tolist()}，库中同类型有 {m['same_type_library_count']} 条、经过该拐角的为 0 条" for m in missing) + "。"))
    elif best["physical_rank"] < 3:
        corners = [p["interaction_points_m"][i] for p in best["selected_paths"]
                   for i, (kind, _) in enumerate(p["propagation_interactions"]) if kind == "diffraction"]
        assert len(corners) == best["matched_observation_count"] and all(np.linalg.norm(np.subtract(corners[0], x)) < 1e-3 for x in corners)
        aliases = [p for p in case["peaks"] if p["nearest_truth_with_period"]["delay_period_index"] != 0
                   and p["nearest_truth_with_period"]["distance_in_engineering_scales"] < 1]
        case.update(category="shared_corner", matched_corner_m=corners[0], delay_alias_peaks=aliases,
            reason_zh=(f"最佳候选匹配 {best['matched_observation_count']} 条路线，但都经过同一拐角 {np.round(corners[0],4).tolist()}，"
                f"只有 {best['physical_rank']} 份独立信息，无法确定位置两个坐标和公共偏置。"))
        if aliases:
            case["reason_zh"] += f"另有 {len(aliases)} 个峰与真实路径相差 {period_ns:g} ns 的整数周期，当前求解未处理该周期。"
    else:
        assert best["rejection_reasons"] == ["local_search_not_converged"]
        term = TERMINATIONS.get(best["termination"], best["termination"])
        case.update(category="not_converged", reason_zh=(f"最佳候选已匹配 {best['matched_observation_count']}/{len(peaks)} 条观测且独立信息达到 3/3，"
            f"但局部优化因“{term}”退出，未通过收敛检查。候选的事后位置误差为 {case['best_candidate']['position_error_offline_m']:.5f} m；"
            "这不是正式输出，也不能仅凭误差决定在线接受。"))
    return case


def publish(output, cases, summary):
    write_json(output / "diagnostics.json", {"summary": summary, "cases": cases})
    fields = ["sample_id", "category_zh", "status", "reason_zh", "rt_path_count", "music_peak_count",
        "true_x_m_offline", "true_y_m_offline", "processing_seconds", "maximum_separable_peak_count",
        "best_matched_count", "best_independent_rank", "best_termination", "best_error_m_offline",
        "true_state_matched_count_offline", "true_state_rank_offline", "true_state_local_checks_pass_offline",
        "supporting_findings_zh", "original_attempt_dir", "localization_record", "truth_archive"]
    with (output / "samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            best, check = case.get("best_candidate") or {}, case.get("offline_true_state_check") or {}
            row = {key: case.get(key) for key in fields if key in case}
            row.update(category_zh=CATEGORIES[case["category"]],
                true_x_m_offline=case["true_position_m_for_offline_analysis"][0],
                true_y_m_offline=case["true_position_m_for_offline_analysis"][1],
                best_matched_count=best.get("matched_observation_count"), best_independent_rank=best.get("physical_rank"),
                best_termination=best.get("termination"), best_error_m_offline=best.get("position_error_offline_m"),
                true_state_matched_count_offline=check.get("matched_observation_count"), true_state_rank_offline=check.get("physical_rank"),
                true_state_local_checks_pass_offline=check.get("local_checks_passed"), supporting_findings_zh="；".join(case["supporting_findings"]),
                localization_record=case["evidence_paths"].get("localization_result", case["evidence_paths"].get("online_log")),
                truth_archive=case["evidence_paths"]["truth_archive"])
            writer.writerow(row)
    lines = ["# 未输出位置样本逐点说明", "", f"完整覆盖 {len(cases)} 个无位置样本。每个编号单独列出直接原因、并存问题和原始证据。", "",
        "真值只在本次离线解释中使用；真实位置处的检查不代表定位成功，也不检验多解。未改写原实验结果。", "",
        f"[可筛选表格]({output / 'samples.csv'}) · [完整数值记录]({output / 'diagnostics.json'})", ""]
    for case in cases:
        lines.extend([f"## {case['sample_id']}：{CATEGORIES[case['category']]}", "", case["reason_zh"], "",
            f"真实路线 {case['rt_path_count']} 条；使用/提取的峰 {case['music_peak_count']} 个；耗时 {case['processing_seconds']:.3f} 秒。", ""])
        lines.extend([finding + "\n" for finding in case["supporting_findings"]])
        paths = case["evidence_paths"]
        lines.append("证据：" + " · ".join(f"[{label}]({paths[key]})" for key, label in [
            ("result", "原始状态"), ("music_peaks", "峰及筛选记录"), ("continuous_search", "求解过程"),
            ("online_log", "超时日志"), ("localization_result", "失败结果")] if key in paths))
        lines.append("")
    (output / "all_samples.md").write_text("\n".join(lines))
    lines = ["# 361 个未输出位置样本的完整复核", "", f"分析时间：{summary['created_at']}。", "",
        f"原实验：`{summary['experiment_root']}`。1000 个样本中 639 个输出位置，361 个未输出。", "",
        "| 直接原因 | 数量 |", "| --- | ---: |"]
    lines.extend(f"| {CATEGORIES[key]} | {summary['category_counts'].get(key, 0)} |" for key in CATEGORIES)
    lines.extend(["", "105 个单路径样本中：" + "、".join(f"{k} {v} 个" for k, v in summary["single_path_types"].items()) + "。",
        "", "筛选排除样本按保存门槛重新计算，每一对相似度均与原结果一致；可保留观测数量只是后续求解机会，不是恢复成功数量。",
        f"其中 {summary['screening']['subset_at_least_two']} 个最多可保留至少两条，{summary['screening']['subset_at_least_three']} 个至少三条。", "",
        "23 个进入求解但没有输出的样本，补做了固定在真实位置和偏置处的检查。该检查只使用旧路线库及旧观测，不搜索位置、不修改正式结果，也不检验收敛或唯一性。", "",
        "| 样本 | 直接原因 | 真值处匹配数/峰数 | 独立信息/3 | 真值处局部条件通过 |", "| --- | --- | ---: | ---: | --- |"])
    for case in cases:
        if "offline_true_state_check" in case:
            check = case["offline_true_state_check"]
            lines.append(f"| {case['sample_id']} | {CATEGORIES[case['category']]} | {check['matched_observation_count']}/{case['music_peak_count']} | {check['physical_rank']}/3 | {'是' if check['local_checks_passed'] else '否'} |")
    lines.extend(["", f"[全部 {len(cases)} 个样本的中文说明]({output / 'all_samples.md'})", "",
        f"[逐样本表格]({output / 'samples.csv'}) · [完整数值证据]({output / 'diagnostics.json'}) · [输入指纹及校验]({output / 'provenance.json'})", "",
        "本次未重新运行完整定位，未调整算法，未验证任何修复后的恢复数量。原结果、失败现场和旧分析均保留。"])
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    publish_interpretation(output, cases, summary)


def publish_interpretation(output, cases, summary):
    lookup = {c["sample_id"]: c for c in cases}
    def get(number):
        return lookup[f"SAMPLE_{number:06d}"]
    def ids(category):
        return "、".join(c["sample_id"].removeprefix("SAMPLE_") for c in cases if c["category"] == category)
    multi_corner = [c for c in cases if c.get("all_true_paths_share_one_diffraction_corner") and c["rt_path_count"] > 1]
    screen_corner = [c for c in multi_corner if c["category"] == "response_screen"]
    count_structural = summary["category_counts"]["single_path"] + len(multi_corner)
    c540, c764 = get(540), get(764)
    b764, t764 = c764["best_candidate"], c764["offline_true_state_check"]
    project = Path(__file__).resolve().parents[1]
    lines = ["# 361 个样本为什么没有输出位置", "",
        "本次把上一轮未展开的 106 个样本补齐，重新从原始结果逐个检查全部 361 个，另对 23 个进入求解后失败或多解的样本补做真值位置处的模型检查。原定位结果没有改动。", "",
        f"[361 个样本逐点中文说明]({output / 'all_samples.md'}) · [可筛选的完整表格]({output / 'samples.csv'}) · [汇总和真值处检查表]({output / 'summary.md'})", "",
        "## 直接停止原因", "", "| 原因 | 数量 | 样本编号（省略 SAMPLE_） |", "| --- | ---: | --- |"]
    for key in CATEGORIES:
        selected = [c for c in cases if c["category"] == key]
        lines.append(f"| {CATEGORIES[key]} | {len(selected)} | {ids(key) if len(selected) < 10 else '见完整逐点表'} |")
    lines.extend(["", "这些分类按第一次实际停止或最终拒绝的原因计数，互不重叠；同一个样本可以同时存在其他问题，已记在逐点说明中。", "",
        "## 105 个确实只有一条路径，000540 是去重损失", "",
        "单路径样本：一次反射 7 个、两次反射 48 个、一次反射后绕射 10 个、两次反射后绕射 40 个。所有 105 个的观测信号成分和粗峰数量也均为 1，没有证据表明它们是多条有效路线被漏检成一条。这里的有效路径仅指本次实际合成 CSI 所保留的路径，不代表真实世界只有这些传播路径。", "",
        "平面位置有两个未知量，公共时间偏置再增加一个。单条纯反射路线最多提供方向和路程两份信息；本实验的末端绕射路线方向还由固定拐角决定，只能提供到拐角距离与偏置的和。提高优化次数不能创造缺失的观测。", "",
        "000540 则原本含一次反射和两次反射各一条：", "",
        "| 路径 | 真实角度/° | 真实带偏置时延/ns |", "| --- | ---: | ---: |"])
    lines.extend(f"| {p['path_type']} | {p['angle_local_deg']:.6f} | {p['observed_delay_ns']:.6f} |" for p in c540["truth_paths"])
    lines.extend(["", c540["reason_zh"], "",
        "两条真实路径本身相差约 0.244°、1.219 ns。这个样本应该归到前端分辨及去重问题；它不等同于 105 个真实单路径样本。放宽去重是否能恢复两个可靠峰及位置，仍需验证。", "",
        "## 231 个筛选排除：大部分仍有观测可以继续用，但其中也混有信息不足", "",
        f"重新计算了全部 {summary['screening']['pairs']} 对触发筛选的响应，相似度均与原结果一致，全部出现在接近 −90° 和 +90° 的两端。半波长间距线阵在两端产生接近相同的响应；当前程序发现任一对相似度达到 0.995 就停止整份样本，尚未尝试其余观测。", "",
        "按相同门槛取互不触发筛选的最大观测子集，228 个可保留至少两条，220 个至少三条。只有 000208、000418、000751 最多剩一条。这个子集只解决响应重复，不证明观测真实、路线合法或位置唯一。", "",
        "本次进一步查出：" + "、".join(c["sample_id"].removeprefix("SAMPLE_") for c in screen_corner)
        + " 的所有真实路线均经过同一绕射拐角。它们不仅触发筛选，本身也缺少独立位置约束。因此不能把 231 个都当成只需删除一个重复峰就能恢复的样本。", "",
        f"加上另两例 000844、000995，以及 105 个单路径样本，至少 {count_structural} 个在当前角度与时延模型及本次保留路径下，已有明确的独立信息不足证据。此数是交叉分析，不能与上面的直接原因计数相加。", "",
        "## 16 个搜索失败要分别处理", "",
        f"**缺少正确路线的 5 个：{ids('missing_route')}。** 两个峰已存在，但真实的两次反射后绕射路线没有进入 4096 条上限的库。四例该类型数量为零；000748 虽有一条，但拐角不同。事后固定在真实位置处，五例也都只能匹配 1/2 条、独立信息 2/3。因此仅增加后续优化迭代无效，需先解决路线构造覆盖。", "",
        f"**只匹配同一拐角的 5 个：{ids('shared_corner')}。** 都只有独立信息 1/3。000432、000865、000939 另有一条纯反射路径，其真实带偏置时延分别约 1363、1375、1373 ns，被提取为约 83、95、93 ns，相差同一个 1280 ns 周期。当前模型直接比较时延，没有表示这个整数周期，因而损失了额外的独立信息。000844 也有折回，但全部三条真实路线都经过同一拐角，修正时延周期仍不能凭当前信息唯一定位；000995 本来就只有同拐角的两条路线。", "",
        f"**未收敛的 6 个：{ids('not_converged')}。** 最佳候选已有足够匹配数和独立信息，但未满足数值停止条件。这是直接拒绝原因，底层情况不同：", "",
        "| 样本 | 直接终止 | 候选事后误差/m | 真值处匹配/峰数 | 解释 |", "| --- | --- | ---: | ---: | --- |"])
    explanations = {
        76: "第二峰比最接近真实路径的时延偏离 35.529 ns；真实位置只能匹配一条，不能仅放宽收敛条件。",
        92: "真实位置可匹配全部三条；保存的最佳候选误差较大，角度偏差、几何敏感性与有限搜索并存。",
        583: "四条真实路径的峰都很接近真值；候选仅偏约 2.49 cm，却因步长停滞拒绝，值得优先检查数值终止。",
        715: "真实位置处四条匹配、局部条件通过，且评分比保存的最佳候选更低；还有一条观测未解释。",
        778: "含另一端的混淆峰以及 1150 ns 边界峰；两条真实长路径落在当前周期搜索窗口的缺口。",
        921: "真实位置可匹配两条，峰也接近真值；约束对扰动较敏感，误差 4.15 m 不能只用低评分解释为正确。",
    }
    for number, explanation in explanations.items():
        case = get(number); best, check = case["best_candidate"], case["offline_true_state_check"]
        lines.append(f"| {number:06d} | {TERMINATIONS[best['termination']]} | {best['position_error_offline_m']:.5f} | {check['matched_observation_count']}/{case['music_peak_count']} | {explanation} |")
    lines.extend(["", "真实位置处的检查只看局部路线、误差和信息独立性，不进行迭代，也不要求解释全部观测。例如 000583 在真实位置处匹配 3/4 条，偏移约 2.49 cm 的候选可匹配 4/4 条；这说明地图几何合法性检查本身也需要保留，不能凭真值接近直接放行。", "",
        "## 7 个多解并非都证明真实场景无法区分", "",
        "| 样本 | 最佳候选匹配/峰数 | 候选位置间距/m | 候选偏置差/ns | 最佳评分 | 另一竞争评分 |", "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for case in cases:
        if case["category"] == "ambiguous":
            best = case["best_candidate"]
            other = next(a for a in case["alternatives"] if a["objective"] <= best["objective"] + .5)
            lines.append(f"| {case['sample_id'].removeprefix('SAMPLE_')} | {best['matched_observation_count']}/{case['music_peak_count']} | {case['first_competitive_position_gap_m']:.4f} | {case['first_competitive_bias_gap_ns']:.4f} | {best['objective']:.6f} | {other['objective']:.6f} |")
    lines.extend(["", f"**000764 有额外的有限搜索问题。** 保存的最佳候选只解释 2/3 条，评分 {b764['objective']:.6f}；固定在真实位置处，同一库和观测能解释 3/3 条，评分仅 {t764['objective']:.6f}。这提供了一个比现有候选明显更好的合法位置证据，说明不能将它概括为场景必然多解。96 个初值保存的最终状态中，最近者仍离真值约 {c764['closest_saved_start_offline']['position_error_offline_m']:.3f} m，且未收敛。", "",
        "000472、000870、000874 的候选位置和偏置变化很大而评分变化很小，存在几何区分能力弱及匹配不完整的情况；000890 两个候选均只解释 4/9 条，五条观测各付固定惩罚形成共同的 80 分基础。当前 0.5 分的多解门槛没有按实际噪声校准，最终也没有重建整份 CSI 作区分。不能靠强行选最低分来验证恢复成功。", "",
        "## 000755 超时", "", get(755)["reason_zh"], "",
        "11 次满足接受条件的初值不等于 11 个不同位置。超时前候选未持久化，因此现有记录无法核对最终位置误差。", "",
        "## 处理顺序", "",
        "1. 优先改进两端方向混淆的处理，保留其余观测，并保留方向的多种解释供地图验证；覆盖数量最大。",
        "2. 补齐有限路线库中的绕射路线，并处理时延相差整数周期及搜索窗缺口。",
        "3. 分别检查 000540 的近邻去重、000583 等的数值终止，以及 000764 的初值与路线搜索；不要把所有未收敛候选统一放行。",
        "4. 对确定缺少独立信息的样本保留无唯一位置结果；若要恢复，需要额外独立观测、时间约束或其他经过验证的信息。",
        "5. 保存逐初值候选和耗时，并以更多完整观测检验多解。", "",
        "上述顺序来自本次证据与涉及数量，不是修改后的效果保证。", "",
        "## 核对边界与代码", "",
        "当前包内 50 个源文件与原实验指纹一致；逐样本状态与原 1000 点汇总完全一致。3748 次记录指纹校验通过，5112 个读取输入在分析结束时复查无变化。未重跑完整定位，也未修改原结果。所有真值使用仅发生于本次独立离线分析。", "",
        f"- [整点筛选]({project / 'src/time_bias_localization/observation_screen.py'}:26)",
        f"- [细峰去重]({project / 'src/time_bias_localization/spectrum_sampling.py'}:196)",
        f"- [路线数量上限]({project / 'src/time_bias_localization/propagation_hypotheses.py'}:388)",
        f"- [当前时延残差]({project / 'src/time_bias_localization/continuous_solver.py'}:142)",
        f"- [数值终止条件]({project / 'src/time_bias_localization/continuous_solver.py'}:247)",
        f"- [多解判定]({project / 'src/time_bias_localization/continuous_solver.py'}:724)", ""])
    (output / "analysis.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("输出必须与只读输入目录分开")
    output.mkdir(parents=True, exist_ok=False)
    evidence = Evidence()
    experiment = evidence.read(source / "experiment.json")
    original_summary = evidence.read(source / "report/summary.json")
    source_root = Path(__file__).resolve().parents[1] / "src/time_bias_localization"
    for filename, expected in experiment["source"]["files"].items():
        evidence.verify({"path": str(source_root / filename), "sha256": expected})
    scene_paths = list((source / "channel_setups").glob("*/scene/scene_2d.json"))
    assert len(scene_paths) == 1
    scene = Scene2D.from_dict(evidence.read(scene_paths[0]))
    rows = [evidence.read(p) for p in sorted((source / "samples").glob("*/result.json"))]
    assert Counter(r["status"] for r in rows) == original_summary["status_counts"]
    failed = [r for r in rows if r["estimated_position_m"] is None]
    assert len(rows) == 1000 and len(failed) == 361, "本脚本针对当前 1000 点实验，数量变化时应重新审查分类规则"
    cases = []
    for i, row in enumerate(failed, 1):
        case = analyze_case(evidence, row, experiment, scene)
        cases.append(case)
        if i % 20 == 0 or row["status"] in ("solver_budget_exhausted", "ambiguous", "timeout") or i == len(failed):
            print(f"[逐点核对] {i}/{len(failed)} {case['sample_id']}：{CATEGORIES[case['category']]}", flush=True)
    screened = [c for c in cases if c["category"] == "response_screen"]
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "experiment_root": str(source),
        "sample_count": len(rows), "no_position_count": len(cases),
        "status_counts": dict(Counter(c["status"] for c in cases)),
        "category_counts": dict(Counter(c["category"] for c in cases)),
        "single_path_types": dict(Counter(c["truth_paths"][0]["path_type"] for c in cases if c["category"] == "single_path")),
        "screening": {"pairs": sum(len(c["screening"]["excluded_pairs"]) for c in screened),
            "subset_size_histogram": dict(Counter(c["maximum_separable_peak_count"] for c in screened)),
            "subset_at_least_two": sum(c["maximum_separable_peak_count"] >= 2 for c in screened),
            "subset_at_least_three": sum(c["maximum_separable_peak_count"] >= 3 for c in screened)},
        "offline_true_state_checks": sum("offline_true_state_check" in c for c in cases),
        "offline_true_state_checks_passed": [c["sample_id"] for c in cases if c.get("offline_true_state_check", {}).get("local_checks_passed")],
        "all_true_paths_one_diffraction_corner": [c["sample_id"] for c in cases if c.get("all_true_paths_share_one_diffraction_corner")],
        "online_localization_rerun": False, "uses_truth_only_for_offline_analysis": True,
        "source_files_match_experiment": True}
    print(f"[来源检查] 重新核对 {len(evidence.files)} 个输入文件，确认分析期间未变化。", flush=True)
    evidence.recheck()
    provenance = {"input_files_sha256": evidence.files, "verified_reference_count": evidence.verified_references,
        "source_package_files_checked": len(experiment["source"]["files"]),
        "audit_script_sha256": digest(__file__), "input_unchanged_after_analysis": True,
        "truth_state_check_scope": "offline_only_fixed_state_no_optimization_no_uniqueness_check"}
    write_json(output / "provenance.json", provenance)
    publish(output, cases, summary)
    print(f"[完成] 已覆盖 {len(cases)} 个样本；汇总={output / 'summary.md'}；逐点说明={output / 'all_samples.md'}", flush=True)


if __name__ == "__main__":
    main()
