"""只读检查候选路线的公式重复、保存候选的下降方向及最低分选择。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np

from analyze_no_position_samples import Evidence, digest, write_json
from time_bias_localization.continuous_solver import ContinuousSolverConfig, _Problem, _huber
from time_bias_localization.propagation_hypotheses import HypothesisBank
from time_bias_localization.propagation_model import ContinuousObservation, PropagationHypothesis, evaluate_hypothesis
from time_bias_localization.scene import Scene2D


def load_bank(evidence, source, number, scene):
    row = evidence.read(source / "samples" / f"SAMPLE_{number:06d}" / "result.json")
    evidence.verify(row["solve_record"])
    folder, = Path(row["attempt_dir"]).glob("localization_unavailable/*")
    progress = evidence.read(folder / "progress.json")
    for name in ("continuous_search", "propagation_hypotheses", "continuous_observations"):
        evidence.verify(progress["artifacts"][name])
    saved = evidence.read(folder / "propagation_hypotheses.json")
    observed = evidence.read(folder / "continuous_observations.json")
    search = evidence.read(folder / "continuous_search.json")
    hypotheses = []
    for value in saved["hypotheses"]:
        item = dict(value)
        item["interactions"] = tuple(tuple(x) for x in item["interactions"])
        hypotheses.append(PropagationHypothesis(**item))
    bank = HypothesisBank(scene, hypotheses[0].receiver_m,
        tuple(ContinuousObservation(**x) for x in observed["observations"]), tuple(hypotheses),
        tuple(tuple(x) for x in saved["observation_hypothesis_indices"]), saved["search_report"])
    return row, bank, search


def formula_vector(hypothesis):
    # 两次反射的到达角还依赖矩阵方向，不能只按等效位置去重。
    return np.r_[np.asarray(hypothesis.affine_image_matrix).ravel(),
        np.asarray(hypothesis.affine_image_offset) - hypothesis.anchor_m,
        hypothesis.fixed_length_m, hypothesis.fixed_aoa_rad or 0., hypothesis.fixed_aoa_rad is not None]


def formula_counts(hypotheses, decimals):
    groups = defaultdict(list)
    for hypothesis in hypotheses:
        vector = formula_vector(hypothesis)
        key = tuple(vector if decimals is None else np.round(vector, decimals))
        groups[key].append(hypothesis)
    spread = max((float(np.ptp(np.stack([formula_vector(h) for h in group]), axis=0).max())
                  for group in groups.values()), default=0.)
    biggest = max(groups.values(), key=len)
    return {"rounding_decimals": decimals, "count": len(hypotheses), "formula_groups": len(groups),
        "repeated_formula_entries": len(hypotheses) - len(groups), "largest_group": len(biggest),
        "max_coefficient_spread_within_group": spread,
        "example_group": [{"hypothesis_id": h.hypothesis_id, "interactions": h.interactions} for h in biggest],
        "meaning": "formula_computation_redundancy_only_finite_wall_domains_must_be_preserved"}


def wall_inventory(scene):
    groups = defaultdict(list)
    endpoints = set()
    for wall in scene.walls:
        normal = wall.normal.copy()
        if normal[0] < -1e-10 or abs(normal[0]) <= 1e-10 and normal[1] < 0:
            normal *= -1
        groups[tuple(np.round(np.r_[normal, normal @ wall.start], 9))].append(wall.wall_id)
        endpoints.add(tuple(sorted([tuple(wall.start), tuple(wall.end)])))
    return {"wall_segments": len(scene.walls), "identical_endpoint_duplicates": len(scene.walls) - len(endpoints),
        "supporting_line_groups_round9": len(groups),
        "multiple_segments_on_same_supporting_line": sum(len(g) > 1 for g in groups.values()),
        "scope": "line_formula_inventory_not_permission_to_bridge_gaps_or_merge_diffraction_edges"}


def gradient_check(bank, search):
    config = ContinuousSolverConfig(**search["diagnostics"]["config"])
    problem = _Problem(bank, config)
    best = search["best_candidate"]
    state = np.r_[best["position_m"], best["beta_m"]]
    assignment, score, _ = problem.assignment(state)
    assert len(assignment) == best["matched_observation_count"]
    residuals, jacobian = problem.residual_jacobian(state, assignment)
    norms = np.linalg.norm(residuals, axis=1)
    weights = np.minimum(1., config.huber_delta / np.maximum(norms, 1e-15))
    j = (jacobian * np.sqrt(weights)[:, None, None]).reshape(-1, 3)
    r = (residuals * np.sqrt(weights)[:, None]).ravel()
    gradient = j.T @ r
    diagonal = np.maximum(np.diag(j.T @ j), 1e-12)
    projected = gradient.copy()
    at_lower = state <= problem.bounds[:, 0] + config.convergence_step_m
    at_upper = state >= problem.bounds[:, 1] - config.convergence_step_m
    projected[at_lower & (gradient > 0)] = 0.
    projected[at_upper & (gradient < 0)] = 0.
    scaled_norm = float(np.linalg.norm(projected / np.sqrt(diagonal)))
    direction = -projected / max(np.linalg.norm(projected), 1e-30)
    base = float(np.sum(_huber(norms ** 2, config.huber_delta)))
    probes = []
    for step in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        proposed = state + step * direction
        in_bounds = bool(problem.in_bounds(proposed))
        value, _ = problem.residual_jacobian(proposed, assignment)
        proposed_score = float(np.sum(_huber(np.sum(value ** 2, axis=1), config.huber_delta)))
        evaluations = [evaluate_hypothesis(bank.scene, bank.hypotheses[k], proposed[:2]) for _, k in assignment]
        invalid = [{"hypothesis_id": bank.hypotheses[k].hypothesis_id,
                    "reason": evaluation.invalid_reason}
                   for (_, k), evaluation in zip(assignment, evaluations) if not evaluation.valid]
        legal = in_bounds and not invalid
        probes.append({"step_m": step, "in_bounds": in_bounds, "all_selected_paths_legal": bool(legal),
                       "fixed_assignment_cost_decrease": base - proposed_score,
                       "position_change_m": (proposed[:2] - state[:2]).tolist(),
                       "beta_change_m": float(proposed[2] - state[2]), "invalid_paths": invalid})
    return {"saved_termination": best["termination"], "saved_converged": best["converged"],
        "saved_best_objective": best["objective"], "recomputed_objective": score,
        "projected_scaled_gradient_norm": scaled_norm, "current_gradient_threshold": 1e-9,
        "passes_current_gradient_test": scaled_norm <= 1e-9, "nearby_descent_probes": probes,
        "scope": "saved_candidate_fixed_assignment_no_optimizer_rerun_no_truth_inputs",
        "last_damping_and_last_step_saved": False}


def physical_paths_at_saved_candidate(bank, search):
    best = search["best_candidate"]
    reasons = Counter()
    valid = []
    for i, hypothesis in enumerate(bank.hypotheses, 1):
        result = evaluate_hypothesis(bank.scene, hypothesis, best["position_m"])
        if result.valid:
            valid.append({"hypothesis_id": hypothesis.hypothesis_id, "interactions": hypothesis.interactions,
                          "nodes_m": result.path.nodes.tolist()})
        else:
            reasons[result.invalid_reason] += 1
        if i % 1024 == 0:
            print(f"[固定候选位置检查] {i}/{len(bank.hypotheses)}", flush=True)
    keys = {tuple(np.round(np.asarray(v["nodes_m"]).ravel(), 6)) for v in valid}
    return {"position_m": best["position_m"], "uses_truth": False, "valid_hypotheses": len(valid),
        "valid_distinct_node_sequences_round6": len(keys), "invalid_reason_counts": dict(reasons),
        "valid_paths": valid, "scope": "one_saved_candidate_only_not_global_domain_pruning"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("输出目录必须与原实验分开")
    output.mkdir(parents=True, exist_ok=False)
    evidence = Evidence()
    experiment = evidence.read(source / "experiment.json")
    package = Path(__file__).resolve().parents[1] / "src/time_bias_localization"
    for name, expected in experiment["source"]["files"].items():
        evidence.verify({"path": str(package / name), "sha256": expected})
    scene_path, = (source / "channel_setups").glob("*/scene/scene_2d.json")
    scene = Scene2D.from_dict(evidence.read(scene_path))
    result = {"input_root": str(source), "wall_inventory": wall_inventory(scene),
              "banks": {}, "stalled_candidates": {}, "best_of_ambiguous_offline": []}
    for number in (103, 414, 685, 748, 820):
        row, bank, search = load_bank(evidence, source, number, scene)
        rr = [h for h in bank.hypotheses if h.reflection_order == 2 and h.diffraction_order == 0]
        result["banks"][row["sample_id"]] = {"unique_id_count": len({h.hypothesis_id for h in bank.hypotheses}),
            "unique_sequence_count": len({h.interactions for h in bank.hypotheses}),
            "all_formulas": [formula_counts(bank.hypotheses, d) for d in (None, 12, 9, 6)],
            "two_reflection_formulas": formula_counts(rr, 9),
            "original_geometry_checks": search["diagnostics"]["path_checks"],
            "original_invalid_geometry_checks": search["diagnostics"]["invalid_path_checks"]}
        print(f"[路线公式] {row['sample_id']}：{len(bank.hypotheses)} 条编号；"
              f"按 9 位小数区分的公式 {result['banks'][row['sample_id']]['all_formulas'][2]['formula_groups']} 组", flush=True)
        if number == 103:
            began = time.perf_counter()
            result["sample103_fixed_candidate_geometry"] = physical_paths_at_saved_candidate(bank, search)
            result["sample103_fixed_candidate_geometry"]["diagnostic_elapsed_s"] = time.perf_counter() - began
            example = result["banks"][row["sample_id"]]["two_reflection_formulas"]["example_group"]
            wall_ids = {key for h in example for _, key in h["interactions"]}
            result["example_wall_fragments"] = [{"wall_id": w.wall_id, "start_m": w.start_m,
                "end_m": w.end_m, "source_object": w.source_object} for w in scene.walls if w.wall_id in wall_ids]
    for number in (92, 583, 715, 778, 921):
        row, bank, search = load_bank(evidence, source, number, scene)
        result["stalled_candidates"][row["sample_id"]] = gradient_check(bank, search)
        value = result["stalled_candidates"][row["sample_id"]]
        print(f"[停滞候选] {row['sample_id']}：归一后的梯度={value['projected_scaled_gradient_norm']:.6g}", flush=True)
    for number in (434, 472, 764, 870, 874, 890, 897):
        row, bank, search = load_bank(evidence, source, number, scene)
        best = search["best_candidate"]
        assert best["acceptable"]
        # 仅此汇总读取已保存的评估真值，用于事后比较直接选最低分的结果。
        error = float(np.linalg.norm(np.subtract(best["position_m"], row["true_position_m"])))
        result["best_of_ambiguous_offline"].append({"sample_id": row["sample_id"],
            "best_objective": best["objective"], "best_position_m": best["position_m"],
            "position_error_offline_m": error, "does_not_change_original_result": True})
        if number == 434:
            alternative = search["alternatives"][0]
            result["sample434_score_gap"] = {
                "best_position_m": best["position_m"], "best_objective": best["objective"],
                "alternative_position_m": alternative["position_m"], "alternative_objective": alternative["objective"],
                "objective_gap": alternative["objective"] - best["objective"],
                "position_distance_m": float(np.linalg.norm(np.subtract(best["position_m"], alternative["position_m"]))),
                "current_gap_threshold": search["diagnostics"]["config"]["ambiguity_cost_tolerance"]}
    evidence.recheck()
    write_json(output / "review.json", result)
    write_json(output / "provenance.json", {"inputs_sha256": evidence.files,
        "reference_checks": evidence.verified_references, "inputs_unchanged": True,
        "review_script_sha256": digest(__file__),
        "helper_script_sha256": digest(Path(__file__).with_name("analyze_no_position_samples.py")),
        "online_code_changed": False, "full_localization_rerun": False})
    lines = ["# 候选路线重复和停止条件复查", "", f"原实验：`{source}`。仅读取已有结果，不修改定位器。", "",
        "## 候选公式", "", "公式系数按小数点后 9 位分组，保留矩阵、偏移、固定段长度和固定角度；不是只按距离相近合并。", "",
        "| 样本 | 候选编号数 | 公式组数 | 重复公式条目 | 二次反射条目 | 二次反射公式组数 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for sid, bank in result["banks"].items():
        all_f, rr = bank["all_formulas"][2], bank["two_reflection_formulas"]
        lines.append(f"| {sid} | {bank['unique_id_count']} | {all_f['formula_groups']} | {all_f['repeated_formula_entries']} | {rr['count']} | {rr['formula_groups']} |")
    geometry = result["sample103_fixed_candidate_geometry"]
    lines.extend(["", f"000103 在原来保存的最佳候选位置处，逐一检查 4096 个走法，几何合法的只有 {geometry['valid_hypotheses']} 个；"
        f"按反射点坐标区分也为 {geometry['valid_distinct_node_sequences_round6']} 条。此检查不使用真实位置，仅说明候选公式数不等于某个位置的有效路径数。", "",
        "同一支撑直线上的不同有限墙段可以对应相同公式，却拥有不同的合法范围。应共享公式计算并保留各段范围，不能直接删除所有同公式条目，也不能跨越门洞合并墙体。", "",
        "## 步长停滞的候选", "", "在保存候选处重算当前路径配对的梯度，并检查沿下降方向的六个小步。没有重新运行优化器，没有使用真实位置。", "",
        "| 样本 | 归一后的梯度 | 当前梯度门槛 | 检查的小步中存在合法下降 |", "| --- | ---: | ---: | --- |"])
    for sid, value in result["stalled_candidates"].items():
        has_descent = any(p["all_selected_paths_legal"] and p["fixed_assignment_cost_decrease"] > 0
                          for p in value["nearby_descent_probes"])
        lines.append(f"| {sid} | {value['projected_scaled_gradient_norm']:.9g} | 1e-9 | {'是' if has_descent else '否'} |")
    lines.extend(["", "000778 在所检查的方向上仍能合法降低误差，因此保存的位置尚未达到局部最小值。"
        "另外四个样本在所检查的方向和步长下，降低公式误差会使至少一条路径无法通过几何检查。"
        "这不能证明所有方向都无法下降，也不能排除候选已经位于可行区域的局部最优边界。", "",
        "旧记录没有保存最后一次压步长参数和局部求解内部状态，因此这里解释的是保存候选附近的现象，不能恢复原优化器最后一次失败的全部过程。", "",
        "## 如果对七个多解样本直接选最低分", "", "以下只对已有候选重新排序并作事后误差统计，不是新一轮定位。", "",
        "| 样本 | 最低分 | 对应位置误差/m |", "| --- | ---: | ---: |"])
    for row in result["best_of_ambiguous_offline"]:
        lines.append(f"| {row['sample_id']} | {row['best_objective']:.6f} | {row['position_error_offline_m']:.6f} |")
    gap = result["sample434_score_gap"]
    lines.extend(["", f"000434 的最低分是 {gap['best_objective']:.9f}，第二个不同位置的分数为 {gap['alternative_objective']:.9f}；"
        f"两位置相距 {gap['position_distance_m']:.6f} m，分差 {gap['objective_gap']:.9f} 小于当前门槛 {gap['current_gap_threshold']}。"
        "分差检查是额外的多解提示规则，并不是求最低分位置所必需的计算；当前实现同时用它阻止位置输出。", "",
        "## 当前代码位置", "",
        f"- 在线候选生成与编号去重：[propagation_hypotheses.py]({package / 'propagation_hypotheses.py'}:76)。",
        f"- 原始网格切片及相同端点去重：[scene.py]({package / 'scene.py'}:530)。",
        f"- 步长停止条件：[continuous_solver.py]({package / 'continuous_solver.py'}:243)。",
        f"- 多解分差及是否输出位置：[continuous_solver.py]({package / 'continuous_solver.py'}:734)。",
        f"- 离线信道生成调用 Sionna：[boundary_channel.py]({package / 'boundary_channel.py'}:422)。"])
    lines.extend(["", f"完整数值和原始文件指纹见 [review.json]({output / 'review.json'})、[provenance.json]({output / 'provenance.json'})。", "",
        "本次未重跑超时样本，不能据此报告该样本的去重后加速比；1800 秒上限未改动。", ""])
    (output / "review.md").write_text("\n".join(lines))
    print(f"[完成] 结果：{output / 'review.md'}", flush=True)


if __name__ == "__main__":
    main()
