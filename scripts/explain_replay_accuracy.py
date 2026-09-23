"""只读：解释恢复样本为什么精度高——逐观测几何可行域的交集面积，以及求解器诊断。

对指定样本，取重放时选中的分支，做两件事：
1. 重算该分支的求解结果，输出逐观测残差、雅可比奇异值、条件数与条件协方差；
2. 在地图上逐格求每条观测的几何可行域，并按观测逐条求交，输出面积收缩过程。
真值只用于最后的离线比较。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np

from time_bias_localization.continuous_pipeline import run_continuous_with_angle_branches
from time_bias_localization.observation_screen import ambiguity_groups
from time_bias_localization.propagation_model import evaluate_hypothesis
from time_bias_localization.scene import Scene2D
import scripts.plot_observation_loci as L
import scripts.measure_region_vs_box as M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True, help="重放目录下的 report/replay.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", default="SAMPLE_000015")
    parser.add_argument("--step", type=float, default=3.0)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    replay = json.loads(args.replay.read_text())
    case, = [item for item in replay["cases"] if item["sample_id"] == args.sample]

    sample_dir = source / "samples" / args.sample
    result_row = json.loads((sample_dir / "result.json").read_text())
    folder, = (Path(result_row["attempt_dir"]) / "localization_unavailable").glob("*")
    peaks_document = json.loads((folder / "music_peaks.json").read_text())
    config = json.loads((folder / "localization_config.json").read_text())["resolved_config"]
    config["music"]["observation_screen"]["ambiguity_policy"] = "enumerate_branches"
    scene_path, = (source / "channel_setups").glob("*/scene/scene_2d.json")
    scene = Scene2D.from_dict(json.loads(scene_path.read_text()))
    observation_path = json.loads((sample_dir / "observation.json").read_text())
    with np.load(observation_path["artifacts"]["truth_npz"]["path"], allow_pickle=False) as truth:
        true_position = np.asarray(truth["ue_position_m"], float)

    peaks = [SimpleNamespace(aoa_rad=float(v["aoa_rad"]), delay_s=float(v["delay_s"]))
             for v in peaks_document["nominal"]]
    indices = list(peaks_document["nominal_source_indices"])
    groups = ambiguity_groups(peaks_document["observation_screen"]["excluded_pairs"], len(peaks))
    bs = np.asarray(config["radio"]["bs_position_m"], float)
    boresight = math.radians(float(config["radio"]["bs_boresight_deg"]))

    selection = run_continuous_with_angle_branches(config, scene, peaks, indices, groups, bs, boresight)
    branch = selection.report["branches"][selection.report["selected_branch"]]
    kept = branch["kept_peak_positions"]
    best = selection.result["best_candidate_for_diagnostics_only"]
    solver = {
        "selected_branch": selection.report["selected_branch"],
        "observation_count": len(kept),
        "matched_observation_count": best["matched_observation_count"],
        "physical_rank": best["physical_rank"],
        "jacobian_singular_values": best["jacobian_singular_values"],
        "condition_number": best["condition_number"],
        "objective": best["objective"],
        "covariance_xy_beta_m2": (best.get("covariance") or {}).get("matrix_xy_beta_m2"),
        "observation_residuals": [
            {"observation_id": row["observation_id"],
             "aoa_error_deg": math.degrees(row["angle_residual_rad"]),
             "length_error_m": row["length_residual_m"],
             "normalized_residual_norm": row["normalized_residual_norm"],
             "interactions": row["propagation_interactions"]}
            for row in best["selected_paths"]],
        "solution_m": best["position_m"], "beta_m": best["beta_m"],
        "position_error_m": float(np.linalg.norm(np.asarray(best["position_m"]) - true_position)),
        "true_position_m": true_position.tolist(),
    }
    if solver["covariance_xy_beta_m2"] is not None:
        matrix = np.asarray(solver["covariance_xy_beta_m2"])
        eigenvalues = np.linalg.eigvalsh(matrix[:2, :2])
        solver["sigma_xy_m"] = float(np.sqrt(max(eigenvalues.max(), 0.0)))
        solver["sigma_xy_eigenvalues_m2"] = eigenvalues.tolist()

    # 逐观测几何可行域与交集
    x_min, x_max, y_min, y_max = scene.bounds_m
    xs = np.arange(x_min + args.step / 2, x_max, args.step)
    ys = np.arange(y_min + args.step / 2, y_max, args.step)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    grid = np.c_[mesh_x.ravel(), mesh_y.ravel()]
    cell = args.step ** 2
    walls = {wall.wall_id: wall for wall in scene.walls}
    hidden = M._fully_hidden_walls_from_receiver(walls, bs)
    from time_bias_localization.propagation_hypotheses import build_hypothesis_bank
    from time_bias_localization.propagation_model import ContinuousObservation
    bank = build_hypothesis_bank(scene, bs, [ContinuousObservation(
        observation_id=f"music_path_{int(indices[i]):02d}",
        aoa_rad=peaks[i].aoa_rad + boresight,
        observed_length_m=299792458.0 * peaks[i].delay_s,
        angle_scale_rad=math.radians(1.0), length_scale_m=0.75) for i in kept],
        max_reflections=int(config["scene"]["max_reflections"]),
        max_diffractions=int(config["scene"].get("max_diffractions", 0)),
        diffraction_position=config["scene"].get("diffraction_position", "any"),
        max_hypotheses=int(config["localization"]["continuous"]["max_hypotheses"]),
        max_enumerated_sequences=int(config["localization"]["continuous"]["max_enumerated_sequences"]),
        aoa_gate_rad=math.radians(float(config["localization"]["continuous"]["aoa_gate_deg"])),
        beta_interval_m=(float(config["localization"]["bias_min_s"]) * 299792458.0,
                         float(config["localization"]["bias_max_s"]) * 299792458.0),
        length_gate_sigma=float(config["localization"]["continuous"]["length_gate_sigma"]))
    gate = math.radians(float(config["localization"]["continuous"]["aoa_gate_deg"]))
    beta_lo = float(config["localization"]["bias_min_s"]) * 299792458.0
    beta_hi = float(config["localization"]["bias_max_s"]) * 299792458.0
    survivors = [index for index, hypothesis in enumerate(bank.hypotheses)
                 if M._receiver_geometry_failure(hypothesis.interactions, walls,
                                                 np.asarray(hypothesis.receiver_m), hidden) is None]

    masks, shrink = [], []
    for observation_index, observation in enumerate(bank.observations):
        mask = np.zeros(len(grid), bool)
        for index in bank.observation_hypothesis_indices[observation_index]:
            if index not in survivors:
                continue
            hypothesis = bank.hypotheses[index]
            candidate = L.analytic_mask(hypothesis, observation, grid, gate,
                                        observation.length_scale_m * float(
                                            config["localization"]["continuous"]["length_gate_sigma"]),
                                        beta_lo, beta_hi)
            if not np.any(candidate):
                continue
            subset = grid[candidate]
            valid = np.asarray([bool(evaluate_hypothesis(scene, hypothesis, point).valid)
                                for point in subset])
            mask[candidate] |= valid
        masks.append(mask)
        running = np.ones(len(grid), bool)
        for value in masks:
            running &= value
        shrink.append({"observation_id": observation.observation_id,
                       "aoa_deg": math.degrees(observation.aoa_rad),
                       "observed_length_m": observation.observed_length_m,
                       "valid_points": int(mask.sum()),
                       "valid_area_m2": float(mask.sum() * cell),
                       "intersection_points": int(running.sum()),
                       "intersection_area_m2": float(running.sum() * cell)})
    final = np.ones(len(grid), bool)
    for value in masks:
        final &= value
    points = grid[final]
    summary = {
        "sample_id": args.sample, "grid_step_m": args.step,
        "map_area_m2": (x_max - x_min) * (y_max - y_min),
        "observation_count": len(kept),
        "kept_peak_positions": kept, "ambiguity_groups": groups,
        "per_observation": shrink,
        "final_intersection_points": int(final.sum()),
        "final_intersection_area_m2": float(final.sum() * cell),
        "final_intersection_contains_truth": bool(
            len(points) and np.min(np.linalg.norm(points - true_position, axis=1)) <= args.step),
        "nearest_intersection_point_to_truth_m": (float(np.min(np.linalg.norm(points - true_position, axis=1)))
                                                  if len(points) else None),
        "solver": solver,
        "uses_truth_only_for_offline_comparison": True,
    }
    (output / "explanation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    figure, ax = plt.subplots(figsize=(11, 11))
    for wall in scene.walls:
        ax.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]],
                color="0.5", lw=0.9, zorder=2)
    palette = plt.get_cmap("tab10")
    for order, mask in enumerate(masks):
        points = grid[mask]
        if not len(points):
            continue
        ax.scatter(points[:, 0], points[:, 1], s=2, color=palette(order % 10), alpha=0.35,
                   zorder=3, label=f"观测{order + 1}（AoA {shrink[order]['aoa_deg']:.1f}°），"
                                   f"{shrink[order]['valid_area_m2']:.0f} m²")
    if len(points):
        ax.scatter(grid[final][:, 0], grid[final][:, 1], s=26, facecolor="none",
                   edgecolor="red", linewidths=1.4, zorder=6,
                   label=f"{len(kept)} 条观测交集，{summary['final_intersection_area_m2']:.0f} m²")
    ax.scatter(*bs, marker="s", s=130, color="black", zorder=7, label="BS")
    ax.scatter(*true_position, marker="*", s=420, color="gold", edgecolor="black", zorder=8,
               label="真实 UE")
    if best.get("position_m"):
        ax.scatter(*best["position_m"], marker="x", s=200, color="magenta", zorder=8,
                   label=f"求解结果（误差 {solver['position_error_m'] * 100:.2f} cm）")
    ax.set_xlim(x_min - 3, x_max + 3)
    ax.set_ylim(y_min - 3, y_max + 3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.15)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{args.sample}：逐观测几何可行域与它们的交集（格点 {args.step:g} m）", fontsize=12)
    figure.tight_layout()
    figure.savefig(output / "explanation.png", dpi=150)
    plt.close(figure)
    print(json.dumps({key: summary[key] for key in (
        "observation_count", "map_area_m2", "final_intersection_area_m2",
        "final_intersection_contains_truth", "nearest_intersection_point_to_truth_m")},
        ensure_ascii=False, indent=1))
    print("[逐观测]", json.dumps([{k: row[k] for k in ("aoa_deg", "valid_area_m2", "intersection_area_m2")}
                                  for row in shrink], ensure_ascii=False))
    print("[求解器]", json.dumps({key: solver[key] for key in (
        "matched_observation_count", "physical_rank", "condition_number",
        "jacobian_singular_values", "sigma_xy_m", "position_error_m")}, ensure_ascii=False))
    print(f"[完成] {output / 'explanation.png'}")


if __name__ == "__main__":
    main()
