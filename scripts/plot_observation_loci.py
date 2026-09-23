"""只读可视化：各观察量的真实可行轨迹及其交集。

对每条观测，在地图上逐格判断“是否存在一条几何合法的候选能复现该观测”，
得到该观测的可用区域（轨迹）。不做定位优化，不修改原实验。

代价控制：只扫描通过 BS 侧必要条件预筛的候选（其余在任何位置都不可能合法）。
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import numpy as np

import scripts.measure_region_vs_box as M
from time_bias_localization.propagation_model import evaluate_hypothesis


def analytic_mask(hypothesis, observation, points, gate, margin, beta_lo, beta_hi):
    matrix = np.asarray(hypothesis.affine_image_matrix, float)
    offset = np.asarray(hypothesis.affine_image_offset, float)
    anchor = np.asarray(hypothesis.anchor_m, float)
    vector = points @ matrix.T + offset - anchor
    length = np.linalg.norm(vector, axis=1) + hypothesis.fixed_length_m
    low = observation.observed_length_m - margin - beta_hi
    high = observation.observed_length_m + margin - beta_lo
    keep = (length >= low) & (length <= high)
    if hypothesis.fixed_aoa_rad is None:
        angle = np.arctan2(vector[:, 1], vector[:, 0])
        delta = np.abs((angle - observation.aoa_rad + math.pi) % (2 * math.pi) - math.pi)
        keep &= delta <= gate
    return keep


def truth_path_rows(source, number, boresight_deg, position, bs, observations, library_patterns):
    """读取真值存档的全部保留路径；标注是否在候选库中存在同类型走法。"""
    document = json.loads((source / "samples" / f"SAMPLE_{number:06d}" / "observation.json").read_text())
    record = document["artifacts"]["truth_npz"]
    rows = []
    with np.load(record["path"], allow_pickle=False) as archive:
        for path_index in np.flatnonzero(archive["retained_mask"]):
            reflections = int(archive["reflection_order"][path_index])
            diffractions = int(archive["diffraction_order"][path_index])
            count = reflections + diffractions
            codes = archive["interactions"][:count, path_index].tolist()
            pattern = tuple("diffraction" if code == 8 else "reflection" for code in codes)
            nodes = np.vstack([np.asarray(position, float)[:2],
                               archive["vertices_m"][:count, path_index, :2],
                               np.asarray(bs, float)])
            global_angle = float(np.rad2deg(archive["aoa_local_rad"][path_index])) + boresight_deg
            nearest = min(range(len(observations)), key=lambda i: abs(
                (global_angle - math.degrees(observations[i].aoa_rad) + 180) % 360 - 180))
            rows.append({
                "label": f"路径{int(path_index)}",
                "path_index": int(path_index),
                "reflections": reflections, "diffractions": diffractions,
                "interaction_codes": codes, "pattern": list(pattern),
                "nodes_m": nodes.tolist(),
                "global_aoa_deg": global_angle,
                "delay_ns": float((archive["absolute_delays_s"][path_index]
                                   + archive["clock_bias_s"]) * 1e9),
                "nearest_observation": observations[nearest].observation_id,
                "diffraction_corners_m": archive["vertices_m"][:count, path_index, :2][
                    np.asarray(codes) == 8].tolist(),
                "route_in_library": pattern in library_patterns or pattern[::-1] in library_patterns,
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=103)
    parser.add_argument("--step", type=float, default=2.0)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    row, folder, scene, hypotheses, observations, memberships, bank_data, search = \
        M.load_bank(source, args.sample)
    report = bank_data["search_report"]
    M._LENGTH_GATE_SIGMA = float(report["length_gate_sigma"])
    gate = float(report["aoa_gate_rad"])
    beta_lo, beta_hi = (float(x) for x in report["beta_interval_m"])
    x_min, x_max, y_min, y_max = scene.bounds_m
    xs = np.arange(x_min + args.step / 2, x_max, args.step)
    ys = np.arange(y_min + args.step / 2, y_max, args.step)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    grid = np.c_[mesh_x.ravel(), mesh_y.ravel()]
    cell_area = args.step ** 2

    walls = {wall.wall_id: wall for wall in scene.walls}
    hidden = M._fully_hidden_walls_from_receiver(walls, np.asarray(hypotheses[0].receiver_m))
    surviving = [index for index, hypothesis in enumerate(hypotheses)
                 if M._receiver_geometry_failure(hypothesis.interactions, walls,
                                                 np.asarray(hypothesis.receiver_m), hidden) is None]

    observation_points = {index: [] for index in range(len(observations))}
    scanned = 0
    for index in surviving:
        hypothesis = hypotheses[index]
        for observation_index in M.memberships_index(memberships, index):
            observation = observations[observation_index]
            margin = observation.length_scale_m * M._LENGTH_GATE_SIGMA
            mask = analytic_mask(hypothesis, observation, grid, gate, margin, beta_lo, beta_hi)
            if not np.any(mask):
                continue
            subset = grid[mask]
            valid = np.asarray([bool(evaluate_hypothesis(scene, hypothesis, point).valid)
                                for point in subset])
            scanned += len(subset)
            if np.any(valid):
                observation_points[observation_index].append(subset[valid])
    print(f"[扫描] 通过预筛候选 {len(surviving)}，几何检查点 {scanned}", flush=True)

    truth = M.localisable_truth(source, row, args.sample)
    position = np.asarray(truth["true_position_m"], float)
    boresight_deg = float(json.loads(
        (source / "experiment.json").read_text())["generation_config"]["radio"]["bs_boresight_deg"])

    library_valid = []
    for index, hypothesis in enumerate(hypotheses):
        if evaluate_hypothesis(scene, hypothesis, position).valid:
            library_valid.append(tuple(kind for kind, _ in hypothesis.interactions))
    truth_rows = truth_path_rows(source, args.sample, boresight_deg, position,
                                 np.asarray(hypotheses[0].receiver_m), observations, library_valid)
    truth_library = {index: [] for index in range(len(observations))}
    for entry in truth_rows:
        if entry["route_in_library"]:
            for observation_index in range(len(observations)):
                if observations[observation_index].observation_id == entry["nearest_observation"]:
                    truth_library[observation_index].append(entry["label"])

    figure, axes = plt.subplots(1, 2, figsize=(21, 10.5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    summary = {"sample_id": row["sample_id"], "grid_step_m": args.step,
               "surviving_candidates_after_bs_prefilter": len(surviving),
               "geometry_checked_points": scanned, "observations": {}}
    point_sets = {}
    for observation_index, observation in enumerate(observations):
        chunks = observation_points[observation_index]
        points = np.vstack(chunks) if chunks else np.empty((0, 2))
        point_sets[observation.observation_id] = points
        summary["observations"][observation.observation_id] = {
            "aoa_deg": math.degrees(observation.aoa_rad),
            "observed_length_m": observation.observed_length_m,
            "allowed_candidates": sum(1 for index, hypothesis in enumerate(hypotheses)
                                      if observation_index in M.memberships_index(memberships, index)),
            "candidates_with_any_valid_point": len(chunks),
            "valid_grid_points": int(len(points)),
            "valid_area_estimate_m2": float(len(points) * cell_area),
            "truth_explained_by_library": bool(truth_library[observation_index]),
            "truth_explaining_candidates": truth_library[observation_index],
        }
    overlap = np.empty((0, 2))
    if len(observation_points[0]) and len(observation_points[1]):
        first = np.vstack(observation_points[0])
        second = np.vstack(observation_points[1])
        try:
            from scipy.spatial import cKDTree
            distance, _ = cKDTree(second).query(first)
            overlap = first[distance <= args.step * 0.75]
        except Exception:
            overlap = np.empty((0, 2))
    summary["intersection"] = {
        "points": int(len(overlap)), "area_estimate_m2": float(len(overlap) * cell_area),
        "contains_true_position": bool(len(overlap) and np.min(
            np.linalg.norm(overlap - position, axis=1)) <= args.step)}
    summary["truth_paths"] = truth_rows
    np.savez(output / "loci_points.npz",
             **{f"obs{index}": point_sets[observation.observation_id]
                for index, observation in enumerate(observations)},
             overlap=overlap, true_position=position)

    all_nodes = np.vstack([entry["nodes_m"] for entry in truth_rows])
    focus = 0.5 * (all_nodes.min(axis=0) + all_nodes.max(axis=0))
    focus_span = float(np.max(all_nodes.max(axis=0) - all_nodes.min(axis=0)))
    for ax, zoom in zip(axes, (False, True)):
        for wall in scene.walls:
            ax.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]],
                    color="0.4", lw=1.0, zorder=2)
        for observation_index, observation in enumerate(observations):
            points = point_sets[observation.observation_id]
            if len(points):
                ax.scatter(points[:, 0], points[:, 1], s=(16 if zoom else 3),
                           color=colors[observation_index % len(colors)],
                           alpha=0.75, zorder=4,
                           label=f"{observation.observation_id}（AoA {math.degrees(observation.aoa_rad):.2f}°，"
                                 f"{len(points)} 格点）")
        if len(overlap):
            ax.scatter(overlap[:, 0], overlap[:, 1], s=(60 if zoom else 18), facecolor="none",
                       edgecolor="red", linewidths=1.3, zorder=6, label="两条观测交集")
        for entry in truth_rows:
            nodes = np.asarray(entry["nodes_m"], float)
            present = entry["route_in_library"]
            style = "-" if present else "--"
            colour = "crimson" if present else "magenta"
            ax.plot(nodes[:, 0], nodes[:, 1], style, color=colour, lw=2.2, zorder=8,
                    label=(f"{entry['label']}：{entry['reflections']}次反射"
                           f"{'+1次绕射' if entry['diffractions'] else ''}，"
                           f"库中{'有' if present else '缺失'} → 解释 "
                           f"{entry['nearest_observation'][-2:]}"))
            ax.scatter(nodes[1:-1, 0], nodes[1:-1, 1], s=70, facecolor="white",
                       edgecolor=colour, zorder=9)
            corners = np.asarray(entry["diffraction_corners_m"], float)
            if corners.size:
                ax.scatter(corners[:, 0], corners[:, 1], marker="D", s=140,
                           facecolor="magenta" if not present else "crimson",
                           edgecolor="black", zorder=10,
                           label=f"{entry['label']} 绕射拐角")
        ax.scatter(*np.asarray(hypotheses[0].receiver_m), marker="s", s=120, color="black",
                   zorder=7, label="BS")
        ax.scatter(*position, marker="*", s=420, color="gold", edgecolor="black", zorder=11,
                   label="真实 UE")
        if zoom:
            half = 0.5 * focus_span + 20.0
            ax.set_xlim(focus[0] - half, focus[0] + half)
            ax.set_ylim(focus[1] - half, focus[1] + half)
            ax.set_title("放大：覆盖全部真实路径")
        else:
            ax.set_xlim(x_min - 3, x_max + 3)
            ax.set_ylim(y_min - 3, y_max + 3)
            ax.set_title("全图")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.legend(loc="best", fontsize=8)
    missing = [entry["label"] for entry in truth_rows if not entry["route_in_library"]]
    figure.suptitle(
        f"{row['sample_id']}：各观测的几何可用轨迹（步长 {args.step:g} m，"
        f"扫描 BS 侧预筛的 {len(surviving)} 个候选）\n"
        f"交集 {summary['intersection']['points']} 个格点 ≈ "
        f"{summary['intersection']['area_estimate_m2']:.0f} m²，"
        f"含真值：{summary['intersection']['contains_true_position']}；"
        f"库中缺失的真实路径：{missing or '无'}", fontsize=13)
    figure.tight_layout()
    figure.savefig(output / "observation_loci.png", dpi=150)
    plt.close(figure)
    (output / "observation_loci.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[完成] {output / 'observation_loci.png'}")


if __name__ == "__main__":
    main()
