"""只读可视化：某条观测的可用轨迹由哪几条候选贡献，以及它们用到的墙。

回答“蓝点从哪来”：逐条候选画合法子区域，并用同色高亮该候选使用的墙段。
不重跑定位，不修改原实验。
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

import scripts.plot_observation_loci as L
import scripts.measure_region_vs_box as M
from time_bias_localization.propagation_model import evaluate_hypothesis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=103)
    parser.add_argument("--observation", type=int, default=0)
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
    walls = {wall.wall_id: wall for wall in scene.walls}
    hidden = M._fully_hidden_walls_from_receiver(walls, np.asarray(hypotheses[0].receiver_m))
    observation = observations[args.observation]

    contributions = []
    for index, hypothesis in enumerate(hypotheses):
        if args.observation not in M.memberships_index(memberships, index):
            continue
        if M._receiver_geometry_failure(hypothesis.interactions, walls,
                                        np.asarray(hypothesis.receiver_m), hidden) is not None:
            continue
        mask = L.analytic_mask(hypothesis, observation, grid, gate,
                               observation.length_scale_m * M._LENGTH_GATE_SIGMA,
                               beta_lo, beta_hi)
        if not np.any(mask):
            continue
        subset = grid[mask]
        valid = np.asarray([bool(evaluate_hypothesis(scene, hypothesis, point).valid)
                            for point in subset])
        if not np.any(valid):
            continue
        contributions.append({"hypothesis": hypothesis, "points": subset[valid]})
    contributions.sort(key=lambda item: -len(item["points"]))

    truth = M.localisable_truth(source, row, args.sample)
    position = np.asarray(truth["true_position_m"], float)
    boresight = float(json.loads((source / "experiment.json").read_text()
                                 )["generation_config"]["radio"]["bs_boresight_deg"])
    patterns = [tuple(kind for kind, _ in h.interactions) for h in hypotheses
                if evaluate_hypothesis(scene, h, position).valid]
    truth_rows = L.truth_path_rows(source, args.sample, boresight, position,
                                   np.asarray(hypotheses[0].receiver_m),
                                   observations, patterns)

    palette = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:brown"]
    figure, axes = plt.subplots(1, 2, figsize=(22, 11))
    summary = {"sample_id": row["sample_id"],
               "observation_id": observation.observation_id,
               "aoa_deg": math.degrees(observation.aoa_rad),
               "observed_length_m": observation.observed_length_m,
               "true_position_m": position.tolist(),
               "contributions": []}
    for ax, zoom in zip(axes, (False, True)):
        for wall in scene.walls:
            ax.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]],
                    color="0.45", lw=1.0, zorder=2)
        for order, entry in enumerate(contributions):
            hypothesis, points = entry["hypothesis"], entry["points"]
            colour = palette[order % len(palette)]
            distance = float(np.min(np.linalg.norm(points - position, axis=1)))
            ax.scatter(points[:, 0], points[:, 1], s=(18 if zoom else 3), color=colour,
                       alpha=0.75, zorder=6,
                       label=f"{hypothesis.hypothesis_id[:10]}… 2次反射，{len(points)} 点，"
                             f"距真值 {distance:.1f} m")
            for _, wall_id in hypothesis.interactions:
                wall = walls[wall_id]
                ax.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]],
                        color=colour, lw=3.0 if zoom else 2.0, alpha=0.9, zorder=5,
                        label=f"  墙 {wall_id}")
            summary["contributions"].append({
                "hypothesis_id": hypothesis.hypothesis_id,
                "reflections": hypothesis.reflection_order,
                "diffractions": hypothesis.diffraction_order,
                "walls": [key for _, key in hypothesis.interactions],
                "walls_geometry_m": {key: [list(walls[key].start), list(walls[key].end)]
                                     for _, key in hypothesis.interactions},
                "valid_grid_points": int(len(points)),
                "distance_to_true_ue_m": distance,
                "point_bbox_m": np.round(np.vstack([points.min(axis=0),
                                                    points.max(axis=0)]), 2).tolist(),
            })
        for entry in truth_rows:
            nodes = np.asarray(entry["nodes_m"], float)
            ax.plot(nodes[:, 0], nodes[:, 1], "--" if not entry["route_in_library"] else "-",
                    color="crimson", lw=2.2, zorder=8,
                    label=f"{entry['label']}：{entry['reflections']}反"
                          f"{'+1绕' if entry['diffractions'] else ''}，"
                          f"库中{'有' if entry['route_in_library'] else '缺失'}")
            ax.scatter(nodes[1:-1, 0], nodes[1:-1, 1], s=60, facecolor="white",
                       edgecolor="crimson", zorder=9)
            corners = np.asarray(entry["diffraction_corners_m"], float)
            if corners.size:
                ax.scatter(corners[:, 0], corners[:, 1], marker="D", s=150,
                           facecolor="magenta", edgecolor="black", zorder=10, label="绕射拐角")
        ax.scatter(*np.asarray(hypotheses[0].receiver_m), marker="s", s=130, color="black",
                   zorder=11, label="BS")
        ax.scatter(*position, marker="*", s=420, color="gold", edgecolor="black", zorder=12,
                   label="真实 UE")
        if zoom:
            ax.set_xlim(-45, 60)
            ax.set_ylim(10, 175)
            ax.set_title("放大")
        else:
            ax.set_xlim(x_min - 3, x_max + 3)
            ax.set_ylim(y_min - 3, y_max + 3)
            ax.set_title("全图")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.legend(loc="upper right", fontsize=7)
    figure.suptitle(f"{row['sample_id']} {observation.observation_id} 可用轨迹的来源："
                    f"{len(contributions)} 条候选，全部为二次反射", fontsize=13)
    figure.tight_layout()
    figure.savefig(output / "locus_sources.png", dpi=150)
    plt.close(figure)
    (output / "locus_sources.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[完成] {output / 'locus_sources.png'}")


if __name__ == "__main__":
    main()
