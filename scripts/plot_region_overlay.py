"""只读可视化：真值路线的解析可行区域叠加地图，并标出真正可走通的子区域。

区域 = 角度锥（半角 aoa_gate）∩ 径向环带（β 区间 × 长度容差）∩ 地图。
“可走通”点由 evaluate_hypothesis 判定（有限墙段 + 可见性 + 绕射阴影）。
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
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np

import scripts.measure_region_vs_box as M
from time_bias_localization.propagation_model import evaluate_hypothesis


def closed_path(polygons):
    vertices, codes = [], []
    for polygon in polygons:
        if polygon is None or len(polygon) < 3:
            continue
        vertices.extend(polygon)
        codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(polygon) - 2)
                      + [MplPath.CLOSEPOLY])
    return MplPath(np.asarray(vertices, float), np.asarray(codes, np.uint8))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=103)
    parser.add_argument("--grid", type=int, default=200)
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
    rect = np.asarray([[x_min, y_min], [x_max, y_min],
                       [x_max, y_max], [x_min, y_max]], float)

    truth = M.localisable_truth(source, row, args.sample)
    position = np.asarray(truth["true_position_m"], float)
    valid_indices = [index for index, hypothesis in enumerate(hypotheses)
                     if evaluate_hypothesis(scene, hypothesis, position).valid]
    assert len(valid_indices) == 1, f"真值处合法候选应为 1 条，实际 {len(valid_indices)}"
    index = valid_indices[0]
    hypothesis = hypotheses[index]
    observation_index = M.memberships_index(memberships, index)[0]
    observation = observations[observation_index]
    geometry = M.region_geometry(hypothesis, observation, gate, beta_lo, beta_hi)
    area, clipped = M.region_area(geometry, rect)
    kind, first, second = M.region_polygon(geometry)
    center, inner, outer, lower, upper, full = geometry

    # 区域内网格：区分“解析可行”与“几何可走通”
    lo = np.maximum(clipped.min(axis=0), [x_min, y_min])
    hi = np.minimum(clipped.max(axis=0), [x_max, y_max])
    xs = np.linspace(lo[0], hi[0], args.grid)
    ys = np.linspace(lo[1], hi[1], args.grid)
    mesh_x, mesh_y = np.meshgrid(xs, ys)
    candidates = np.c_[mesh_x.ravel(), mesh_y.ravel()]
    inside = M.region_contains(geometry, candidates)
    candidates = candidates[inside]
    valid_flags = np.asarray([bool(evaluate_hypothesis(scene, hypothesis, point).valid)
                              for point in candidates])
    valid_points = candidates[valid_flags]
    usable_area = float(area * len(valid_points) / max(len(candidates), 1))

    evaluation = evaluate_hypothesis(scene, hypothesis, position)
    nodes = np.asarray(evaluation.path.nodes, float) if evaluation.path is not None else None

    figure, axes = plt.subplots(1, 2, figsize=(19, 8.5))
    for ax, zoom in zip(axes, (False, True)):
        for wall in scene.walls:
            ax.plot([wall.start[0], wall.end[0]], [wall.start[1], wall.end[1]],
                    color="0.35", lw=1.0, zorder=2)
        ax.add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               fill=False, ec="0.6", ls="--", lw=1.0, zorder=1))
        patch = PathPatch(closed_path([first, second]), facecolor="tab:blue", alpha=0.28,
                          edgecolor="tab:blue", lw=1.4, zorder=3)
        ax.add_patch(patch)
        if len(valid_points):
            ax.scatter(valid_points[:, 0], valid_points[:, 1], s=6, color="tab:red",
                       zorder=5, label=f"几何可走通 {len(valid_points)} 点")
        ax.scatter(*np.asarray(hypothesis.receiver_m), marker="s", s=90, color="black",
                   zorder=6, label="BS")
        ax.scatter(*center, marker="x", s=90, color="darkgreen", zorder=6,
                   label="等效圆心 c（镜像点）")
        ax.scatter(*position, marker="*", s=320, color="gold", edgecolor="black",
                   zorder=7, label="真实 UE")
        if nodes is not None:
            ax.plot(nodes[:, 0], nodes[:, 1], "-o", color="tab:orange", lw=2.0, ms=5,
                    zorder=6, label="真实路径")
        if zoom:
            pad = 8.0
            ax.set_xlim(lo[0] - pad, hi[0] + pad)
            ax.set_ylim(lo[1] - pad, hi[1] + pad)
            ax.set_title(f"放大：区域 bbox + {pad:g} m")
        else:
            ax.set_xlim(x_min - 4, x_max + 4)
            ax.set_ylim(y_min - 4, y_max + 4)
            ax.set_title("全图")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.legend(loc="best", fontsize=8)
    figure.suptitle(
        f"{row['sample_id']} 真值候选 {hypothesis.hypothesis_id}"
        f"（{hypothesis.reflection_order} 次反射）解析区域 vs 几何可走通子区域", fontsize=12)
    figure.tight_layout()
    figure.savefig(output / "region_overlay.png", dpi=150)
    plt.close(figure)

    summary = {
        "sample_id": row["sample_id"],
        "hypothesis_id": hypothesis.hypothesis_id,
        "interactions": [list(item) for item in hypothesis.interactions],
        "fixed_aoa_rad": hypothesis.fixed_aoa_rad,
        "region_kind": "annulus_full_ring" if full else "annular_sector",
        "gate_deg": math.degrees(gate),
        "beta_interval_m": [beta_lo, beta_hi],
        "equivalent_center_m": center.tolist(),
        "inner_radius_m": float(inner), "outer_radius_m": float(outer),
        "azimuth_deg": [math.degrees(lower), math.degrees(upper)],
        "analytic_region_area_m2": float(area),
        "box_area_m2": (x_max - x_min) * (y_max - y_min),
        "grid_points_in_region": int(len(candidates)),
        "grid_points_geometry_valid": int(len(valid_points)),
        "geometry_valid_fraction": float(len(valid_points) / max(len(candidates), 1)),
        "usable_area_estimate_m2": usable_area,
        "true_position_m": position.tolist(),
        "true_position_inside_region": bool(M.region_contains(geometry, position[None, :])[0]),
        "true_path_nodes_m": None if nodes is None else nodes.tolist(),
        "scope": "analytic_region_and_sampled_geometry_only; read_only_no_localization_rerun",
    }
    (output / "region_overlay.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in (
        "region_kind", "analytic_region_area_m2", "box_area_m2",
        "grid_points_in_region", "grid_points_geometry_valid", "geometry_valid_fraction",
        "usable_area_estimate_m2", "true_position_inside_region")}, ensure_ascii=False, indent=2))
    print(f"[完成] {output / 'region_overlay.png'}")


if __name__ == "__main__":
    main()
