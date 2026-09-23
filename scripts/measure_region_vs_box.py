"""只读量级检查：单条观测的精确可行区域 vs 现有包围盒过近似。

对指定样本的已保存候选库，逐条候选计算 L(x)=|x-c|+fixed_length 与
θ(x) 在观测容差内的精确可行区域（锥∩环带∩地图，不含遮挡与有限墙段），
与现有 ``_compatible_observations`` 隐含的“整个包围盒都可行”比较面积与数量。
不重跑定位，不修改原实验；真值只在最后的健全性检查中读取。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath

from time_bias_localization.propagation_model import (
    ContinuousObservation, PropagationHypothesis, evaluate_hypothesis,
)
from time_bias_localization.propagation_hypotheses import (
    _fully_hidden_walls_from_receiver, _receiver_geometry_failure,
)
from time_bias_localization.scene import Scene2D


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def wrap_angle(angle):
    return float((angle + math.pi) % (2 * math.pi) - math.pi)


def polygon_area(poly):
    if poly is None or len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)))


def clip_convex(poly, rect):
    """Sutherland–Hodgman：凸多边形 rect（逆时针）裁剪 poly。"""
    out = np.asarray(poly, float)
    for index in range(len(rect)):
        if len(out) == 0:
            return np.empty((0, 2))
        a = rect[index]
        b = rect[(index + 1) % len(rect)]
        edge = b - a
        source, out = out, []
        for position in range(len(source)):
            p = source[position]
            q = source[(position + 1) % len(source)]
            dp = edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])
            dq = edge[0] * (q[1] - a[1]) - edge[1] * (q[0] - a[0])
            inside_p, inside_q = dp >= -1e-12, dq >= -1e-12
            if inside_p:
                out.append(p)
            if inside_p != inside_q:
                out.append(p + (dp / (dp - dq)) * (q - p))
        out = np.asarray(out, float) if out else np.empty((0, 2))
    return out


def circle_polygon(center, radius, steps):
    angle = np.linspace(0.0, 2 * math.pi, steps, endpoint=False)
    return center + radius * np.c_[np.cos(angle), np.sin(angle)]


def region_geometry(hypothesis, observation, gate_rad, beta_lo, beta_hi, steps=1440):
    """返回 (等效圆心, 内半径, 外半径, 方位下限, 方位上限, 是否整圈)。"""
    matrix = np.asarray(hypothesis.affine_image_matrix, float)
    offset = np.asarray(hypothesis.affine_image_offset, float)
    anchor = np.asarray(hypothesis.anchor_m, float)
    center = matrix.T @ (anchor - offset)
    margin = observation.length_scale_m * _LENGTH_GATE_SIGMA
    fixed = hypothesis.fixed_length_m
    outer = observation.observed_length_m + margin - beta_lo - fixed
    inner = max(observation.observed_length_m - margin - beta_hi - fixed, 0.0)
    if outer <= 0 or outer < inner:
        return None
    if hypothesis.fixed_aoa_rad is None:
        direction = matrix.T @ np.array([math.cos(observation.aoa_rad),
                                         math.sin(observation.aoa_rad)])
        base = math.atan2(direction[1], direction[0])
        return center, inner, outer, base - gate_rad, base + gate_rad, False
    return center, inner, outer, -math.pi, math.pi, True


_LENGTH_GATE_SIGMA = 5.0


def region_polygon(geometry, steps=1440):
    center, inner, outer, lower, upper, full = geometry
    if full:
        return ("annulus", circle_polygon(center, outer, steps),
                circle_polygon(center, inner, steps) if inner > 1e-12 else None)
    angle = np.linspace(lower, upper, steps)
    ring = center + outer * np.c_[np.cos(angle), np.sin(angle)]
    if inner <= 1e-12:
        return ("sector", np.vstack([ring, center]), None)
    back = center + inner * np.c_[np.cos(angle[::-1]), np.sin(angle[::-1])]
    return ("sector", np.vstack([ring, back]), None)


def region_area(geometry, rect):
    if geometry is None:
        return 0.0, None
    kind, first, second = region_polygon(geometry)
    clipped = clip_convex(first, rect)
    area = polygon_area(clipped)
    if kind == "annulus" and second is not None:
        area -= polygon_area(clip_convex(second, rect))
    return max(area, 0.0), clipped


def region_contains(geometry, points):
    """多边形内点判据；整圈时扣除内孔。"""
    kind, first, second = region_polygon(geometry)
    inside = MplPath(first, closed=True).contains_points(points)
    if second is not None:
        inside &= ~MplPath(second, closed=True).contains_points(points)
    return inside


def mask_points(geometry, points):
    """(θ, L) 直接判据；用于与多边形面积交叉核对。"""
    center, inner, outer, lower, upper, full = geometry
    relative = points - center
    radius = np.linalg.norm(relative, axis=1)
    radial = (radius >= inner) & (radius <= outer) & (radius > 1e-12)
    if full:
        return radial
    middle = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    angle = np.arctan2(relative[:, 1], relative[:, 0])
    delta = np.abs((angle - middle + math.pi) % (2 * math.pi) - math.pi)
    return radial & (delta <= half)


def load_bank(source, number):
    row_path = source / "samples" / f"SAMPLE_{number:06d}" / "result.json"
    row = json.loads(row_path.read_text())
    folder, = (Path(row["attempt_dir"]) / "localization_unavailable").glob("*")
    bank_data = json.loads((folder / "propagation_hypotheses.json").read_text())
    observed = json.loads((folder / "continuous_observations.json").read_text())
    search = json.loads((folder / "continuous_search.json").read_text())
    scene_path, = (source / "channel_setups").glob("*/scene/scene_2d.json")
    scene = Scene2D.from_dict(json.loads(scene_path.read_text()))
    hypotheses = []
    for value in bank_data["hypotheses"]:
        item = dict(value)
        item["interactions"] = tuple(tuple(x) for x in item["interactions"])
        hypotheses.append(PropagationHypothesis(**item))
    observations = tuple(ContinuousObservation(**x) for x in observed["observations"])
    memberships = tuple(tuple(x) for x in bank_data["observation_hypothesis_indices"])
    return row, folder, scene, hypotheses, observations, memberships, bank_data, search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=103)
    parser.add_argument("--grid", type=int, default=1200)
    parser.add_argument("--occlusion-tests", type=int, default=96)
    parser.add_argument("--occlusion-samples", type=int, default=1500)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("输出必须与只读输入分开")
    output.mkdir(parents=True, exist_ok=False)

    global _LENGTH_GATE_SIGMA
    row, folder, scene, hypotheses, observations, memberships, bank_data, search = \
        load_bank(source, args.sample)
    report = bank_data["search_report"]
    _LENGTH_GATE_SIGMA = float(report["length_gate_sigma"])
    gate = float(report["aoa_gate_rad"])
    beta_lo, beta_hi = (float(x) for x in report["beta_interval_m"])
    x_min, x_max, y_min, y_max = scene.bounds_m
    rect = np.asarray([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], float)
    box_area = (x_max - x_min) * (y_max - y_min)

    walls = {wall.wall_id: wall for wall in scene.walls}
    hidden = _fully_hidden_walls_from_receiver(walls, np.asarray(hypotheses[0].receiver_m)) \
        if hypotheses else {}

    entries = []
    regions = {}
    for index, hypothesis in enumerate(hypotheses):
        best = {"hypothesis_id": hypothesis.hypothesis_id,
                "reflections": hypothesis.reflection_order,
                "diffractions": hypothesis.diffraction_order,
                "interactions": [list(item) for item in hypothesis.interactions],
                "area_m2": 0.0, "matched_observation_indices": list(memberships_index(memberships, index))}
        record = None
        for observation_index in memberships_index(memberships, index):
            observation = observations[observation_index]
            geometry = region_geometry(hypothesis, observation, gate, beta_lo, beta_hi)
            area, clipped = region_area(geometry, rect)
            if record is None or area > record[1]:
                record = (geometry, area, clipped, observation_index)
        if record is not None:
            best["area_m2"] = record[1]
            regions[index] = record
        best["region_empty_reason"] = (None if record is not None and record[1] > 1e-9
                                       else "outside_map_or_no_feasible_beta")
        best["region_nonempty"] = best["area_m2"] > 1e-9
        best["receiver_geometry_failure"] = _receiver_geometry_failure(
            hypothesis.interactions, walls, np.asarray(hypothesis.receiver_m), hidden)
        entries.append(best)

    populated = [item for item in entries if item["region_nonempty"]]
    geometry_ok = [item for item in entries if item["receiver_geometry_failure"] is None]
    both = [item for item in populated if item["receiver_geometry_failure"] is None]
    areas = np.asarray([item["area_m2"] for item in populated]) if populated else np.zeros(0)

    # 交叉核对：随机点直接判据 vs 多边形面积
    rng = np.random.default_rng(0)
    check_rows = []
    for item in populated[:20]:
        index = next(i for i, h in enumerate(hypotheses) if h.hypothesis_id == item["hypothesis_id"])
        geometry, _, clipped, _ = regions[index]
        if clipped is None or len(clipped) < 3:
            continue
        lo, hi = clipped.min(axis=0), clipped.max(axis=0)
        samples = rng.uniform(lo, hi, size=(4000, 2))
        inside = region_contains(geometry, samples)
        direct = mask_points(geometry, samples)
        check_rows.append({"hypothesis_id": item["hypothesis_id"],
                           "samples": int(len(samples)),
                           "disagreements": int(np.sum(inside != direct)),
                           "polygon_points": int(inside.sum()), "direct_points": int(direct.sum())})

    # 遮挡影响量级：无偏随机抽样（含真值处合法候选），区域内随机取样检查真实几何
    truth = localisable_truth(source, row, args.sample)
    pool = sorted(i for i in regions if regions[i][1] > 1e-9)
    truth_indices = []
    if truth:
        true_position = np.asarray(truth["true_position_m"], float)
        truth_indices = [i for i, h in enumerate(hypotheses)
                         if evaluate_hypothesis(scene, h, true_position).valid]
    drawn = rng.choice(pool, size=min(args.occlusion_tests, len(pool)), replace=False).tolist()
    order = sorted(set(truth_indices) | set(int(i) for i in drawn))
    occlusion = []
    for index in order:
        geometry, area, clipped, observation_index = regions[index]
        if clipped is None or len(clipped) < 3:
            continue
        lo, hi = clipped.min(axis=0), clipped.max(axis=0)
        samples = rng.uniform(lo, hi, size=(args.occlusion_samples, 2))
        samples = samples[region_contains(geometry, samples)]
        if not len(samples):
            continue
        valid = np.asarray([bool(evaluate_hypothesis(scene, hypotheses[index], point).valid)
                            for point in samples])
        occlusion.append({"hypothesis_id": hypotheses[index].hypothesis_id,
                          "region_area_m2": float(area),
                          "points_inside_region": int(len(samples)),
                          "points_geometry_valid": int(valid.sum()),
                          "valid_fraction": float(valid.mean()),
                          "occlusion_valid_area_m2": float(area * valid.mean())})

    truth_check = truth_region_check(scene, hypotheses, regions, observations, truth) if truth else None
    occlusion_valid_areas = np.asarray([row_["occlusion_valid_area_m2"] for row_ in occlusion])
    summary = {
        "sample_id": row["sample_id"], "status": row["status"],
        "gate_rad": gate, "gate_deg": math.degrees(gate),
        "beta_interval_m": [beta_lo, beta_hi],
        "length_margin_rule": f"{_LENGTH_GATE_SIGMA:g} * observation.length_scale_m",
        "scene_bounds_m": [x_min, x_max, y_min, y_max],
        "box_area_m2": box_area,
        "candidate_count_saved": len(entries),
        "region_nonempty_count": len(populated),
        "region_empty_count": len(entries) - len(populated),
        "receiver_geometry_pass_count": len(geometry_ok),
        "region_nonempty_and_geometry_pass": len(both),
        "region_area_m2": {
            "count": int(areas.size), "min": float(areas.min()) if areas.size else 0.0,
            "median": float(np.median(areas)) if areas.size else 0.0,
            "mean": float(areas.mean()) if areas.size else 0.0,
            "max": float(areas.max()) if areas.size else 0.0,
            "sum": float(areas.sum()) if areas.size else 0.0,
        },
        "box_over_region_factor": {
            "median": float(box_area / np.median(areas)) if areas.size else None,
            "max": float(box_area / areas.min()) if areas.size else None,
        },
        "region_share_of_box": {
            "median": float(np.median(areas) / box_area) if areas.size else None,
            "max": float(areas.max() / box_area) if areas.size else None,
        },
        "occlusion_scan": {
            "hypotheses_tested": len(occlusion),
            "median_valid_fraction": float(np.median([r["valid_fraction"] for r in occlusion])) if occlusion else None,
            "zero_valid_hypotheses": int(sum(r["points_geometry_valid"] == 0 for r in occlusion)),
            "median_occlusion_valid_area_m2": float(np.median(occlusion_valid_areas)) if occlusion else None,
            "box_over_occlusion_valid_factor": (
                float(box_area / np.median(occlusion_valid_areas))
                if occlusion and np.median(occlusion_valid_areas) > 0 else None),
        },
        "population_by_family": population_by_family(entries, populated),
        "polygon_vs_direct_checks": check_rows,
        "occlusion_samples": occlusion,
        "truth_sanity": truth, "truth_region_check": truth_check,
        "scope": "analytic_cone_annulus_intersect_map_only; occlusion_and_finite_segment_shrink_further; "
                 "box_area_is_the_implied_ambiguity_of_the_current_admission_test",
        "uses_truth": truth is not None,
    }
    (output / "region_vs_box.json").write_text(
        json.dumps({"summary": summary, "candidates": entries},
                   ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_markdown(output / "region_vs_box.md", summary)
    (output / "provenance.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(source), "sample_result_sha256": digest(
            source / "samples" / f"SAMPLE_{args.sample:06d}" / "result.json"),
        "hypotheses_sha256": digest(folder / "propagation_hypotheses.json"),
        "observations_sha256": digest(folder / "continuous_observations.json"),
        "code_changed": False, "full_localization_rerun": False,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"[完成] {output / 'region_vs_box.md'}", flush=True)


def memberships_index(memberships, hypothesis_index):
    return [i for i, indices in enumerate(memberships) if hypothesis_index in indices]


def population_by_family(entries, populated):
    def bucket(items):
        result = {}
        for item in items:
            key = f"reflection_{item['reflections']}_diffraction_{item['diffractions']}"
            result[key] = result.get(key, 0) + 1
        return dict(sorted(result.items()))
    return {"saved": bucket(entries), "region_nonempty": bucket(populated)}


def truth_region_check(scene, hypotheses, regions, observations, truth):
    """离线健全性：真值处几何合法的候选，其解析区域必须包含真值点。"""
    position = np.asarray(truth["true_position_m"], float)
    rows = []
    for index, hypothesis in enumerate(hypotheses):
        if not evaluate_hypothesis(scene, hypothesis, position).valid:
            continue
        if index not in regions:
            continue
        geometry, area, _, observation_index = regions[index]
        observation = observations[observation_index]
        center, inner, outer, _, _, _ = geometry
        radius = float(np.linalg.norm(position - center))
        length = radius + hypothesis.fixed_length_m
        beta = observation.observed_length_m - length
        rows.append({"hypothesis_id": hypothesis.hypothesis_id,
                     "interactions": [list(item) for item in hypothesis.interactions],
                     "region_area_m2": float(area),
                     "true_point_inside_region": bool(region_contains(geometry, position[None, :])[0]),
                     "implied_beta_m": beta,
                     "radius_in_band": bool(inner - 1e-9 <= radius <= outer + 1e-9)})
    return {"true_position_m": position.tolist(),
            "geometry_valid_at_true_position": len(rows), "rows": rows,
            "all_true_valid_points_inside_their_region": bool(rows) and all(
                r["true_point_inside_region"] for r in rows)}


def localisable_truth(source, row, number):
    """离线健全性检查：真值处的合法候选是否都通过解析区域判据。"""
    observation_path = source / "samples" / f"SAMPLE_{number:06d}" / "observation.json"
    if not observation_path.exists():
        return None
    observation = json.loads(observation_path.read_text())
    record = observation.get("artifacts", {}).get("truth_npz")
    if not record:
        return None
    with np.load(record["path"], allow_pickle=False) as archive:
        position = archive["ue_position_m"].tolist()
    return {"true_position_m": position, "read_only_for_sanity": True,
            "true_position_from": record["path"]}


def write_markdown(path, summary):
    areas = summary["region_area_m2"]
    factor = summary["box_over_region_factor"]
    scan = summary["occlusion_scan"]
    lines = [
        "# 精确可行区域 vs 包围盒过近似", "",
        f"样本 {summary['sample_id']}（状态 {summary['status']}）。只读已保存候选库与观测，不重跑定位。", "",
        "候选函数满足 `L(x)=|x-c|+fixed_length`、无绕射时 `θ(x)=angle(A(x-c))`。",
        "单条观测的可行区域 = 角度锥（半角为 aoa_gate）∩ 径向环带（β 区间 × 长度容差）∩ 地图，",
        "不含遮挡与有限墙段约束；这些只会让区域更小。", "",
        "## 量级", "",
        f"- 地图包围盒面积：{summary['box_area_m2']:.1f} m²",
        f"- 已保存候选：{summary['candidate_count_saved']}",
        f"- 解析区域非空：{summary['region_nonempty_count']}",
        f"- 解析区域为空（纯属过近似放行）：{summary['region_empty_count']}",
        f"- 另通过 BS 侧几何预筛：{summary['receiver_geometry_pass_count']}",
        f"- 两者同时通过：{summary['region_nonempty_and_geometry_pass']}", "",
        "## 精确区域面积", "",
        f"- 数量 {areas['count']}，最小 {areas['min']:.3f} m²，中位 {areas['median']:.3f} m²，"
        f"均值 {areas['mean']:.3f} m²，最大 {areas['max']:.3f} m²",
        f"- 中位区域只占包围盒的 {summary['region_share_of_box']['median']:.3e}",
        f"- 包围盒是中位区域的 {factor['median']:.3e} 倍" if factor["median"] else "",
        "", "## 按传播类型", "",
        f"- 保存：{summary['population_by_family']['saved']}",
        f"- 区域非空：{summary['population_by_family']['region_nonempty']}", "",
        "## 遮挡扫描（解析区域非空候选随机抽样，含真值处合法候选）", "",
        f"- 受检候选 {scan['hypotheses_tested']}，其中区域内几何完全不可行的 {scan['zero_valid_hypotheses']}",
        f"- 合法比例中位数 {scan['median_valid_fraction']}",
        f"- 扣除遮挡后的有效面积中位数 {scan['median_occlusion_valid_area_m2']} m²",
        f"- 包围盒是遮挡后有效面积的 {scan['box_over_occlusion_valid_factor']} 倍", "",
        "## 真值健全性（离线只读，不参与选择）", ""]
    truth = summary.get("truth_region_check")
    if truth:
        lines.append(f"- 真值处几何合法的候选 {truth['geometry_valid_at_true_position']} 条；"
                     f"全部落在各自解析区域内：{truth['all_true_valid_points_inside_their_region']}")
        for row in truth["rows"]:
            lines.append(f"  - {row['hypothesis_id']} {row['interactions']} "
                         f"区域 {row['region_area_m2']:.2f} m²，隐含 β={row['implied_beta_m']:.3f} m，"
                         f"真值点在区域内={row['true_point_inside_region']}")
    else:
        lines.append("- 未读取真值。")
    lines.extend(["", "## 交叉核对（多边形 vs 直接判据）", ""])
    for row in summary["polygon_vs_direct_checks"]:
        lines.append(f"- {row['hypothesis_id']}：{row['samples']} 点，分歧 {row['disagreements']} "
                     f"（多边形 {row['polygon_points']}，判据 {row['direct_points']}）")
    lines.extend(["", "## 遮挡明细（前 10 条）", "",
        "| 候选 | 区域内取样点 | 几何合法点 | 合法比例 |", "| --- | ---: | ---: | ---: |"])
    for row in summary["occlusion_samples"][:10]:
        lines.append(f"| {row['hypothesis_id']} | {row['points_inside_region']} | "
                     f"{row['points_geometry_valid']} | {row['valid_fraction']:.4f} |")
    lines.extend(["", "## 边界", "", summary["scope"], ""])
    path.write_text("\n".join(line for line in lines if line is not None) + "\n")


if __name__ == "__main__":
    main()
