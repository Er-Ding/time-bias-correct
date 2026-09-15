"""点簇压缩：先覆盖参考位置，再补足整个合法偏差区间。

只对聚类后的成员做覆盖计算，不提前建立求解器轨迹。两成员间的位置差
关于 beta 是一次式；距离阈值内的区间由二次不等式解析求出，再取区间并集。
因此不会把有限几个偏差检查点误称为整个区间的覆盖保证。
"""
from __future__ import annotations

import numpy as np
from .constants import SPEED_OF_LIGHT_M_S


def select_cover_members(members, first_index, radius_m, beta_interval_m, max_representatives=None):
    if max_representatives is not None and (isinstance(max_representatives, (bool, np.bool_))
            or not isinstance(max_representatives, (int, np.integer)) or max_representatives < 1):
        raise ValueError("绕射簇代表上限必须为正整数或 None")
    limit = len(members) if max_representatives is None else max_representatives
    if isinstance(radius_m, (bool, np.bool_)) or not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("绕射代表点覆盖距离必须为有限正数")
    bounds = np.asarray(beta_interval_m, float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError("覆盖检查需要有限、递增的公共偏差区间")
    reference = members[0].reference_bias_s * SPEED_OF_LIGHT_M_S
    if not bounds[0] <= reference <= bounds[1]:
        raise ValueError("参考偏差必须位于覆盖检查区间内")
    positions = np.asarray([p.position_m for p in members])
    directions = np.asarray([p.endpoint_direction for p in members])
    remaining = np.asarray([np.dot(np.asarray(p.position_m) - p.endpoint_origin_m, p.endpoint_direction)
                            for p in members])
    # 绕射点、墙面与地图边界上的零长/退化末段不属于合法路径。
    upper = np.minimum(bounds[1] - reference, remaining - 1e-7)
    lower = np.maximum(bounds[0] - reference, remaining - [p.endpoint_free_distance_m for p in members] + 1e-7)
    if np.any(lower >= upper) or np.any(remaining <= 0):
        raise ValueError("成员必须有非空合法末段和偏差区间")
    uncovered = [[(float(lo), float(hi))] for lo, hi in zip(lower, upper)]
    selected = []
    nearest = np.full(len(members), np.inf)
    interval_tolerance = 1e-9

    def add(index):
        selected.append(int(index))
        np.minimum(nearest, np.linalg.norm(positions - positions[index], axis=1), out=nearest)
        delta_p = positions - positions[index]
        delta_d = directions - directions[index]
        a = np.sum(delta_d ** 2, axis=1)
        b = -2 * np.sum(delta_p * delta_d, axis=1)
        c = np.sum(delta_p ** 2, axis=1) - radius_m ** 2
        for i in range(len(members)):
            if not uncovered[i]:
                continue
            lo, hi = max(lower[i], lower[index]), min(upper[i], upper[index])
            if lo > hi:
                continue
            if a[i] <= 1e-24:
                if c[i] > 1e-12:
                    continue
            else:
                discriminant = b[i] ** 2 - 4 * a[i] * c[i]
                if discriminant < 0:
                    continue
                root = np.sqrt(discriminant)
                lo = max(lo, (-b[i] - root) / (2 * a[i]))
                hi = min(hi, (-b[i] + root) / (2 * a[i]))
            if lo > hi:
                continue
            pieces = []
            for start, end in uncovered[i]:
                if hi < start or lo > end:
                    pieces.append((start, end))
                    continue
                if lo > start + interval_tolerance:
                    pieces.append((start, lo))
                if hi < end - interval_tolerance:
                    pieces.append((hi, end))
            uncovered[i] = pieces

    add(first_index)
    while float(nearest.max()) > radius_m + 1e-10 and len(selected) < limit:
        add(int(np.argmax(nearest)))
    reference_count = len(selected)
    while any(uncovered) and len(selected) < limit:
        amounts = [sum(end - start for start, end in pieces) for pieces in uncovered]
        index = int(np.argmax(amounts))
        if index in selected:
            raise RuntimeError("偏差区间覆盖计算未收敛")
        add(index)
    return selected, {
        "coverage_distance_m": float(radius_m),
        "reference_max_nearest_representative_distance_m": float(nearest.max()),
        "reference_coverage_representative_count": reference_count,
        "bias_coverage_extra_representative_count": len(selected) - reference_count,
        "all_member_valid_bias_intervals_covered": not any(uncovered),
        "representative_max": max_representatives,
        "representative_limit_reached": len(selected) >= limit and bool(any(uncovered)),
        "uncovered_member_count": sum(bool(pieces) for pieces in uncovered),
        "uncovered_member_bias_length_m": float(sum(end - start for pieces in uncovered for start, end in pieces)),
        "bias_interval_m": bounds.tolist(),
        "bias_coverage_rule": "union_of_analytic_quadratic_distance_sublevel_intervals",
        "bias_interval_tolerance_m": interval_tolerance,
        "coverage_scope": "generated_cluster_members_only_not_unsampled_space_or_localization_accuracy",
    }


def cap_observation_representatives(representatives, max_count):
    """每条观测的绕射代表合计上限；先轮流保留各簇中心，再轮流补点。

    簇数本身超过上限时，按原始成员数降序保留，记录全部被删簇。
    镜面代表不计入此预算。截断后不能沿用截断前的完整覆盖声明。
    """
    from collections import defaultdict
    from dataclasses import replace
    if max_count is None:
        return representatives, {"enabled": False}
    if isinstance(max_count, (bool, np.bool_)) or not isinstance(max_count, (int, np.integer)) or max_count < 1:
        raise ValueError("每条观测绕射代表上限必须为正整数或 None")
    observations = defaultdict(lambda: defaultdict(list))
    for rep in representatives:
        if rep.point.has_diffraction:
            observations[rep.point.observation_id][rep.metadata["point_cluster_id"]].append(rep)
    keep_ids, replacements, records = set(), {}, []
    for observation, clusters in sorted(observations.items()):
        ordered = sorted(clusters, key=lambda key: (-len(clusters[key][0].members), key))
        chosen = []
        for rank in range(max(map(len, clusters.values()), default=0)):
            for key in ordered:
                if rank < len(clusters[key]) and len(chosen) < max_count:
                    chosen.append(clusters[key][rank])
            if len(chosen) >= max_count:
                break
        ids = {rep.candidate_id for rep in chosen}
        keep_ids.update(ids)
        pruned_clusters = []
        for key, values in clusters.items():
            kept = [rep for rep in values if rep.candidate_id in ids]
            if not kept:
                pruned_clusters.append(key)
            for rep in kept:
                metadata = {**rep.metadata, "representative_count": len(kept),
                            "representative_count_before_observation_cap": len(values),
                            "observation_representative_max": max_count,
                            "observation_budget_pruned_cluster_members": len(kept) < len(values)}
                if len(kept) < len(values):
                    prior_coverage = {name: metadata.get(name) for name in (
                        "all_member_valid_bias_intervals_covered", "uncovered_member_count",
                        "uncovered_member_bias_length_m", "reference_max_nearest_representative_distance_m")}
                    positions = np.asarray([p.position_m for p in rep.members])
                    selected_positions = np.asarray([r.point.position_m for r in kept])
                    reference_gap = np.linalg.norm(positions[:, None] - selected_positions[None, :], axis=2).min(axis=1).max()
                    metadata.update(all_member_valid_bias_intervals_covered=False,
                                    coverage_revalidated_after_observation_cap=False,
                                    coverage_before_observation_cap=prior_coverage,
                                    reference_max_nearest_representative_distance_m=float(reference_gap),
                                    uncovered_member_count=None, uncovered_member_bias_length_m=None)
                replacements[rep.candidate_id] = replace(rep, metadata=metadata)
        records.append({"observation_id": observation, "before_count": sum(map(len, clusters.values())),
                        "after_count": len(chosen), "max_count": max_count,
                        "dropped_representative_ids": [rep.candidate_id for values in clusters.values()
                                                       for rep in values if rep.candidate_id not in ids],
                        "dropped_cluster_ids": sorted(pruned_clusters)})
    kept = [replacements[rep.candidate_id] if rep.point.has_diffraction else rep
            for rep in representatives if not rep.point.has_diffraction or rep.candidate_id in keep_ids]
    return kept, {"enabled": True, "per_observation_max": max_count,
                  "rule": "cluster_size_descending_then_round_robin_coverage_order", "observations": records}
