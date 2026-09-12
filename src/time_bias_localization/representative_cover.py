"""点簇压缩：先覆盖参考位置，再补足整个合法偏差区间。

只对聚类后的成员做覆盖计算，不提前建立求解器轨迹。两成员间的位置差
关于 beta 是一次式；距离阈值内的区间由二次不等式解析求出，再取区间并集。
因此不会把有限几个偏差检查点误称为整个区间的覆盖保证。
"""
from __future__ import annotations

import numpy as np
from .constants import SPEED_OF_LIGHT_M_S


def select_cover_members(members, first_index, radius_m, beta_interval_m):
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
    while float(nearest.max()) > radius_m + 1e-10:
        add(int(np.argmax(nearest)))
    reference_count = len(selected)
    while any(uncovered):
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
        "all_member_valid_bias_intervals_covered": True,
        "bias_interval_m": bounds.tolist(),
        "bias_coverage_rule": "union_of_analytic_quadratic_distance_sublevel_intervals",
        "bias_interval_tolerance_m": interval_tolerance,
        "coverage_scope": "generated_cluster_members_only_not_unsampled_space_or_localization_accuracy",
    }
