"""由公共地图、BS 和原始观测直接构造连续函数；不生成 UE 点云。

各传播类型轮流获得枚举预算，计算量有明确上限。提前停止时保存每种
传播类型尚未搜索的数量，不能把预算截断解释成不存在其它物理解。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import product
import math
from typing import Sequence

import numpy as np

from .diffraction import diffraction_edges, reflection_leg
from .propagation_model import (
    ContinuousObservation, PropagationHypothesis, _make_hypothesis_formula,
    wrap_angle_rad,
)
from .scene import Scene2D


@dataclass(frozen=True)
class HypothesisBank:
    scene: Scene2D
    receiver_m: tuple[float, float]
    observations: tuple[ContinuousObservation, ...]
    hypotheses: tuple[PropagationHypothesis, ...]
    observation_hypothesis_indices: tuple[tuple[int, ...], ...]
    search_report: dict

    def to_dict(self):
        return {
            "schema": "continuous_propagation_hypotheses_v1",
            "receiver_m": list(self.receiver_m),
            "observations": [observation.to_dict() for observation in self.observations],
            "hypotheses": [hypothesis.to_dict() for hypothesis in self.hypotheses],
            "observation_hypothesis_indices": [list(indices) for indices in self.observation_hypothesis_indices],
            "search_report": self.search_report,
            "construction_source": "public_map_and_original_refined_peaks",
        }


def _wall_sequences(wall_ids, order):
    if order == 0:
        yield ()
    elif order == 1:
        for key in wall_ids:
            yield (key,)
    else:
        for sequence in product(wall_ids, repeat=order):
            if all(a != b for a, b in zip(sequence, sequence[1:])):
                yield sequence


def _sequence_count(wall_count, order):
    return 1 if order == 0 else wall_count * max(wall_count - 1, 0) ** (order - 1)


def _family_sequences(wall_ids, edge_ids, before_order, after_order=None):
    if after_order is None:
        for sequence in _wall_sequences(wall_ids, before_order):
            yield tuple(("reflection", key) for key in sequence)
        return
    # Every edge is visited before moving on to the next wall sequence. Reflection
    # sequences on opposite sides are independent, so R(w), D(e), R(w) is retained.
    for after in _wall_sequences(wall_ids, after_order):
        for before in _wall_sequences(wall_ids, before_order):
            for edge_id in edge_ids:
                yield (tuple(("reflection", key) for key in before)
                       + (("diffraction", edge_id),)
                       + tuple(("reflection", key) for key in after))


def _round_robin_unique(groups):
    """各观测依次给出一个新顺序，重复路线只占一份函数预算。"""
    queue, seen = deque(iter(group) for group in groups), set()
    while queue:
        iterator = queue.popleft()
        try:
            value = next(iterator)
        except StopIteration:
            continue
        queue.append(iterator)
        if value not in seen:
            seen.add(value)
            yield value


def _segment_angle_intervals(starts, ends, receiver, angle, lower, upper):
    """有限线段角域与当前观测角窗相交；坐标相对 angle，单位弧度。

    不采样墙面。非退化线段的方向覆盖由两个端点精确确定；若接收点
    落在线段上，保守保留整个角窗。宽角窗跨越分支切口时取并集的包络，
    因此最多多留路线，不会误删可能成立的路线。
    """
    starts, ends = np.asarray(starts, float), np.asarray(ends, float)
    va, vb = starts - receiver, ends - receiver
    aa, ab = np.arctan2(va[:, 1], va[:, 0]), np.arctan2(vb[:, 1], vb[:, 0])
    delta = wrap_angle_rad(ab - aa)
    center = wrap_angle_rad(aa + delta / 2 - angle)
    radius = np.abs(delta) / 2
    segment = ends - starts
    denom = np.maximum(np.sum(segment * segment, axis=1), 1e-30)
    fraction = np.clip(np.sum((receiver - starts) * segment, axis=1) / denom, 0., 1.)
    at_receiver = np.linalg.norm(starts + fraction[:, None] * segment - receiver, axis=1) < 1e-8
    lo, hi = np.full(len(starts), np.inf), np.full(len(starts), -np.inf)
    for shift in (-2 * np.pi, 0., 2 * np.pi):
        left = np.maximum(center - radius + shift - 1e-12, lower)
        right = np.minimum(center + radius + shift + 1e-12, upper)
        valid = left <= right
        lo = np.where(valid, np.minimum(lo, left), lo)
        hi = np.where(valid, np.maximum(hi, right), hi)
    return np.where(at_receiver, lower, lo), np.where(at_receiver, upper, hi)


def _window_gap(lower, upper):
    return np.maximum(lower, np.maximum(-upper, 0.))


def _receiver_window_catalogs(walls, edges, bs, observations, max_order, gate):
    """从 BS 一侧展开有限墙段；输出按观测公平排序的连续角域候选。

    两次反射时，BS→最后墙→经最后墙镜像后的前一墙必须共享同一个
    角度窗口。绕射后缀还要求镜像墙角的精确方向落在这个交集中。
    """
    wall_ids = tuple(sorted(walls))
    starts = np.asarray([walls[key].start for key in wall_ids]).reshape(-1, 2)
    ends = np.asarray([walls[key].end for key in wall_ids]).reshape(-1, 2)
    normals = np.asarray([walls[key].normal for key in wall_ids]).reshape(-1, 2)
    midpoints = (starts + ends) / 2
    edge_ids = tuple(sorted(edges))
    edge_points = np.asarray([edges[key].position_m for key in edge_ids]).reshape(-1, 2)
    per_observation = {order: [] for order in range(max_order + 1)}
    suffixes_per_observation = {order: [] for order in range(max_order + 1)}
    interval_checks = 0
    interval_rejections = 0
    suffix_direction_checks = 0
    suffix_direction_rejections = 0
    for observation in observations:
        angle = observation.aoa_rad
        domains = {0: [((), -gate, gate)]}
        if max_order:
            lower, upper = _segment_angle_intervals(starts, ends, bs, angle, -gate, gate)
            valid = np.flatnonzero(lower <= upper)
            interval_checks += len(wall_ids)
            interval_rejections += len(wall_ids) - len(valid)
            midpoint_error = np.abs(wrap_angle_rad(np.arctan2((midpoints-bs)[:, 1], (midpoints-bs)[:, 0])-angle))
            last_order = sorted(valid, key=lambda i: (
                _window_gap(lower[i], upper[i]), midpoint_error[i],
                np.linalg.norm(midpoints[i]-bs), wall_ids[i]))
            domains[1] = [((wall_ids[i],), float(lower[i]), float(upper[i])) for i in last_order]
            if max_order == 2:
                groups = []
                for last in last_order:
                    normal, origin = normals[last], starts[last]
                    image_starts = starts - 2 * ((starts-origin) @ normal)[:, None] * normal
                    image_ends = ends - 2 * ((ends-origin) @ normal)[:, None] * normal
                    lo, hi = _segment_angle_intervals(image_starts, image_ends, bs, angle,
                                                       lower[last], upper[last])
                    eligible = np.flatnonzero((lo <= hi) & (np.arange(len(wall_ids)) != last))
                    interval_checks += len(wall_ids) - 1
                    interval_rejections += len(wall_ids) - 1 - len(eligible)
                    image_midpoints = (image_starts + image_ends) / 2 - bs
                    error = np.abs(wrap_angle_rad(np.arctan2(image_midpoints[:, 1], image_midpoints[:, 0])-angle))
                    previous_order = sorted(eligible, key=lambda i: (
                        _window_gap(lo[i], hi[i]), error[i], np.linalg.norm(image_midpoints[i]), wall_ids[i]))
                    groups.append([((wall_ids[i], wall_ids[last]), float(lo[i]), float(hi[i]))
                                   for i in previous_order])
                # 每一面可能的 BS 侧最后墙都先获得一次机会。
                domains[2] = list(_round_robin_unique(groups))
        for order in range(max_order + 1):
            per_observation[order].append([sequence for sequence, _, _ in domains[order]])
            edge_groups = []
            for sequence, lower, upper in domains[order]:
                image_edges = edge_points.copy()
                for wall_id in sequence:
                    wall = walls[wall_id]
                    image_edges -= 2 * ((image_edges-wall.start) @ wall.normal)[:, None] * wall.normal
                vectors = image_edges - bs
                lengths = np.linalg.norm(vectors, axis=1)
                relative_angles = wrap_angle_rad(np.arctan2(vectors[:, 1], vectors[:, 0])-angle)
                eligible = np.flatnonzero((relative_angles >= lower-1e-12)
                                         & (relative_angles <= upper+1e-12) & (lengths > 1e-7))
                suffix_direction_checks += len(edge_ids)
                suffix_direction_rejections += len(edge_ids) - len(eligible)
                eligible = sorted(eligible, key=lambda i: (abs(relative_angles[i]), lengths[i], edge_ids[i]))
                edge_groups.append([(edge_ids[i], sequence) for i in eligible])
            suffixes_per_observation[order].append(list(_round_robin_unique(edge_groups)))
    specular = {order: list(_round_robin_unique(groups)) for order, groups in per_observation.items()}
    suffixes = {order: list(_round_robin_unique(groups)) for order, groups in suffixes_per_observation.items()}
    specular_memberships = {}
    for groups in per_observation.values():
        for observation_index, sequences in enumerate(groups):
            for sequence in sequences:
                specular_memberships.setdefault(sequence, set()).add(observation_index)
    report = {"specular_window_counts_per_observation": {
        str(order): [len(group) for group in groups] for order, groups in per_observation.items()},
        "fixed_diffraction_suffix_counts_per_observation": {
            str(order): [len(group) for group in groups] for order, groups in suffixes_per_observation.items()},
        "specular_unique_window_counts": {str(order): len(group) for order, group in specular.items()},
        "fixed_diffraction_unique_suffix_counts": {str(order): len(group) for order, group in suffixes.items()},
        "condition": "common_angle_interval_of_finite_unfolded_segments",
        "finite_segment_interval_checks": interval_checks,
        "finite_segment_interval_rejections": interval_rejections,
        "fixed_edge_direction_checks": suffix_direction_checks,
        "fixed_edge_direction_rejections": suffix_direction_rejections,
        "calculation": "batched_endpoint_interval_arithmetic_per_observation_and_bs_side_wall",
        "uses_position_sampling": False}
    return specular, suffixes, specular_memberships, report


def _diagonal_prefix_suffix_sequences(wall_ids, before_order, suffixes):
    """两端路线交替扩展，避免先遍历完某一个墙角或某一面墙。"""
    prefix_count = _sequence_count(len(wall_ids), before_order)
    if not prefix_count or not suffixes:
        return
    iterator = iter(_wall_sequences(wall_ids, before_order))
    prefixes = []
    for diagonal in range(prefix_count + len(suffixes) - 1):
        lo = max(0, diagonal - prefix_count + 1)
        hi = min(len(suffixes)-1, diagonal)
        for suffix_index in range(hi, lo-1, -1):
            prefix_index = diagonal - suffix_index
            while len(prefixes) <= prefix_index:
                prefixes.append(next(iterator))
            edge_id, after = suffixes[suffix_index]
            yield (tuple(("reflection", key) for key in prefixes[prefix_index])
                   + (("diffraction", edge_id),)
                   + tuple(("reflection", key) for key in after))


def _compatible_observations(hypothesis, scene, observations, beta_interval_m,
                             aoa_gate_rad, length_gate_sigma):
    x_min, x_max, y_min, y_max = scene.bounds_m
    lower = np.asarray([x_min, y_min])
    upper = np.asarray([x_max, y_max])
    anchor = hypothesis.equivalent_anchor_m
    nearest = np.maximum(lower - anchor, np.maximum(anchor - upper, 0.0))
    min_length = float(np.linalg.norm(nearest)) + hypothesis.fixed_length_m
    farthest = np.maximum(np.abs(lower - anchor), np.abs(upper - anchor))
    max_length = float(np.linalg.norm(farthest)) + hypothesis.fixed_length_m
    center = (lower + upper) / 2
    matrix = np.asarray(hypothesis.affine_image_matrix)
    vector = matrix @ (center - anchor)
    center_distance = float(np.linalg.norm(vector))
    box_radius = float(np.linalg.norm((upper - lower) / 2))
    angular_center = math.atan2(vector[1], vector[0])
    angular_radius = (math.pi if center_distance <= box_radius else
                      math.asin(min(1.0, box_radius / center_distance)))
    matches = []
    for index, observation in enumerate(observations):
        if aoa_gate_rad is not None:
            center_angle = hypothesis.fixed_aoa_rad
            radius = 0.0
            if center_angle is None:
                center_angle = angular_center
                radius = angular_radius
            difference = abs(float(wrap_angle_rad(observation.aoa_rad - center_angle)))
            if difference > radius + aoa_gate_rad + 1e-12:
                continue
        if beta_interval_m is not None:
            observed = observation.observed_length_m
            margin = length_gate_sigma * observation.length_scale_m
            if (min_length + beta_interval_m[0] > observed + margin
                    or max_length + beta_interval_m[1] < observed - margin):
                continue
        matches.append(index)
    return tuple(matches)


def build_hypothesis_bank(
    scene: Scene2D,
    bs_position_m: Sequence[float],
    observations: Sequence[ContinuousObservation],
    *,
    max_reflections: int = 2,
    max_diffractions: int = 1,
    diffraction_position: str = "any",
    max_hypotheses: int = 4096,
    max_enumerated_sequences: int = 50000,
    aoa_gate_rad: float | None = None,
    beta_interval_m: Sequence[float] | None = None,
    length_gate_sigma: float = 5.0,
) -> HypothesisBank:
    """在声明的传播阶数内直接建模；搜索上限与遗漏量一并返回。

    角度门限是显式工程范围，不是噪声统计保证。范围筛选使用整个地图
    的距离下界和上界，不用已知 UE、注入噪声、旧候选点或代表方向。
    """
    from .path_policy import validate_diffraction_position
    validate_diffraction_position(diffraction_position)
    if (isinstance(max_reflections, bool) or not isinstance(max_reflections, int)
            or max_reflections not in (0, 1, 2)):
        raise ValueError("连续模型支持 0、1 或 2 次反射")
    if (isinstance(max_diffractions, bool) or not isinstance(max_diffractions, int)
            or max_diffractions not in (0, 1)):
        raise ValueError("连续模型支持 0 或 1 次绕射")
    for name, value in (("max_hypotheses", max_hypotheses),
                        ("max_enumerated_sequences", max_enumerated_sequences)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} 必须是正整数")
    if aoa_gate_rad is not None and (not math.isfinite(float(aoa_gate_rad))
                                     or not 0 < aoa_gate_rad <= math.pi):
        raise ValueError("角度筛选范围必须位于 (0, pi]")
    if not math.isfinite(float(length_gate_sigma)) or length_gate_sigma < 0:
        raise ValueError("长度筛选的误差倍数必须是非负有限数")
    bs = np.asarray(bs_position_m, float)
    if bs.shape != (2,) or not np.all(np.isfinite(bs)) or not scene.contains(bs):
        raise ValueError("BS 必须是地图内的二维有限坐标")
    observations = tuple(observations)
    if not observations or not all(isinstance(item, ContinuousObservation) for item in observations):
        raise ValueError("至少需要一条 ContinuousObservation 观测")
    if len({item.observation_id for item in observations}) != len(observations):
        raise ValueError("每条原始观测必须有独立且唯一的编号")
    if beta_interval_m is not None:
        interval = np.asarray(beta_interval_m, float)
        if interval.shape != (2,) or not np.all(np.isfinite(interval)) or interval[0] > interval[1]:
            raise ValueError("公共偏差范围必须是递增的两个有限数，单位米")
        beta_interval_m = tuple(float(value) for value in interval)

    walls = {wall.wall_id: wall for wall in scene.walls}
    if len(walls) != len(scene.walls):
        raise ValueError("公共地图的墙编号必须唯一")
    edges = ({edge.edge_id: edge for edge in diffraction_edges(scene)}
             if max_diffractions else {})

    def bearing_rank(point):
        vector = np.asarray(point) - bs
        bearing = math.atan2(vector[1], vector[0])
        difference = min(abs(float(wrap_angle_rad(bearing - observation.aoa_rad)))
                         for observation in observations)
        return difference, float(np.linalg.norm(vector))

    wall_ids = tuple(sorted(walls, key=lambda key: (*bearing_rank((walls[key].start + walls[key].end) / 2), key)))
    edge_ids = tuple(sorted(edges, key=lambda key: (*bearing_rank(edges[key].position_m), key)))
    catalogs = None
    window_report = {"enabled": False, "finite_segment_interval_checks": 0,
                     "fixed_edge_direction_checks": 0}
    if aoa_gate_rad is not None:
        import time
        began = time.perf_counter()
        specular_windows, suffix_windows, specular_memberships, window_report = _receiver_window_catalogs(
            walls, edges, bs, observations, max_reflections, aoa_gate_rad)
        window_report.update(enabled=True, elapsed_seconds=time.perf_counter()-began)
        catalogs = (specular_windows, suffix_windows)
    families = []
    queue = deque()
    for order in range(max_reflections + 1):
        key = f"reflection_{order}_diffraction_0"
        count = _sequence_count(len(walls), order)
        admitted = count if catalogs is None else len(specular_windows[order])
        family = {"family": key, "total_sequences": count, "enumerated_sequences": 0,
                  "retained_hypotheses": 0, "excluded_by_observation_bounds": 0,
                  "excluded_by_fixed_geometry": 0, "angle_admissible_sequences": admitted,
                  "excluded_by_receiver_angle_domain": count-admitted}
        families.append(family)
        if admitted:
            iterator = (_family_sequences(wall_ids, edge_ids, order) if catalogs is None else
                        (tuple(("reflection", key) for key in sequence) for sequence in specular_windows[order]))
            queue.append((family, iter(iterator)))
        if max_diffractions:
            for before_order in range(order + 1):
                # 内部顺序为 UE→BS；BS→UE 的末次绕射对应这里的首次交互。
                if diffraction_position == "last_from_bs" and before_order != 0:
                    continue
                after_order = order - before_order
                count = (len(edges) * _sequence_count(len(walls), before_order)
                         * _sequence_count(len(walls), after_order))
                admitted = count if catalogs is None else (
                    len(suffix_windows[after_order]) * _sequence_count(len(walls), before_order))
                family = {"family": f"reflection_{before_order}_diffraction_1_reflection_{after_order}",
                          "total_sequences": count, "enumerated_sequences": 0,
                          "retained_hypotheses": 0, "excluded_by_observation_bounds": 0,
                          "excluded_by_fixed_geometry": 0, "angle_admissible_sequences": admitted,
                          "excluded_by_receiver_angle_domain": count-admitted}
                families.append(family)
                if admitted:
                    iterator = (_family_sequences(wall_ids, edge_ids, before_order, after_order)
                                if catalogs is None else _diagonal_prefix_suffix_sequences(
                                    wall_ids, before_order, suffix_windows[after_order]))
                    queue.append((family, iter(iterator)))
    hypotheses = []
    memberships = [[] for _ in observations]
    suffix_validity = {}
    attempts = 0
    while queue and attempts < max_enumerated_sequences and len(hypotheses) < max_hypotheses:
        family, iterator = queue.popleft()
        try:
            interactions = next(iterator)
        except StopIteration:
            continue
        queue.append((family, iterator))
        family["enumerated_sequences"] += 1
        attempts += 1
        try:
            hypothesis = _make_hypothesis_formula(scene, bs, interactions, walls, edges,
                                                  validate_fixed_leg=False)
        except ValueError:
            family["excluded_by_fixed_geometry"] += 1
            continue
        matches = _compatible_observations(hypothesis, scene, observations, beta_interval_m,
                                           aoa_gate_rad, length_gate_sigma)
        if catalogs is not None and not hypothesis.diffraction_order:
            allowed = specular_memberships[tuple(key for _, key in interactions)]
            matches = tuple(index for index in matches if index in allowed)
        if not matches:
            family["excluded_by_observation_bounds"] += 1
            continue
        if hypothesis.diffraction_order:
            index = next(i for i, (kind, _) in enumerate(interactions) if kind == "diffraction")
            edge = edges[interactions[index][1]]
            after = tuple(key for _, key in interactions[index + 1:])
            suffix_key = (edge.edge_id, after)
            if suffix_key not in suffix_validity:
                suffix_validity[suffix_key] = reflection_leg(
                    scene, np.asarray(edge.position_m), bs, after,
                    source_walls=edge.incident_wall_ids) is not None
            if not suffix_validity[suffix_key]:
                family["excluded_by_fixed_geometry"] += 1
                continue
        hypothesis_index = len(hypotheses)
        hypotheses.append(hypothesis)
        family["retained_hypotheses"] += 1
        for observation_index in matches:
            memberships[observation_index].append(hypothesis_index)

    for family in families:
        family["unsearched_sequences"] = family["angle_admissible_sequences"] - family["enumerated_sequences"]
    total = sum(family["total_sequences"] for family in families)
    pruned = sum(family["excluded_by_receiver_angle_domain"] for family in families)
    unsearched = total - pruned - attempts
    stop_reason = ("complete" if not unsearched else "max_hypotheses" if len(hypotheses) >= max_hypotheses
                   else "max_enumerated_sequences")
    report = {
        "max_reflections": max_reflections, "max_diffractions": max_diffractions,
        "diffraction_position": diffraction_position,
        "max_hypotheses": max_hypotheses, "max_enumerated_sequences": max_enumerated_sequences,
        "aoa_gate_rad": aoa_gate_rad, "beta_interval_m": beta_interval_m,
        "length_gate_sigma": length_gate_sigma,
        "wall_count": len(walls), "edge_count": len(edges),
        "total_sequences": total, "enumerated_sequences": attempts,
        "retained_hypotheses": len(hypotheses), "unsearched_sequences": unsearched,
        "excluded_by_receiver_angle_domain": pruned,
        "angle_admissible_sequences": total-pruned,
        "receiver_angle_preselection": window_report,
        "enumeration_budget_scope": "continuous_function_constructions_after_separately_counted_angle_preselection",
        "complete_within_configured_orders": unsearched == 0,
        "budget_exhausted": unsearched > 0, "stop_reason": stop_reason,
        "enumeration_policy": ("round_robin_families_and_observations_with_finite_unfolded_wall_windows"
                               if catalogs is not None else "round_robin_families_without_angle_preselection"),
        "unsearched_families": [dict(family) for family in families if family["unsearched_sequences"]],
        "families": families,
        "excluded_by_observation_bounds": sum(f["excluded_by_observation_bounds"] for f in families),
        "excluded_by_fixed_geometry": sum(f["excluded_by_fixed_geometry"] for f in families),
        "observations_without_hypotheses": [observation.observation_id for observation, indices
                                            in zip(observations, memberships) if not indices],
        "domain_validation": "rebuild_at_every_proposed_and_final_ue_position",
        "observational_gate_is_statistical_guarantee": False,
    }
    return HypothesisBank(scene, tuple(float(value) for value in bs), observations,
                          tuple(hypotheses), tuple(tuple(indices) for indices in memberships), report)
